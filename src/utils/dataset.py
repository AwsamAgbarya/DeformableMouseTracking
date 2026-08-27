import torch
import pandas as pd
import numpy as np
from utils.normalizers import Normalizer3D, NormalizerMV
from utils.pose_extraction import extract_poses
from utils.geometry import project_points

DEFAULT_PART_NAMES = [
    "base",             # 0
    "spine_base",       # 1
    "spine_mid",        # 2
    "spine_front",      # 3
    "neck",             # 4
    "head",             # 5
    "nose",             # 6
    "ear_l",            # 7
    "ear_r",            # 8
    "shoulder_l",       # 9
    "elbow_l",          # 10
    "wrist_l",          # 11
    "shoulder_r",       # 12
    "elbow_r",          # 13
    "wrist_r",          # 14
    "hip_l",            # 15
    "knee_l",           # 16
    "ankle_l",          # 17
    "hindpaw_l",        # 18
    "hip_r",            # 19
    "knee_r",           # 20
    "ankle_r",          # 21
    "hindpaw_r",        # 22
    "tail_base",        # 23
    "tail_mid",         # 24
    "tail_end",         # 25
]

# BASE INDEX EXCLUDED
DEFAULT_RIGID_EDGES = [
    [1, 0], # spine mid → spine base
    [1, 2], # spine mid → spine front
    [2, 3], # spine front → neck
    [3, 4], # neck → head
    [4, 5], # head → nose
    [4, 6], # head → ear_l
    [4, 7], # head → ear_r
    [3, 8], # spine front → shoulder_l
    [8, 9], # shoulder_l → elbow_l
    [9, 10], # elbow_l → wrist_l
    [3, 11], # spine front → shoulder_r
    [11, 12], # shoulder_r → elbow_r
    [12, 13], # elbow_r → wrist_r
    [1, 14], # spine base → hip_l
    [14, 15], # hip_l → knee_l
    [15, 16], # knee_l → ankle_l
    [16, 17], # ankle_l → hindpaw_l
    [1, 18], # spine base → hip_r
    [18, 19], # hip_r → knee_r
    [19, 20], # knee_r → ankle_r
    [20, 21], # ankle_r → hindpaw_r
    [0, 22], # spine base → tail_base
    [22, 23], # tail_base → tail_mid
    [23, 24], # tail_mid → tail_end
]

def finite_diff(x):
    out = torch.zeros_like(x)
    out[1:] = x[1:] - x[:-1]
    return out

def center_and_rotate(deform_data, forward_vec):
    """
    Center and rotate the pose data so that the forward vector aligns with the x-axis.
    """
    centered_data = deform_data - deform_data[:, 0][:, None, :]
    theta = torch.arctan2(forward_vec[:, 1], forward_vec[:, 0])
    c, s = torch.cos(-theta), torch.sin(-theta)
    gx, gy = centered_data[..., 0].clone(), centered_data[..., 1].clone()
    centered_data[..., 0] = c[:, None] * gx - s[:, None] * gy
    centered_data[..., 1] = s[:, None] * gx + c[:, None] * gy
    return centered_data
    
class RawDataset:
    def __init__(self, file_path, reorder_parts = True, part_names = DEFAULT_PART_NAMES, config=None):
        raw = pd.read_csv(file_path).drop(columns=['frame', 'joint_id'])
        self.part_names = [part for part in raw['part'].unique()]
        self.N = len(self.part_names) - 1

        # Sort properly
        if reorder_parts:
            part_order = {p: i for i, p in enumerate(part_names)}
            raw["_ord"] = raw["part"].map(part_order)
            order = np.argsort(raw["_ord"].to_numpy().reshape(-1, self.N), axis=1)
            raw = raw.sort_values(["time", "_ord"]).drop(columns=["_ord"])
            self.part_names = part_names

        self.file_path = file_path
        self.time = raw['time'].values
        self.parts = raw['part'].values

        self.rigid_g  = torch.tensor(raw[['x_r', 'y_r', 'z_r']].to_numpy(), dtype=torch.float32).reshape(-1, self.N+1, 3)[:, 1:]
        deform_raw    = torch.tensor(raw[['x_d', 'y_d', 'z_d']].to_numpy(), dtype=torch.float32).reshape(-1, self.N+1, 3)
        self.deform_g = deform_raw[:, 1:]
        self.base     = deform_raw[:, 0]
        self.local_g  = torch.tensor(raw[['x_l', 'y_l', 'z_l']].to_numpy(), dtype=torch.float32).reshape(-1, self.N+1, 3)[:, 1:]

        # Local = Centered around base, Unrotated, Not normalized
        # Deform - base = Centered around base, rotated, Not normalized
        # Deform - Rigid = Relative distance, Unrotated, Not normalized

        self.T = self.deform_g.shape[0]
        self.config = config


class ProjectionDataset(RawDataset):
    """
    Designed for Multiview projected 2d data
    """
    def __init__(self, file_path, projections, reorder_parts = True, part_names = DEFAULT_PART_NAMES, reference_indices=[8, 11], n_rotations=10, config=None):
        super().__init__(file_path, reorder_parts, part_names, config)
        normalize = self.config.get("normalize", False)
        self.mask_ratio = 0.0
        self.views = {k:view for k,view in enumerate(projections)}
        self.view_count = len(projections)

        unique_poses, augmented_poses = extract_poses(self.local_g, reference_indices=reference_indices, n_rotations=n_rotations)

        traj_list_d = []
        depth_list = []
        # Project to 2D (unnormalized)
        for i, view in enumerate(projections):
            proj_data_d, depths_d = project_points(augmented_poses, view)
            traj_list_d.append(proj_data_d[:, None, ...])
            depth_list.append(depths_d[:, None, :])
        
        self.coords = torch.concatenate(traj_list_d, dim=1)
        self.depths = torch.concatenate(depth_list, dim=1)
            
        if normalize:
            load_norm_path = self.config.get("load_norm_path", None)
            save_norm_path = self.config.get("save_norm_path", None)
            # Initialize normalizer
            self.normalizer = NormalizerMV()
            
            if load_norm_path is not None:
                print(f"Loading normalizer from {load_norm_path}")
                self.normalizer.load(load_norm_path)
            else:
                print(f"New normalizer initialized")

            if not self.normalizer.is_fitted:
                print("Fitting normalizer on 2D data...")
                self.normalizer.fit(self.coords, self.depths)
                if save_norm_path is not None:
                    self.normalizer.save(save_norm_path)
            
            self.coords = self.normalizer.normalize(self.coords)  
            self.depths = self.normalizer.normalize_depth(self.depths)  

class SnapshotDataset(RawDataset):
    """
    Designed for a SnapshotModel 3D Singular view
    Introduces Data augmentation by Augmenting unique poses with random rotations 
    """
    def __init__(self, file_path, reorder_parts = True, part_names = DEFAULT_PART_NAMES, n_rotations=10, reference_indices=[8, 11], config=None):
        super().__init__(file_path, reorder_parts, part_names, config)
        normalize = self.config.get("normalize", False)
        self.unique_poses, self.augmented_poses = extract_poses(self.local_g, reference_indices=reference_indices, n_rotations=n_rotations)
        
        if normalize:
            load_norm_path = self.config.get("load_norm_path", None)
            save_norm_path = self.config.get("save_norm_path", None)
            # Initialize normalizer
            self.normalizer = Normalizer3D()
            if load_norm_path is not None:
                print(f"Loading normalizer from {load_norm_path}")
                self.normalizer.load(load_norm_path)
            else:
                print(f"New normalizer initialized")

            if not self.normalizer.is_fitted:
                print("Fitting normalizer on 2D data...")
                self.normalizer.fit(self.augmented_poses)
                if save_norm_path is not None:
                    self.normalizer.save(save_norm_path)
            self.data = self.normalizer.normalize(self.augmented_poses, center=True)
        else:
            self.data = self.augmented_poses


class TemporalDataset(RawDataset):
    """
    Designed for a Temporal Model 3D
    Main task (data_d): Centered + Unrotated coordinates (T, N, 3)
    Auxillary task (relative_dist_d): Relative distances between rigid and deformable coordinates (T, N, 3)
    """
    def __init__(self, file_path, reorder_parts = True, part_names = DEFAULT_PART_NAMES, reference_edge=(9,12), edges=DEFAULT_RIGID_EDGES, config=None):
        super().__init__(file_path, reorder_parts, part_names, config)
        
        # Relative distances
        self.relative_dist = self.deform_g - self.rigid_g        # (T, N, 3)
        self.d2r_bond_lengths = self.relative_dist.norm(dim=-1)  # (T, N)
        self.d2r_bond_mean = self.d2r_bond_lengths.mean(dim=0)   # (N,)
        self.d2r_bond_std  = self.d2r_bond_lengths.std(dim=0)    # (N,)

        # Build Adjacency Matrix
        self.edges = torch.tensor(edges)
        self.adj = torch.zeros(self.N, self.N, dtype=torch.uint8)
        self.adj[self.edges[:, 0], self.edges[:, 1]] = 1.0
        self.parent = [-1] * self.N
        for p, c in self.edges.tolist():
            self.parent[c] = p
        self.parent[0] = -1

        # Normalization
        normalize = self.config.get("normalize", False)
        if normalize:
            load_norm_path = self.config.get("load_norm_path", None)
            save_norm_path = self.config.get("save_norm_path", None)
            # Initialize normalizer
            self.normalizer = Normalizer3D()
            self.relative_dist_normalizer = Normalizer3D()
            if load_norm_path is not None:
                print(f"Loading normalizer from {load_norm_path}")
                self.normalizer.load(load_norm_path)
            else:
                print(f"New normalizer initialized")

            if not self.normalizer.is_fitted:
                print("Fitting normalizer on 2D data...")
                self.normalizer.fit(self.local_g)
                self.relative_dist_normalizer.fit(self.relative_dist)
                if save_norm_path is not None:
                    self.normalizer.save(save_norm_path)
            
            self.data_d = self.normalizer.normalize(self.local_g, center=False)
            self.relative_dist_d = self.relative_dist_normalizer.normalize(self.relative_dist, center=False)

            # Bone edges
            lengths =[]
            self.mean_deform_norm = self.data_d.mean(dim=0) # (N, 3)
            for p, c in self.edges:
                lengths.append(torch.norm(self.mean_deform_norm[c] - self.mean_deform_norm[p]).item())
            self.d2d_bone_lengths = torch.tensor(lengths, dtype=torch.float32)

        else:
            self.data_d = self.local_g.clone()
            self.relative_dist_d = self.relative_dist