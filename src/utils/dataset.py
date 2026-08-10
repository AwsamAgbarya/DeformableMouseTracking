import torch
import pandas as pd
import numpy as np
from utils.normalizers import Normalizer3D, NormalizerMV
from utils.pose_extraction import extract_poses

DEFAULT_PART_NAMES = [
    "head center",      # 0
    "tail base",        # 1
    "nose",             # 2
    "right ear",        # 3
    "left ear",         # 4
    "tail root",        # 5
    "left front paw",   # 6
    "right front paw",  # 7
    "left hind paw",    # 8
    "right hind paw",   # 9
]

DEFAULT_RIGID_EDGES = [
    [0, 1],   # head center → tail base
    [0, 2],   # head center → nose
    [0, 3],   # head center → right ear
    [0, 4],   # head center → left ear
    [1, 5],   # tail base   → tail root
    [1, 8],   # tail base   → left hind paw
    [1, 9],   # tail base   → right hind paw
    [0, 6],   # head center → left front paw
    [0, 7],   # head center → right front paw
]

DEFAULT_PART_GROUPS = {
    "head":      [0, 2, 3, 4],   # head, nose, ears
    "tail":      [1, 5],      # tail
    "paw_FL":    [6],
    "paw_FR":    [7],
    "paw_BL":    [8],
    "paw_BR":    [9],
}

def finite_diff(x):
    out = torch.zeros_like(x)
    out[1:] = x[1:] - x[:-1]
    return out

def flag_outlier_frames(rigid_coords, part_names, reference_edge=(0, 1), window=5, thresh_deg=90.0):
    """
    Flag frames where the reference bone undergoes an implausibly sharp direction change, indicating a tracking swap or identity flip.

    Returns
    -------
    outlier_mask  : BoolTensor (T,)        True = outlier frame
    angles        : Tensor (T,)            smoothed per-frame direction-change angle
    report        : str                    human-readable summary
    """
    p_idx, c_idx = reference_edge
    T = rigid_coords.shape[0]

    # Unit direction vector of the bone at every frame: (T, 3)
    bone_vec = rigid_coords[:, c_idx, :] - rigid_coords[:, p_idx, :]
    norms    = bone_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_vec = bone_vec / norms

    # Angle between consecutive frames (dot product → degrees)
    cos_sim  = (unit_vec[1:] * unit_vec[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
    raw_angles = torch.acos(cos_sim) * (180.0 / torch.pi)
    raw_angles = torch.cat([torch.zeros(1), raw_angles])
    outlier_mask = raw_angles > thresh_deg
 
    p_name   = part_names[p_idx]
    c_name   = part_names[c_idx]
    n_out    = outlier_mask.sum().item()
    out_idxs = outlier_mask.nonzero(as_tuple=True)[0].tolist()
    out_preview = out_idxs[:10]
    preview_str = ", ".join(str(f) for f in out_preview)
    if len(out_idxs) > 10:
        preview_str += f" ... (+{len(out_idxs)-10} more)"
    report = (
        f"Reference bone : {p_name} → {c_name}\n"
        f"Threshold       : {thresh_deg}°  |  smoothing window: {window}\n"
        f"Total frames    : {T}\n"
        f"Outlier frames  : {n_out}  ({100*n_out/T:.1f}%)\n"
        f"Frame indices   : [{preview_str}]\n"
        f"Max Outlier angle seen  : {raw_angles.max().item():.1f}°  (frame {raw_angles.argmax().item()})\n"
        f"Min Outlier angle seen  : {raw_angles[outlier_mask].min().item():.1f}°  (frame {raw_angles[outlier_mask].argmin().item()})\n"
    )
    print(report)
    return outlier_mask

class RawDataset:
    def __init__(self, file_path, reorder_parts = True, part_names = DEFAULT_PART_NAMES, poses=True, n_rotations=18, reference_edge=["head center", "tail base"], config=None):
        raw = pd.read_csv(file_path)
        self.part_names = [part for part in raw['part'].unique()]
        self.N = len(self.part_names)

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
        self.rigid_g = torch.tensor(raw[['x_r', 'y_r', 'z_r']].to_numpy(), dtype=torch.float32).reshape(-1, self.N, 3)
        self.deform_g = torch.tensor(raw[['x_d', 'y_d', 'z_d']].to_numpy(), dtype=torch.float32).reshape(-1, self.N, 3)
        if poses: 
            self.unique_poses, self.com, self.augmented_poses = extract_poses(self.rigid_g, self.deform_g, self.part_names, reference_parts=reference_edge, n_rotations=n_rotations)
        self.T = self.deform_g.shape[0]
        self.config = config


class SnapshotDataset(RawDataset):
    """
    Designed for a SnapshotModel 3D Singular view
    Introduces Data augmentation by Augmenting unique poses with random rotations 
    """
    def __init__(self, file_path, reorder_parts = True, part_names = DEFAULT_PART_NAMES, extract_poses=True, n_rotations=18, reference_parts=["head center", "tail base"], config=None):
        super().__init__(file_path, reorder_parts, part_names, extract_poses, n_rotations, reference_parts, config)
        validate = self.config.get("validate", False)
        normalize = self.config.get("normalize", False)
        
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

        if validate:
            dataset_size = len(self.data)
            indices = list(range(dataset_size))
            np.random.shuffle(indices)
            split = int(np.floor((1-self.config['train_ratio']) * dataset_size))
            self.train_indices, self.val_indices = indices[:split], indices[split:]
        else:
            self.train_indices = list(range(len(self.data)))
            self.val_indices = []


class TemporalDataset(RawDataset):
    """
    Designed for a Temporal Model 3D
    """
    def __init__(self, file_path, reorder_parts = True, part_names = DEFAULT_PART_NAMES, extract_poses=False, n_rotations=0, reference_edge=(0,1), edges=DEFAULT_RIGID_EDGES, config=None):
        super().__init__(file_path, reorder_parts, part_names, extract_poses, n_rotations, reference_edge, config)
        
        # COM-centred node positions
        self.com = self.rigid_g.mean(dim=1)
        rigid_c  = self.rigid_g - self.com.unsqueeze(1)
        deform_c = self.deform_g - self.com.unsqueeze(1)

        # Global & per-node body motion
        self.global_vel   = finite_diff(self.com) # COM Velocity
        self.global_speed = self.global_vel.norm(dim=-1, keepdim=True) # COM Speed
        self.global_accel = finite_diff(self.global_vel) # COM Acceleration
        self.node_vel     = finite_diff(deform_c) # Node Velocity
        self.node_speed   = self.node_vel.norm(dim=-1, keepdim=True) # Node Speed
        self.node_accel   = finite_diff(self.node_vel) # Node Acceleration

        # Translation invariance + Rigid information extraction
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

        self.outlier_mask = flag_outlier_frames(rigid_c, part_names, reference_edge = reference_edge, window = self.config.get("outlier_window", 5), thresh_deg = self.config.get("outlier_thresh", 90.0))


        # Normalization
        normalize = self.config.get("normalize", False)
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
                self.normalizer.fit(deform_c)
                if save_norm_path is not None:
                    self.normalizer.save(save_norm_path)
            
            self.data_d = self.normalizer.normalize(deform_c, center=False)
            self.data_r = self.normalizer.normalize(rigid_c, center=False)

            # Bone edges
            lengths =[]
            self.mean_deform_norm = self.data_d.mean(dim=0) # (N, 3)
            for p, c in self.edges:
                lengths.append(torch.norm(self.mean_deform_norm[c] - self.mean_deform_norm[p]).item())
            self.d2d_bone_lengths = torch.tensor(lengths, dtype=torch.float32)

        else:
            self.data_d = deform_c
            self.data_r = rigid_c