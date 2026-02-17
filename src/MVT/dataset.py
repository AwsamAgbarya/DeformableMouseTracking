import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.geometry import project_points

class DataDrivenNormalization:
    """
    Normalize 2D coordinates to [-1, 1] while preserving aspect ratio.
    Uses the largest dimension's range to scale both X and Y equally.
    """
    
    def __init__(self):
        # 2D normalization
        self.center_x = None
        self.center_y = None
        self.scale = None
        
        # Depth normalization
        self.depth_min = None
        self.depth_max = None
        self.is_fitted = False
    
    def fit(self, keypoints_2d, depths):
        """
        Compute normalization bounds using the maximum range across X and Y.
        """
        if isinstance(keypoints_2d, torch.Tensor):
            keypoints_2d = keypoints_2d.detach().cpu().numpy()
        
        # Find Center
        x_min, x_max = keypoints_2d[..., 0].min(), keypoints_2d[..., 0].max()
        y_min, y_max = keypoints_2d[..., 1].min(), keypoints_2d[..., 1].max()
        
        self.center_x = (x_max + x_min) / 2.0
        self.center_y = (y_max + y_min) / 2.0
        
        # Range
        range_x = x_max - x_min
        range_y = y_max - y_min
        
        # Isotropic scaling
        max_range = max(range_x, range_y)
        margin = 0.05
        self.scale = (max_range * (1 + margin)) / 2.0 

        if depths is not None:
            if isinstance(depths, torch.Tensor):
                depths = depths.detach().cpu().numpy()
            self.depth_min = depths.min()
            self.depth_max = depths.max()
            
            # Add margin to depth
            d_range = self.depth_max - self.depth_min
            self.depth_min -= d_range * margin
            self.depth_max += d_range * margin
        
        self.is_fitted = True
        print(f"✓ Isotropic Scale: {self.scale:.4f}")
        return self
    
    def normalize(self, keypoints_2d):
        """Normalize 2D keypoints to [-1, 1]"""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted! Call fit() first.")
        kpts_2d = keypoints_2d.clone() if isinstance(keypoints_2d, torch.Tensor) else keypoints_2d.copy()
        
        # Normalize x and y
        kpts_2d[..., 0] = (kpts_2d[..., 0] - self.center_x) / self.scale
        kpts_2d[..., 1] = (kpts_2d[..., 1] - self.center_y) / self.scale
        
        return kpts_2d
    
    def denormalize(self, keypoints_2d_norm):
        """Denormalize from [-1, 1] back to original scale"""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted! Call fit() first.")
        kpts_2d = keypoints_2d_norm.clone() if isinstance(keypoints_2d_norm, torch.Tensor) else keypoints_2d_norm.copy()
        
        # Denormalize x and y
        kpts_2d[..., 0] = keypoints_2d_norm[..., 0] * self.scale + self.center_x
        kpts_2d[..., 1] = keypoints_2d_norm[..., 1] * self.scale + self.center_y
        
        return kpts_2d
    
    def normalize_depth(self, depths):
        """Normalize depths to [-1, 1]"""
        if self.depth_min is None or self.depth_max is None:
            raise ValueError("Depth normalization not fitted!")
        
        is_torch = isinstance(depths, torch.Tensor)
        depths_norm = depths.clone() if is_torch else depths.copy()
        
        depths_norm = 2 * (depths - self.depth_min) / (self.depth_max - self.depth_min) - 1
        return depths_norm
    
    def denormalize_depth(self, depths_norm):
        """Denormalize depths from [-1, 1] to original scale"""
        if self.depth_min is None or self.depth_max is None:
            raise ValueError("Depth normalization not fitted!")
        
        is_torch = isinstance(depths_norm, torch.Tensor)
        depths = depths_norm.clone() if is_torch else depths_norm.copy()
        
        depths = (depths_norm + 1) / 2 * (self.depth_max - self.depth_min) + self.depth_min
        return depths

    def save(self, path):
        np.save(path, {"center_x": self.center_x, "center_y": self.center_y, "scale": self.scale, "depth_min": self.depth_min, 'depth_max':self.depth_max})

    def load(self, path):
        p = np.load(path, allow_pickle=True).item()
        self.center_x, self.center_y, self.scale = p["center_x"], p["center_y"], p["scale"]
        self.depth_max, self.depth_min = p['depth_max'], p['depth_min']
        self.is_fitted = True
        return self

class Pose_data():
    def __init__(self, conf):
        self.conf = conf
        # Read the data
        dataset = pd.read_csv(conf['file_path'])
        self.body_parts = {part:idx for idx,part in enumerate(dataset['part'].unique())}
        dataset['p_idx'] = dataset["part"].map(self.body_parts)
        self.dataset = dataset[["time"] + ["p_idx"] + dataset.columns.drop(["time", "p_idx"]).tolist()].drop(columns=['part'])
        self.part_count = len(self.body_parts.keys())

        # Extract coordinates
        self.rigid_coords = torch.tensor(self.dataset.iloc[:, 0:5].to_numpy(), dtype=torch.float32).view(-1, self.part_count, 5)
        self.deformable_coords = torch.tensor((self.dataset.drop(columns=['x_r', 'y_r', 'z_r']).iloc[:, 0:5]).to_numpy(), dtype=torch.float32).view(-1, self.part_count, 5)
        
        # Compute Headings of reference vector
        part1, part2 = conf.get('reference_parts', ("head center", "tail base"))
        T = self.rigid_coords.shape[0]
        if (part1 not in self.body_parts) or (part2 not in self.body_parts):
            self.headings = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
        else:
            i1 = self.body_parts[part1]
            i2 = self.body_parts[part2]

            x1 = self.rigid_coords[:, i1, 2]
            y1 = self.rigid_coords[:, i1, 3]
            x2 = self.rigid_coords[:, i2, 2]
            y2 = self.rigid_coords[:, i2, 3]

            dx = x2 - x1
            dy = y2 - y1

            angles = torch.atan2(dy, dx)
            neg_angles = -angles
            cos_a = torch.cos(neg_angles)
            sin_a = torch.sin(neg_angles)

            norm_sq = dx * dx + dy * dy
            valid = (norm_sq > 1e-12) & torch.isfinite(norm_sq)

            cos_a = torch.where(valid, cos_a, torch.ones_like(cos_a))
            sin_a = torch.where(valid, sin_a, torch.zeros_like(sin_a))

            R = torch.zeros((T, 3, 3), dtype=torch.float32)
            R[:, 2, 2] = 1.0
            R[:, 0, 0] = cos_a
            R[:, 0, 1] = -sin_a
            R[:, 1, 0] = sin_a
            R[:, 1, 1] = cos_a

            self.headings = R

        com = self.rigid_coords[:, :, 2:].mean(dim=1, keepdim=True)  # (n_frames, 1, 3)
        
        # Rotate in place
        deformable_centered = self.deformable_coords[:, :, 2:] - com
        deformable_aligned = torch.einsum('fij,fpj->fpi', self.headings, deformable_centered) 
        
        # Find unique poses
        unique_indices = find_unique_poses(deformable_aligned, threshold=self.conf['pose_threshold'])
        self.unique_poses = deformable_aligned[unique_indices]
        self.com = com[unique_indices]
        print(f"Found {len(self.unique_poses)} unique poses from {len(deformable_aligned)} frames")
    
    def get_unique_poses(self):
        return self.unique_poses, self.com

class MV_Dataset(Dataset):
    def __init__(self, pose_data, com, projections, n_rotations=18, part_count=10, normalize=True, load_norm_path=None, save_norm_path=None):
        self.part_count = part_count
        self.mask_ratio = 0.0
        # Initialize normalizer
        self.normalizer = DataDrivenNormalization()
        if load_norm_path is not None:
            print(f"Loading normalizer from {load_norm_path}")
            self.normalizer.load(load_norm_path)
        else:
            print(f"New normalizer initialized")
        self.views = {k:view for k,view in enumerate(projections)}
        self.view_count = len(projections)
        self.n_rotations = n_rotations

        augmented_data = self.augment_dataset_with_rotation(pose_data, com)

        traj_list_d = []
        depth_list = []
        # Project to 2D (unnormalized)
        for i, view in enumerate(projections):
            proj_data_d, depths_d = project_points(augmented_data, view)
            traj_list_d.append(proj_data_d[:, None, ...])
            depth_list.append(depths_d[:, None, :])

        self.deformable_coords = torch.concatenate(traj_list_d, dim=1)
        self.depths = torch.concatenate(depth_list, dim=1)
        # Normalize 2D coordinates using isotropic minmax
        if normalize:
            self.centers = self.deformable_coords.mean(dim=2)
            coords_centered = self.deformable_coords - self.centers.unsqueeze(2)

            if not self.normalizer.is_fitted:
                print("Fitting normalizer on 2D data...")
                self.normalizer.fit(coords_centered, self.depths)
                if save_norm_path is not None:
                    self.normalizer.save(save_norm_path)
            
            self.deformable_coords = self.normalizer.normalize(coords_centered)  
            self.depths = self.normalizer.normalize_depth(self.depths)  

    def augment_dataset_with_rotation(self, unique_poses, unique_com):
        """
        - Augment with random z-rotations
        """
        # Augment each unique pose with random rotations
        augmented_poses = []

        angles = get_stratified_angles(self.n_rotations) 
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)
        zeros = torch.zeros_like(angles)
        ones = torch.ones_like(angles)
        R_z_batch = torch.stack([
            torch.stack([cos_a, -sin_a, zeros], dim=1),
            torch.stack([sin_a,  cos_a, zeros], dim=1),
            torch.stack([zeros,  zeros,  ones], dim=1)
        ], dim=1)

        for pose in unique_poses:
            rotated_poses = torch.einsum('nij,pj->npi', R_z_batch, pose)
            augmented_poses.append(rotated_poses)
        
        augmented_poses = torch.stack(augmented_poses).view(-1, *unique_poses.shape[1:])  # (n_unique * n_rotations, part_count, 3)
        com_expanded = unique_com.repeat_interleave(self.n_rotations, dim=0)
        final_poses = augmented_poses + com_expanded
        return final_poses

    def set_occlusion(self, ratio):
        self.mask_ratio = float(ratio)

    def __getitem__(self, idx):
        if self.mask_ratio <= 0:
            mask = torch.ones((self.view_count, self.part_count, 1), dtype=torch.bool)
        else:
            n_hide = int(round(self.mask_ratio * self.part_count))
            n_hide = max(0, min(self.part_count, n_hide))
            mask = torch.ones((self.view_count, self.part_count, 1), dtype=torch.bool)
            for v in range(self.view_count):
                m = exact_mask(self.part_count, self.mask_ratio) 
                mask[v, :, 0] = m

        return self.deformable_coords[idx], self.depths[idx], mask
        
    def __len__(self):
        return self.deformable_coords.shape[0]
    
    def denormalize_2d(self, keypoints_2d_norm):
        return self.normalizer.denormalize(keypoints_2d_norm)
    
    def denormalize_depths(self, depths_norm):
        return self.normalizer.denormalize_depth(depths_norm)
    
def get_stratified_angles(n_rotations):
    """
    Generates n_rotations angles that are guaranteed to cover 
    the full 360 circle evenly, with random jitter.
    """
    # Create the base intervals (e.g. for n=4: 0, 90, 180, 270)
    base_angles = torch.linspace(0, 2 * torch.pi, n_rotations + 1)[:-1]
    sector_width = 2 * torch.pi / n_rotations
    
    # Add random jitter within that sector width
    noise = torch.rand(n_rotations) * sector_width
    
    final_angles = base_angles + noise
    return final_angles


def find_unique_poses(poses, threshold=0.01):
    """
    Find unique poses using greedy clustering.
    poses: (N, P, 3) tensor of centered poses
    Returns: indices of unique poses
    """
    N = poses.shape[0]
    kept_indices = [0]
    flat_poses = poses.view(N, -1)  # (N, P*3)
    
    for i in range(1, N):
        current = flat_poses[i:i+1]
        kept = flat_poses[kept_indices]
        dists = torch.cdist(current, kept)
        
        if torch.min(dists) > threshold:
            kept_indices.append(i)
    return torch.tensor(kept_indices)

def exact_mask(K: int, ratio: float, *, generator=None, device=None):
    n_hide = int(round(ratio * K))
    n_hide = max(0, min(K, n_hide))

    m = torch.ones(K, dtype=torch.bool, device=device)
    if n_hide == 0:
        return m

    idx = torch.randperm(K, generator=generator, device=device)[:n_hide]
    m[idx] = False  # False = hidden
    return m
