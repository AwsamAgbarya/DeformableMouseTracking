import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.geometry import project_points
from utils.normalizers import NormalizerMV

class MV_Dataset(Dataset):
    def __init__(self, pose_data, com, projections, n_rotations=18, part_count=10, normalize=True, load_norm_path=None, save_norm_path=None):
        self.part_count = part_count
        self.mask_ratio = 0.0
        # Initialize normalizer
        self.normalizer = NormalizerMV()
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
