import torch
import numpy as np

# Forward direction function (approximate based on shoulder blade joints)
def forward_facing(data, LR_joint_indices, forward_sign=1):
    li, ri = LR_joint_indices
    hip_vec = data[:, ri, :2] - data[:, li, :2]  # (T, 2)
    if isinstance(data, torch.Tensor):
        forward_vec = torch.stack([-hip_vec[:, 1], hip_vec[:, 0]], dim=1) * forward_sign
        norms = torch.linalg.norm(forward_vec, axis=1, keepdims=True)
    elif isinstance(data, np.ndarray):   
        forward_vec = np.stack([-hip_vec[:, 1], hip_vec[:, 0]], axis=1) * forward_sign
        norms = np.linalg.norm(forward_vec, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    forward_vec = forward_vec / norms
    return forward_vec, norms

def extract_poses(local_coords, reference_indices=(8,11), pose_threshold=0.01, n_rotations=18):
    local_coords = torch.tensor(local_coords, dtype=torch.float32)
    # Get forward direction of the body
    forward_vecs, norms = forward_facing(local_coords, reference_indices)
    # Get angles from axes
    theta = np.arctan2(forward_vecs[:, 1], forward_vecs[:, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    # Remove rotation
    gx, gy = local_coords[..., 0].clone(), local_coords[..., 1].clone()
    local_coords[..., 0] = c[:, None] * gx - s[:, None] * gy
    local_coords[..., 1] = s[:, None] * gx + c[:, None] * gy

    # Find unique poses
    unique_indices = find_unique_poses(local_coords, threshold=pose_threshold)
    unique_poses = local_coords[unique_indices]
    print(f"Found {len(unique_poses)} unique poses from {len(local_coords)} frames")
    final_poses = augment_dataset_with_rotation(unique_poses, n_rotations)
    return unique_poses, final_poses   

def find_unique_poses(poses, threshold=0.01):
    """
    Find unique poses using greedy clustering.
    poses: (N, P, 3) tensor of centered poses
    Returns: indices of unique poses
    """
    N = poses.shape[0]
    kept_indices = [0]
    print(poses.shape)
    flat_poses = poses.reshape(N, -1)  # (N, P*3)
    
    for i in range(1, N):
        current = flat_poses[i:i+1]
        kept = flat_poses[kept_indices]
        dists = torch.cdist(current, kept)
        
        if torch.min(dists) > threshold:
            kept_indices.append(i)
    return torch.tensor(kept_indices)

def augment_dataset_with_rotation(unique_poses, n_rotations):
    # Augment each unique pose with random rotations
    augmented_poses = []

    angles = get_stratified_angles(n_rotations)
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
    return augmented_poses

def get_stratified_angles(n_rotations):
    """
    Generates n_rotations angles that are guaranteed to cover 
    the full 360 circle evenly, with random jitter.
    """
    if n_rotations == 1:
        return torch.tensor([0.0])
    # Create the base intervals (e.g. for n=4: 0, 90, 180, 270)
    base_angles = torch.linspace(0, 2 * torch.pi, n_rotations + 1)[:-1]
    sector_width = 2 * torch.pi / n_rotations
    
    # Add random jitter within that sector width
    noise = torch.rand(n_rotations) * sector_width
    
    final_angles = base_angles + noise
    return final_angles