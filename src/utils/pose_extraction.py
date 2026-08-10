import torch
    
def extract_poses(rigid_coords, deformable_coords, body_parts, reference_parts=("head center", "tail base"), pose_threshold=0.01, n_rotations=18):
    body_parts_map = {part: idx for idx, part in enumerate(body_parts)}
    rigid_coords = torch.tensor(rigid_coords, dtype=torch.float32)
    deformable_coords = torch.tensor(deformable_coords, dtype=torch.float32)

    # Compute Headings of reference vector
    part1, part2 = reference_parts
    T = rigid_coords.shape[0]
    if (part1 not in body_parts_map) or (part2 not in body_parts_map):
        headings = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(T, 1, 1)
    else:
        i1 = body_parts_map[part1]
        i2 = body_parts_map[part2]

        x1 = rigid_coords[:, i1, 0]
        y1 = rigid_coords[:, i1, 1]
        x2 = rigid_coords[:, i2, 0]
        y2 = rigid_coords[:, i2, 1]

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

        headings = R

    com = rigid_coords.mean(dim=1, keepdim=True)  # (n_frames, 1, 3)

    # Rotate in place
    deformable_centered = deformable_coords[:, :, 2:] - com
    deformable_aligned = torch.einsum('fij,fpj->fpi', headings, deformable_centered)

    # Find unique poses
    unique_indices = find_unique_poses(deformable_aligned, threshold=pose_threshold)
    unique_poses = deformable_aligned[unique_indices]
    com = com[unique_indices]
    print(f"Found {len(unique_poses)} unique poses from {len(deformable_aligned)} frames")
    final_poses = augment_dataset_with_rotation(unique_poses, com, n_rotations)
    return unique_poses, com, final_poses   

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

def augment_dataset_with_rotation(unique_poses, unique_com, n_rotations):
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
    com_expanded = unique_com.repeat_interleave(n_rotations, dim=0)
    final_poses = augmented_poses + com_expanded
    return final_poses

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