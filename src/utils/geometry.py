import numpy as np
import torch

def project_points(points_3d: torch.Tensor, P: torch.Tensor):
    """
    points_3d: [B, N, 3]  (or [N,3] -> will be unsqueezed)
    P:         [B, 3, 4] or [3,4] (broadcastable across batch)

    Returns:
        pixels: [B, N, 2]
        depths: [B, N]  (z in camera frame)
    """
    if points_3d.dim() == 2:
        points_3d = points_3d.unsqueeze(0)  # [1, N, 3]
    B, N, _ = points_3d.shape

    if P.dim() == 2:
        P = P.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 4]

    ones = torch.ones(B, N, 1, device=points_3d.device, dtype=points_3d.dtype)
    points_h = torch.cat([points_3d, ones], dim=-1)  # [B, N, 4]

    # [B, 3, 4] @ [B, 4, N] -> [B, 3, N] -> [B, N, 3]
    proj = (P @ points_h.transpose(-1, -2)).transpose(-1, -2)  # [B, N, 3]

    depths = proj[..., 2]                          # [B, N]
    pixels = proj[..., :2] / (depths.unsqueeze(-1) + 1e-8)

    return pixels, depths

def triangulate_points(points_2d, Ps):
    """
    Linear multi-view triangulation via SVD.

    Args:
        points_2d: [B, V, N, 2] 2D points (u, v) per view
        Ps:        [V, 3, 4] world-to-cam projection matrices for each view

    Returns:
        points_3d: [B, N, 3] triangulated 3D points in world coordinates
    """
    B, V, N, _ = points_2d.shape
    if Ps.dim() == 3:
        Ps = Ps.unsqueeze(0).expand(B, -1, -1, -1)  # [B, V, 3, 4]
    assert Ps.shape == (B, V, 3, 4)

    # [B, V, N, 1]
    u = points_2d[..., 0:1]
    v = points_2d[..., 1:2]

    # [B, V, N, 3, 4]
    Ps_exp = Ps.unsqueeze(2).expand(-1, -1, N, -1, -1)
    P0 = Ps_exp[..., 0, :]  # [B, V, N, 4]
    P1 = Ps_exp[..., 1, :]
    P2 = Ps_exp[..., 2, :]

    # Build A: [B, V, N, 2, 4]
    A_u = u * P2 - P0
    A_v = v * P2 - P1
    A = torch.stack([A_u, A_v], dim=3)  # [B, V, N, 2, 4]

    # Move views and the 2 rows into one dimension: [B, N, 2V, 4]
    A = A.permute(0, 2, 1, 3, 4).reshape(B, N, 2 * V, 4)

    # SVD per point: reshape to batch of matrices [(B*N), 2V, 4]
    A_flat = A.reshape(B * N, 2 * V, 4)

    # torch.linalg.svd returns U, S, Vh with Vh shape [(B*N), 4, 4]
    _, _, Vh = torch.linalg.svd(A_flat)
    X_h = Vh[..., -1, :]  # last row is the right singular vector for smallest singular value

    # De-homogenize
    X = X_h[..., :3] / X_h[..., 3:].clamp(min=1e-8)

    # Reshape back to [B, N, 3]
    return X.reshape(B, N, 3)

    
def unproject_points(pixels_2d, depths, P):
    """
    Unproject 2D pixels + depth using projection matrix P directly
    
    Args:
        pixels_2d: [B, V, N, 2]
        depths: [B, V, N, 1] - depth in camera frame (Z_cam)
        P: [B, V, 3, 4] - projection matrices
    
    Returns:
        points_3d: [B, V, N, 3] - 3D points in world coordinates
    """
    B, V, N, _ = pixels_2d.shape
    
    # Build homogeneous 2D points scaled by depth
    # For P @ X_world = λ [u, v, 1]^T, and given λ (depth), solve for X_world
    uv_homog = torch.cat([pixels_2d, torch.ones_like(pixels_2d[..., :1])], dim=-1)  # [B, V, N, 3]
    uv_scaled = uv_homog * depths  # [B, V, N, 3]
    
    # Solve P @ [X, Y, Z, 1]^T = depth * [u, v, 1]^T
    # This requires computing pseudo-inverse of P
    M = P[..., :3]  # [B, V, 3, 3]
    p4 = P[..., 3:4]  # [B, V, 3, 1]
    
    # Rearrange: M @ X_world = uv_scaled - p4
    target = uv_scaled.unsqueeze(-1) - p4.unsqueeze(2)  # [B, V, N, 3, 1]
    
    # Solve for X_world
    M_expanded = M.unsqueeze(2).expand(B, -1, N, -1, -1)  # [B, V, N, 3, 3]
    X_world = torch.linalg.solve(M_expanded.reshape(B*V*N, 3, 3), 
                                   target.reshape(B*V*N, 3, 1))
    X_world = X_world.reshape(B, V, N, 3)
    
    return X_world

def get_predefined_cams(direction):
    if direction == "front":
        return torch.tensor([
                            [0.9577030433069325, 1.27591706539075, -0.05103668261562999, 80.57423917326688],
                            [0, 0.9186602870813396, -0.9952153110047846, 249.5693779904306],
                            [0, 0.003987240829346092, -0.0001594896331738437, 0.9999999999999999]
                        ])
    elif direction == "top_angled":
        return torch.tensor([
                            [0.6, 0, -0.8, 170],
                            [0, -0.6, -0.6, 390],
                            [0, 0, -0.0025, 1]
                        ])
    elif direction == "top":
        return torch.tensor([
                            [0.6, 0, 0.0, 170],
                            [0, -0.6, 0.0, 390],
                            [0, 0, 0.0, 1]
                        ])
    elif direction == "right":
        return torch.tensor([
                    [1.13759300176461, 0.9777876595579652, -0.3818615751789976, 25.04504946649224],
                    [-0.005729641489431923, 0.286482074471596, -1.360596637235798, 345.229552852311],
                    [-5.966587112171838e-05, 0.002983293556085919, -0.001193317422434368, 1]
                ])
    elif direction == "left":
        return torch.tensor([
                            [1.211934729986154, 0.9601647494347173, -0.3936039360393604, 27.8976393698744],
                            [0.005905829727114331, 0.2952914863557164, -1.402435402218449, 352.8925528981115],
                            [6.150061500615007e-05, 0.003075030750307503, -0.001230012300123001, 1]
                        ])
    elif direction == "side":
        return torch.tensor([
                            [-0.4262120404901438, 0.3199146553827633, -0.01704848161960576, 240.0213361543092],
                            [-0.3068726691529035, 0, -0.3324453915823122, 236.8034096963239],
                            [-0.001331912626531699, 0, -5.327650506126798e-05, 0.9999999999999999]
                        ])
    else:
        return None

def baseline_triangulation(keypoints_2d, visibility_mask, camera_matrices):
    """
    keypoints_2d: (num_views, num_keypoints, 2)
    visibility_mask: (num_views, num_keypoints) - bool
    camera_matrices: (num_views, 3, 4) - projection matrices
    """
    num_keypoints = keypoints_2d.shape[1]
    num_views = keypoints_2d.shape[0]
    points_list = []
    
    for kp_idx in range(num_keypoints):
        # Collect visible observations
        visible_views = visibility_mask[:, kp_idx]
        if visible_views.sum() < num_views:
            if visible_views.sum() < 2:
                points_list.append(torch.zeros((3,)))
            else:
                # Triangulate from visible views
                points_list.append(triangulate_points(
                    keypoints_2d[visible_views, kp_idx][:, None, :],
                    [x for i,x in enumerate(camera_matrices) if visible_views[i]]
                ))
    # Project back to all views
    points_3d = torch.vstack(points_list)
    traj_list = []
    for view in camera_matrices:
        proj_data = project_points(points_3d,view)[0]
        traj_list.append(proj_data)
    predicted_2d = torch.vstack(traj_list).view(num_views, -1, 2)

    # Re-insert the data back into the masked spots
    counter = 0
    result = []
    for i in range(num_keypoints):
        if visibility_mask[:, i].sum() == num_views:
            result.append(keypoints_2d[:, i, :])
        else:
            result.append(predicted_2d[:,counter,:])
            counter+=1
    tmp = torch.vstack(result) # (N,2)
    return torch.stack([tmp[i::num_views] for i in range(num_views)]) # (C, N, 2)

def compute_linear_velocities(coords, fps):
    # Delta Position / Delta Time
    dt = 1.0 / fps
    diff_points = torch.vstack([torch.zeros_like(coords[0]).unsqueeze(0), coords[1:] - coords[:-1]])
    return diff_points / dt

def compute_angular_velocity(centered, fps):
    """
    Computes the global Angular Velocity vector for the skeleton per frame.
    Uses Procrustes Analysis (SVD) to find the rotation between frames.
    
    Returns:
        ang_vel: (T-1, 3) Vector w where:
                    - Direction is the axis of rotation
                    - Magnitude is the speed (radians/second)
    """
    dt = 1.0 / fps
    
    # We want Rotation R that transforms frame t to t+1
    src = centered[:-1] # (T-1, P, 3)
    tgt = centered[1:]  # (T-1, P, 3)
    
    # Compute Cross-Covariance Matrix H = src.T @ tgt (T-1, 3, P) @ (T-1, P, 3) -> (T-1, 3, 3)
    H = torch.matmul(src.transpose(1, 2), tgt)
    
    # SVD to find Rotation Matrix
    U, S, Vh = torch.linalg.svd(H)
    
    # Rotation R = V @ U.T
    V = Vh.mH # Conjugate transpose for real matrices is just transpose
    R = torch.matmul(V, U.transpose(1, 2))
    
    # Handle Reflection case (Det = -1)
    det = torch.linalg.det(R)
    mask = det < 0
    if mask.any():
        # Create correction matrix for cases where det is negative
        diag = torch.ones((mask.sum(), 3), device=centered.device)
        diag[:, 2] = -1
        Z = torch.diag_embed(diag)
        
        # Apply correction to V only for the masked batches
        V_fixed = V[mask]
        U_fixed = U[mask]
        # Recompute R for these
        R[mask] = torch.matmul(torch.matmul(V_fixed, Z), U_fixed.transpose(1, 2))

    # Convert Rotation Matrix R to Angular Velocity Vector
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    
    # Calculate axis vector from skew-symmetric parts
    # [ R_21 - R_12 ]
    # [ R_02 - R_20 ]
    # [ R_10 - R_01 ]
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], dim=1)
    
    # Normalize axis
    axis_norm = torch.norm(axis, dim=1, keepdim=True)
    # Handle zero rotation case (avoid divide by zero)
    axis = torch.where(axis_norm > 1e-6, axis / axis_norm, torch.zeros_like(axis))
    ang_vel = (theta.unsqueeze(1) * axis) / dt
    
    return ang_vel

def extract_deformable_coordinates(noisy_points, template, confidence_weights=None, device='cuda'):
    """
    Closed-form weighted Procrustes alignment (Kabsch algorithm) to extract the rigid coordinates.
    
    Args:
        noisy_points: (10, 3) Target points
        confidence_weights: (10,) Optional weights
    
    Returns:
        aligned_points: (10, 3) Aligned template
        rotation: (3, 3) Rotation matrix
        translation: (3,) Translation vector
    """
    noisy_points = noisy_points.to(device)
    template = template.clone().to(device)
    
    if confidence_weights is None:
        confidence_weights = torch.ones(template.shape[0], device=device)
    else:
        confidence_weights = confidence_weights.to(device)
    
    # Normalize weights
    W = confidence_weights / confidence_weights.sum()
    
    # Weighted centroids
    centroid_template = (template * W[:, None]).sum(dim=0)
    centroid_noisy = (noisy_points * W[:, None]).sum(dim=0)
    
    # Center the point clouds
    template_centered = template - centroid_template
    noisy_centered = noisy_points - centroid_noisy
    
    # Weighted covariance matrix
    H = (template_centered.T * W[None, :]) @ noisy_centered  # (3, 3)
    
    # SVD for optimal rotation (Kabsch algorithm)
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure det(R) = 1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Optimal translation
    t = centroid_noisy - R @ centroid_template
    
    # Apply transformation
    aligned = (R @ template.T).T + t
    
    return aligned.cpu(), R.cpu(), t.cpu()

def decompose_projection(P, validate=True):
    """
    Robustly decompose a 3x4 projection matrix into intrinsics (K) and world-to-camera extrinsics [R|t].
    Args:
        P: torch.Tensor of shape (3, 4) - projection matrix P = K[R|t]
        validate: bool - verify that K @ [R|t] reconstructs P
        
    Returns:
        K (3, 3) - intrinsic matrix (upper triangular, positive diagonal)
        extrinsics_w2c (3, 4) - [R|t] world-to-camera transform
    """
    P = P.clone().float()
    M = P[:, :3]  # Should be K @ R
    p4 = P[:, 3:4]  # Should be K @ t

    # RQ decomposition using QR on flipped matrix
    M_flip = torch.flip(M, dims=[0, 1])
    Q_flip, R_flip = torch.linalg.qr(M_flip.T)
    K = torch.flip(R_flip.T, dims=[0, 1])
    R = torch.flip(Q_flip.T, dims=[0, 1])

    # Ensure K has positive diagonal
    signs = torch.sign(torch.diag(K))
    signs[signs == 0] = 1.0
    D = torch.diag(signs)
    K = K @ D
    R = D @ R
    
    # Ensure det(R) = +1
    if torch.det(R) < 0:
        K[2, :] = -K[2, :]
        R[:, 2] = -R[:, 2]

    # Solve for translation
    t = torch.linalg.solve(K, p4)
    extrinsics_w2c = torch.cat([R, t], dim=1)
    
    if validate:
        print("\n" + "="*70)
        print("CAMERA PARAMETER ANALYSIS")
        print("="*70)
        
        # INTRINSICS
        print("\n📷 INTRINSIC PARAMETERS")
        print("-" * 70)
        
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
        skew = K[0, 1].item()
        scale = K[2, 2].item()
        
        print(f"  Focal length (fx): {fx:.6f}")
        print(f"  Focal length (fy): {fy:.6f}")
        print(f"  Focal length ratio (fx/fy): {fx/fy:.6f}")
        print(f"  Principal point (cx, cy): ({cx:.6f}, {cy:.6f})")
        print(f"  Skew coefficient: {skew:.6e}")
        print(f"  Scale factor (K[2,2]): {scale:.6e}")
        print(f"  Aspect ratio (fy/fx): {fy/fx:.6f}")
        
        if abs(fx - fy) < 1e-6:
            print("  ✓ Square pixels (fx ≈ fy)")
        else:
            print("  ⚠ Non-square pixels")
        
        # EXTRINSICS
        print("\n🌍 EXTRINSIC PARAMETERS (World-to-Camera)")
        print("-" * 70)
        
        tx, ty, tz = t[0, 0].item(), t[1, 0].item(), t[2, 0].item()
        print(f"  Translation (tx, ty, tz): ({tx:.6f}, {ty:.6f}, {tz:.6f})")
        print(f"  Camera distance from origin: {torch.norm(t).item():.6f}")
        
        print(f"\n  Rotation matrix R:")
        for i in range(3):
            print(f"    [{R[i,0].item():9.6f}, {R[i,1].item():9.6f}, {R[i,2].item():9.6f}]")
        
        # Euler angles
        R_np = R.cpu().numpy()
        if abs(R_np[2, 0]) < 0.99999:
            pitch = np.arcsin(-R_np[2, 0])
            roll = np.arctan2(R_np[2, 1], R_np[2, 2])
            yaw = np.arctan2(R_np[1, 0], R_np[0, 0])
        else:
            yaw = 0
            if R_np[2, 0] < 0:
                pitch = np.pi / 2
                roll = np.arctan2(R_np[0, 1], R_np[1, 1])
            else:
                pitch = -np.pi / 2
                roll = np.arctan2(-R_np[0, 1], R_np[1, 1])
        
        print(f"\n  Euler angles (in degrees):")
        print(f"    Roll  (X-axis): {np.degrees(roll):8.3f}°")
        print(f"    Pitch (Y-axis): {np.degrees(pitch):8.3f}°")
        print(f"    Yaw   (Z-axis): {np.degrees(yaw):8.3f}°")
        
        # Camera axes in world frame
        R_c2w = R.T
        cam_x = R_c2w[:, 0]
        cam_y = R_c2w[:, 1]
        cam_z = R_c2w[:, 2]
        
        print(f"\n  Camera axes in world coordinates:")
        print(f"    X-axis (right):   [{cam_x[0].item():7.4f}, {cam_x[1].item():7.4f}, {cam_x[2].item():7.4f}]")
        print(f"    Y-axis (down):    [{cam_y[0].item():7.4f}, {cam_y[1].item():7.4f}, {cam_y[2].item():7.4f}]")
        print(f"    Z-axis (forward): [{cam_z[0].item():7.4f}, {cam_z[1].item():7.4f}, {cam_z[2].item():7.4f}]")
        
        # Camera position in world
        camera_pos_world = -R.T @ t
        print(f"\n  Camera position in world coordinates:")
        print(f"    ({camera_pos_world[0,0].item():.6f}, {camera_pos_world[1,0].item():.6f}, {camera_pos_world[2,0].item():.6f})")
        
        # VALIDATION
        print("\n✅ VALIDATION")
        print("-" * 70)
        
        P_reconstructed = K @ extrinsics_w2c
        max_error = (P - P_reconstructed).abs().max().item()
        mean_error = (P - P_reconstructed).abs().mean().item()
        
        print(f"  Max reconstruction error: {max_error:.2e}")
        print(f"  Mean reconstruction error: {mean_error:.2e}")
        print(f"  R orthogonality error: {(R @ R.T - torch.eye(3)).abs().max().item():.2e}")
        print(f"  det(R) = {torch.det(R).item():.6f} (should be ≈ 1.0)")
        
        if max_error < 1e-5:
            print("\n  ✅ PERFECT RECONSTRUCTION!")
        else:
            print(f"\n  ⚠️  Reconstruction error detected")
        
        print("="*70 + "\n")
        
    return K, extrinsics_w2c