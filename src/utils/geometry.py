import numpy as np
import torch
from scipy.interpolate import CubicSpline

# MVT
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
# MVT
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

# MVT
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

#MVT
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
                            [ 0.6000,  0.0000, -0.0000,  170.0000],
                            [ 0.0000, -0.6000, -0.0000,  390.0000],
                            [ 0.0000,  0.0000, -0.0025,   1.0000 ],
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

# triangulation MVT and data analysis
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

# OAT
def linear_interpolate_keypoints(
    keypoints: np.ndarray,
    mask: np.ndarray,
    outlier_frames: np.ndarray = None,
) -> np.ndarray:
    """
    Naive linear interpolation for occluded keypoints over time.

    For each joint, linearly interpolates missing frames using the nearest
    visible frames before and after. Leading/trailing gaps are filled by
    repeating the nearest visible value (constant extrapolation).

    Args:
        keypoints: (T, N, 3) array of 3D keypoint coordinates over time.
        mask:      (T, N, 1) or (T, N) boolean array. True = visible.
        outlier_frames: (T,) boolean array. True = outlier frame.

    Returns:
        (T, N, 3) array with occluded frames filled via linear interpolation.
    """
    keypoints = np.array(keypoints, dtype=np.float64)
    mask = np.array(mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0]  # (T, N)

    T, N, D = keypoints.shape
    assert D == 3, f"Expected (T, N, 3), got last dim {D}"
    assert mask.shape == (T, N), f"mask must be (T, N) or (T, N, 1), got {mask.shape}"

    result = keypoints.copy()
    t_idx = np.arange(T, dtype=float)

    for n in range(N):
        visible = mask[:, n]           # (T,) — which frames are visible for joint n
        # To not count outlier frames, we set their entire mask to true so they dont get interpolated over
        if outlier_frames is not None:
            visible = visible | outlier_frames

        vis_t = np.where(visible)[0]

        if len(vis_t) == 0:
            continue  # No visible frames; leave as-is
        if len(vis_t) == T:
            continue  # All visible; nothing to do

        occ_t = np.where(~visible)[0]

        for dim in range(3):
            result[occ_t, n, dim] = np.interp(
                t_idx[occ_t],
                vis_t.astype(float),
                keypoints[vis_t, n, dim],
            )

    return result

# OAT test
def cubic_spline_interpolate_keypoints(
    keypoints: np.ndarray,
    mask: np.ndarray,
    min_visible_for_spline: int = 4,
    outlier_frames: np.ndarray = None,
) -> np.ndarray:
    """
    Cubic spline interpolation with optional bone-length projection.

    For each joint, fits a cubic spline through its visible frames and uses
    it to fill occluded frames. Falls back to linear interpolation when fewer
    than `min_visible_for_spline` visible frames exist for a joint.

    Args:
        keypoints:                  (T, N, 3) array of 3D keypoints over time.
        mask:                       (T, N, 1) or (T, N) boolean. True = visible.
        min_visible_for_spline:     Minimum visible frames to fit a cubic spline
                                    (must be >= 4). Falls back to linear otherwise.
        outlier_frames:             (T,) boolean array. True = outlier frame.

    Returns:
        (T, N, 3) array with occluded frames filled and optionally projected.
    """
    keypoints = np.array(keypoints, dtype=np.float64)
    mask = np.array(mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0]  # (T, N)

    T, N, D = keypoints.shape
    assert D == 3, f"Expected (T, N, 3), got last dim {D}"
    assert mask.shape == (T, N), f"mask must be (T, N) or (T, N, 1), got {mask.shape}"
    assert min_visible_for_spline >= 4, "cubic spline requires at least 4 control points"

    result = keypoints.copy()
    t_idx = np.arange(T, dtype=float)

    for n in range(N):
        visible = mask[:, n]           # (T,) visibility for joint n
        # To not count outlier frames, we set their entire mask to true so they dont get interpolated over
        if outlier_frames is not None:
            visible = visible | outlier_frames
        vis_t = np.where(visible)[0]
        occ_t = np.where(~visible)[0]

        if len(vis_t) == 0 or len(occ_t) == 0:
            continue

        # --- Step 1: Spline or linear interpolation per joint trajectory ---
        if len(vis_t) >= min_visible_for_spline:
            for dim in range(3):
                cs = CubicSpline(
                    vis_t.astype(float),
                    keypoints[vis_t, n, dim],
                    extrapolate=True,
                )
                result[occ_t, n, dim] = cs(occ_t.astype(float))
        else:
            # Fall back to linear for this joint
            for dim in range(3):
                result[occ_t, n, dim] = np.interp(
                    t_idx[occ_t],
                    vis_t.astype(float),
                    keypoints[vis_t, n, dim],
                )

    return result