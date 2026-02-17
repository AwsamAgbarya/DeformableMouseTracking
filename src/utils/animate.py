import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def visualize_predictions(timeframe, keypoints_gt, keypoints_pred, masks, 
                          view_names=['Top', 'Right', 'Front'],
                          body_parts=None, denormalize=False, normalizer=None):
    """
    Visualize ground truth and predictions for all views.
    
    Args:
        timeframe: Frame number
        keypoints_gt: Ground truth keypoints (V, N, 2) - normalized or unnormalized
        keypoints_pred: Predicted keypoints (V, N, 2) - normalized or unnormalized
        masks: Visibility masks (V, N) - True=visible, False=occluded
        view_names: Names of camera views
        body_parts: Dictionary mapping part names to indices
        denormalize: Whether to denormalize coordinates for visualization
        normalizer: Normalizer object for denormalization
    """
    num_views = keypoints_gt.shape[0]
    num_keypoints = keypoints_gt.shape[1]
    
    # Denormalize if needed
    if denormalize and normalizer is not None:
        keypoints_gt = normalizer.denormalize(keypoints_gt.unsqueeze(0)).squeeze(0)
        keypoints_pred = normalizer.denormalize(keypoints_pred.unsqueeze(0)).squeeze(0)
    
    fig, axes = plt.subplots(1, num_views, figsize=(6*num_views, 6))
    if num_views == 1:
        axes = [axes]
    
    for view_idx in range(num_views):
        ax = axes[view_idx]
        
        gt = keypoints_gt[view_idx].cpu().numpy()  # (N, 2)
        pred = keypoints_pred[view_idx].cpu().numpy()
        visible = masks[view_idx].cpu().numpy().squeeze()  # (N,)
        
        # Plot visible keypoints (ground truth)
        visible_gt = gt[visible]
        if len(visible_gt) > 0:
            ax.scatter(visible_gt[:, 0], visible_gt[:, 1], 
                      c='green', s=100, marker='o', 
                      label='GT Visible', alpha=0.6, edgecolors='darkgreen', linewidths=2)
        
        # Plot occluded keypoints (ground truth) - show as X
        occluded_gt = gt[~visible]
        if len(occluded_gt) > 0:
            ax.scatter(occluded_gt[:, 0], occluded_gt[:, 1], 
                      c='red', s=100, marker='x', 
                      label='GT Occluded', alpha=0.7, linewidths=3)
        
        # Plot predictions for occluded keypoints
        occluded_pred = pred[~visible]
        if len(occluded_pred) > 0:
            ax.scatter(occluded_pred[:, 0], occluded_pred[:, 1], 
                      c='blue', s=80, marker='o', 
                      label='Predicted', alpha=0.7, edgecolors='darkblue', linewidths=2)
        
        # Draw error lines for occluded predictions
        for kpt_idx in range(num_keypoints):
            if not visible[kpt_idx]:  # Only for occluded
                ax.plot([gt[kpt_idx, 0], pred[kpt_idx, 0]], 
                       [gt[kpt_idx, 1], pred[kpt_idx, 1]], 
                       'r--', alpha=0.5, linewidth=1)
        
        # Add keypoint labels
        if body_parts is not None:
            part_names = {v: k for k, v in body_parts.items()}
            for kpt_idx in range(num_keypoints):
                if not visible[kpt_idx]:  # Label occluded points
                    ax.text(pred[kpt_idx, 0], pred[kpt_idx, 1], 
                           part_names.get(kpt_idx, str(kpt_idx)),
                           fontsize=8, ha='right')
        
        ax.set_title(f'{view_names[view_idx]} View (Frame {timeframe})', fontsize=14, fontweight='bold')
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Invert y-axis for image coordinates
        ax.invert_yaxis()
    
    plt.tight_layout()
    return fig

def animate_keypoints(kpts_array, title="Keypoint Animation", interval=50, normalized=False):
    """
    Animates a sequence of 2D keypoints.
    
    Args:
        kpts_array (np.ndarray): Shape (Frames, Keypoints, 2)
        title (str): Title of the plot
        interval (int): Time between frames in ms
        
    Returns:
        HTML object: Displayable animation in Jupyter
    """
    if isinstance(kpts_array, torch.Tensor):
        kpts_array = kpts_array.detach().cpu().numpy()
        
    frames, num_kpts, _ = kpts_array.shape
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if not normalized:
        min_x = np.min(kpts_array[:,:,0])
        max_x = np.max(kpts_array[:,:,0])
        min_y = np.min(kpts_array[:,:,1])
        max_y = np.max(kpts_array[:,:,1])
    else:
        min_x, max_x = (-1.1, 1.1)
        min_y, max_y = (-1.1, 1.1)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.invert_yaxis()
    
    # Initialize Plot Elements
    # Scatter for points
    scatter = ax.scatter([], [], s=30, c='red', zorder=2)
    
    # Optional: Lines connecting points (if you want a skeleton)
    # lines, = ax.plot([], [], lw=2, c='blue', alpha=0.5, zorder=1)
    
    # Update function for animation
    def update(frame_idx):
        # Get current frame data
        current_kpts = kpts_array[frame_idx]
        
        # Update scatter plot
        scatter.set_offsets(current_kpts)
        
        # Update title with frame number
        ax.set_title(f"{title} - Frame {frame_idx}/{frames}")
        return scatter,

    # Create Animation
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.close() # Prevent static plot from showing up
    
    return HTML(anim.to_jshtml())

def visualize_centering_effect(dataset, num_samples=20):
    """
    Visualizes the data pipeline stages with specific axis inversions for Views 2 & 3.
    """
    
    # 1. Setup the figure (3 Rows x 3 Views)
    num_views = dataset.view_count
    fig, axes = plt.subplots(3, num_views, figsize=(5 * num_views, 12))

    # Titles & Axis Inversion (DO THIS ONCE HERE)
    rows = ['Stage 1: Absolute', 'Stage 2: Centered', 'Stage 3: Normalized']
    
    for r in range(3):
        for c in range(num_views):
            ax = axes[r, c]
            
            # Title only on top row
            if r == 0:
                ax.set_title(f"View {c+1}")
            
            # Row labels on left column
            if c == 0:
                ax.set_ylabel(rows[r], rotation=90, size='large')
            
            # INVERT Y-AXIS FOR VIEW 2 (index 1) AND VIEW 3 (index 2)
            # This logic runs once per subplot.
            if c > 0: # Indices 1 and 2
                ax.invert_yaxis()

    # 2. Collect Data
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Track min/max for absolute plot scaling
    abs_min_x, abs_max_x = float('inf'), float('-inf')
    abs_min_y, abs_max_y = float('inf'), float('-inf')

    colors = plt.cm.jet(np.linspace(0, 1, num_samples))
    print(f"Visualizing {num_samples} random frames...")

    for i, idx in enumerate(indices):
        # Retrieve data
        kpts_norm, _, masks, centers = dataset[idx]
        
        kpts_norm = kpts_norm.cpu().numpy()     
        centers = centers.cpu().numpy()         
        masks = masks.cpu().numpy()             

        # Reconstruct Stages
        kpts_centered = dataset.normalizer.denormalize(torch.tensor(kpts_norm)).numpy()
        kpts_absolute = kpts_centered + centers[:, np.newaxis, :]

        # 3. Plotting Loop
        for v in range(num_views):
            # Mask logic (updated to use all points for now per your code)
            mask_v = np.ones_like(kpts_absolute[v,:,0], dtype=bool)
            
            # --- Row 1: Absolute ---
            if np.any(mask_v):
                x = kpts_absolute[v, mask_v, 0]
                y = kpts_absolute[v, mask_v, 1]
                axes[0, v].scatter(x, y, s=10, color=colors[i], alpha=0.6)
                
                # Update bounds
                abs_min_x = min(abs_min_x, x.min())
                abs_max_x = max(abs_max_x, x.max())
                abs_min_y = min(abs_min_y, y.min())
                abs_max_y = max(abs_max_y, y.max())

            # --- Row 2: Centered ---
            if np.any(mask_v):
                x = kpts_centered[v, mask_v, 0]
                y = kpts_centered[v, mask_v, 1]
                axes[1, v].scatter(x, y, s=10, color=colors[i], alpha=0.6)
                axes[1, v].scatter([0], [0], marker='+', color='red', s=50)

            # --- Row 3: Normalized ---
            if np.any(mask_v):
                x = kpts_norm[v, mask_v, 0]
                y = kpts_norm[v, mask_v, 1]
                axes[2, v].scatter(x, y, s=10, color=colors[i], alpha=0.6)
                
    # 4. Final Formatting
    pad = 50
    for v in range(num_views):
        # Set absolute limits
        axes[0, v].set_xlim(abs_max_x + pad, abs_min_x - pad)
        axes[0, v].set_ylim(abs_max_y + pad, abs_min_y - pad)
        
        # Set normalized limits
        axes[2, v].set_xlim(1.1, -1.1)
        axes[2, v].set_ylim(1.1, -1.1) # Note: Y-axis is already inverted above if needed

        # Common formatting
        for r in range(3):
            axes[r, v].set_aspect('equal')
            axes[r, v].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_fitting_results(noisy_points, fitted_points, gt_points, skeleton_rig, visibility_mask=None):
    noisy = noisy_points.detach().cpu().numpy()
    fitted = fitted_points.detach().cpu().numpy()
    gt = gt_points.detach().cpu().numpy()
    
    if visibility_mask is None:
        visibility_mask = torch.ones(noisy.shape[0], dtype=torch.bool)
    mask = visibility_mask.cpu().numpy()
    
    idx_to_name = {v: k for k, v in skeleton_rig.parts_map.items()}
    
    fig = go.Figure()
    
    # Add ground truth
    fig.add_trace(go.Scatter3d(
        x=gt[:, 0], y=gt[:, 1], z=gt[:, 2],
        mode='markers+text',
        name='Ground Truth',
        marker=dict(size=6, color='blue'),
        visible=True
    ))
    
    # Add noisy points
    fig.add_trace(go.Scatter3d(
        x=noisy[mask, 0], y=noisy[mask, 1], z=noisy[mask, 2],
        mode='markers',
        name='Noisy Input',
        marker=dict(size=5, color='red'),
        visible=True
    ))
    
    # Add fitted points with labels
    labels = [idx_to_name.get(i, f"pt_{i}") for i in range(len(fitted))]
    fig.add_trace(go.Scatter3d(
        x=fitted[:, 0], y=fitted[:, 1], z=fitted[:, 2],
        mode='markers+text',
        name='Fitted Prediction',
        marker=dict(size=6, color='green'),
        text=labels,
        textposition='top center',
        visible=True
    ))
    
    # Add buttons to toggle visibility
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(label="All", method="update",
                         args=[{"visible": [True, True, True]}]),
                    dict(label="GT Only", method="update",
                         args=[{"visible": [True, False, False]}]),
                    dict(label="Prediction Only", method="update",
                         args=[{"visible": [False, False, True]}]),
                    dict(label="GT + Prediction", method="update",
                         args=[{"visible": [True, False, True]}]),
                    dict(label="Noisy", method="update",
                         args=[{"visible": [False, True, False]}]),
                    dict(label="Noisy + GT", method="update",
                         args=[{"visible": [True, True, False]}]),
                ],
                x=0.0, xanchor="left", y=1.15, yanchor="top"
            ),
        ],
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title="Interactive Skeleton Fitting"
    )
    
    fig.show()
    
def get_segment(coords, k, mode, tail_len=10):
    """
    Helper to slice trail according to mode and frame index k
    
    Parameters
    ----------
    coords : torch.Tensor
        Shape (T, 3) - trajectory for a single part
    """
    if mode == "full":
        seg = coords[:k+1]
    elif mode == "tail":
        start = max(0, k - (tail_len - 1))
        seg = coords[start:k+1]
    elif mode == "current":
        seg = coords[k:k+1]
    else:
        raise ValueError(mode)
    return seg[:,0], seg[:,1], seg[:,2]


def get_init_traces(trajectory, parts, colors, max_frames=200, tail_len=10):
    """
    Prepare initial scatter3d traces for the first frame.
    
    Parameters
    ----------
    trajectory : torch.Tensor
        Shape (T, P, 3) - trajectories for all parts
    """
    initial_traces = []
    n = min(trajectory.shape[0], max_frames)
    
    for i, part in enumerate(parts):
        coords = trajectory[:n, i, :]  # Extract trajectory for part i
        x0, y0, z0 = get_segment(coords, 0, mode="tail", tail_len=tail_len)
        
        # Convert to numpy for plotly
        x0_np = x0.cpu().numpy() if x0.is_cuda else x0.numpy()
        y0_np = y0.cpu().numpy() if y0.is_cuda else y0.numpy()
        z0_np = z0.cpu().numpy() if z0.is_cuda else z0.numpy()
        
        initial_traces.append(
            go.Scatter3d(
                x=x0_np, y=y0_np, z=z0_np,
                mode='markers+lines' if len(x0_np)>1 else 'markers',
                name=part,
                marker=dict(size=4),
                line=dict(width=3),
                marker_color=colors[i],
                showlegend=True
            )
        )
    return initial_traces


def build_frames(trajectory, parts, n, mode, colors, tail_len=10):
    """
    Build animation frames for a given mode ('full', 'tail', 'current').
    
    Parameters
    ----------
    trajectory : torch.Tensor
        Shape (T, P, 3) - trajectories for all parts
    """
    frames = []
    for k in range(n):
        data = []
        for i, part in enumerate(parts):
            coords = trajectory[:n, i, :]  # Extract trajectory for part i
            x, y, z = get_segment(coords, k, mode, tail_len=tail_len)
            
            # Convert to numpy for plotly
            x_np = x.cpu().numpy() if x.is_cuda else x.numpy()
            y_np = y.cpu().numpy() if y.is_cuda else y.numpy()
            z_np = z.cpu().numpy() if z.is_cuda else z.numpy()

            data.append(go.Scatter3d(
                x=x_np, y=y_np, z=z_np,
                mode='markers+lines' if len(x_np)>1 else 'markers',
                marker_color=colors[i],
                marker=dict(size=4),
                line=dict(width=3),
                showlegend=True
            ))
        frames.append(go.Frame(data=data, name=f"{mode}_{k}"))
    return frames


def fix_aspect_ratio(trajectory, n, pad=1.0):
    """
    Compute equalized aspect ratio and padded axis ranges.
    
    Parameters
    ----------
    trajectory : torch.Tensor
        Shape (T, P, 3) - trajectories for all parts
    """
    # Reshape to (T*P, 3) and convert to numpy
    all_coords = trajectory[:n, :, :].reshape(-1, 3)
    all_coords_np = all_coords.cpu().numpy() if all_coords.is_cuda else all_coords.numpy()

    x_min, x_max = all_coords_np[:,0].min()-pad, all_coords_np[:,0].max()+pad
    y_min, y_max = all_coords_np[:,1].min()-pad, all_coords_np[:,1].max()+pad
    z_min, z_max = all_coords_np[:,2].min()-pad, all_coords_np[:,2].max()+pad

    xr = x_max - x_min
    yr = y_max - y_min
    zr = z_max - z_min
    maxr = max(xr, yr, zr)

    aspect = dict(x = xr/maxr, y = yr/maxr, z = zr/maxr)
    return aspect, [x_min, x_max], [y_min, y_max], [z_min, z_max]


def single_camera(ax, positive=True, distance=2):
    """
    Returns camera dict looking along a single axis.
    """
    d = distance if positive else -distance
    eye = {'x': 0, 'y': 0, 'z': 0}
    eye[ax] = d
    if ax == 'x':
        up = dict(x=0, y=0, z=1)
    elif ax == 'y':
        up = dict(x=0, y=0, z=1)
    else:  # z
        up = dict(x=0, y=1, z=0)
    return dict(eye=eye, up=up, center=dict(x=0, y=0, z=0))


def camera_along_axis(axis=None, positive=True, distance=2):
    """
    Return Plotly camera(s) looking along given axis or all axes if None.
    """
    if axis is None:
        return {ax: single_camera(ax, positive=positive, distance=distance) for ax in ('x', 'y', 'z')}
    else:
        axis = axis.lower()
        if axis not in ('x', 'y', 'z'):
            raise ValueError("axis must be 'x', 'y', 'z', or None")
        return single_camera(axis, positive=positive, distance=distance)


def animate(trajectory, parts, max_frames=200, tail_len=10, pad=1.0, direction='z', positive=True, distance=2.0, frame_duration=100):
    """
    Animate 3D trajectories of multiple parts using Plotly.

    Parameters
    ----------
    trajectory : torch.Tensor
        Tensor of shape (T, P, 3) where T is the number of time steps,
        P is the number of parts, and the last dimension contains (x, y, z) coordinates.
    parts : list of str
        List of part names to display in the legend. Length must match P.
    max_frames : int, default=200
        Maximum number of frames to animate.
    tail_len : int, default=10
        Number of trailing frames to show when using the "tail" animation mode.
    pad : float, default=1.0
        Padding added to the 3D axis ranges to prevent trajectories from touching plot boundaries.
    direction : str, default='z'
        Axis along which the camera is oriented. Must be one of 'x', 'y', or 'z'.
    positive : bool, default=True
        If True, camera looks in the positive direction of the specified axis; otherwise negative.
    distance : float, default=2.0
        Distance of the camera from the origin along the chosen axis.

    Returns
    -------
        Displays an interactive Plotly figure with the animated 3D trajectories.
    """
    # Validate input shape
    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError(f"trajectory must be shape (T, P, 3). Got {trajectory.shape}")
    
    T, P, _ = trajectory.shape
    
    if len(parts) != P:
        raise ValueError(f"Length of parts ({len(parts)}) must match P dimension ({P})")
    
    palette = px.colors.qualitative.Plotly
    colors = [palette[i % len(palette)] for i in range(len(parts))]
    
    n = min(T, max_frames)  # Number of frames

    # First frame
    initial_traces = get_init_traces(trajectory, parts, colors, max_frames=max_frames, tail_len=tail_len)

    # Build frames for each mode
    frames_full = build_frames(trajectory, parts, n, 'full', colors, tail_len=tail_len)
    frames_tail = build_frames(trajectory, parts, n, 'tail', colors, tail_len=tail_len)
    frames_current = build_frames(trajectory, parts, n, 'current', colors, tail_len=tail_len)

    # Build the figure 
    fig = go.Figure(
        data=initial_traces,
        layout=go.Layout(
            title=f"3D Trajectories (tail {tail_len} steps)",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='manual'),
            updatemenus=[
                dict(type="buttons", showactive=False, x=0.05, y=0.05,
                    buttons=[dict(label=f"Play (Last {tail_len} steps)",
                                method="animate",
                                args=[[f.name for f in frames_tail],
                                      # USE frame_duration HERE
                                      {"frame": {"duration": frame_duration, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}])]),
                dict(type="buttons", showactive=False, x=0.30, y=0.05,
                    buttons=[dict(label="Play (Full trail)",
                                method="animate",
                                args=[[f.name for f in frames_full],
                                      # USE frame_duration HERE
                                      {"frame": {"duration": frame_duration, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}])]),
                dict(type="buttons", showactive=False, x=0.55, y=0.05,
                    buttons=[dict(label="Play (Current only)",
                                method="animate",
                                args=[[f.name for f in frames_current],
                                      # USE frame_duration HERE
                                      {"frame": {"duration": frame_duration, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 0}}])]),
                dict(type="buttons", showactive=False, x=0.80, y=0.05,
                    buttons=[dict(label="Pause",
                                method="animate",
                                args=[[], {"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate", "transition": {"duration": 0}}])])
            ]
        ),
        frames=frames_tail + frames_full + frames_current
    )


    aspect, x_range, y_range, z_range = fix_aspect_ratio(trajectory, n, pad=pad)
    camera = camera_along_axis(direction, positive, distance)
    fig.update_layout(
        scene = dict(
            xaxis=dict(range=x_range, autorange=False, title='X'),
            yaxis=dict(range=y_range, autorange=False, title='Y'),
            zaxis=dict(range=z_range, autorange=False, title='Z'),
            aspectmode='manual',
            aspectratio=aspect,
            camera=camera
        ),
        uirevision='fixed_view'
    )

    fig.show()
