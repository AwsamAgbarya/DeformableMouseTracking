import matplotlib.pyplot as plt
import os
import shutil

import plotly.graph_objects as go
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import FancyArrowPatch
from utils.pose_extraction import forward_facing

# Path plotting function that displays a top down view + direction
def plot_direction(path, forward_vec, base_index=0, step = 50, output_path=None):
    dx, dy = forward_vec[:, 0], forward_vec[:, 1]
    x, y = path[:, base_index, 0], path[:, base_index, 1]
    xs, ys = x[::step], y[::step]
    us, vs = dx[::step], dy[::step]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(x, y, s=1, zorder=1)  # plain path, no color-by-progress
    ax.quiver(xs, ys, us, vs,
            angles='xy', scale_units='xy',
            scale=None,          # let matplotlib auto-scale, or set manually below
            width=0.003,
            headwidth=4, headlength=6, headaxislength=5,
            color='crimson', zorder=5)
    ax.scatter(x[0], y[0], c='green', s=80, label='start', zorder=6)
    ax.scatter(x[-1], y[-1], c='black', s=80, marker='s', label='end', zorder=6)

    ax.set_aspect('equal')
    plt.legend()
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()

# Plotting side by side the movement + Path of the dataset in video format
def visualize_skeleton_motion(
    keypoints,
    skeleton_edges,
    output_path="skeleton_motion.mp4",
    fps=30,
    stride=1,
    root_index=None,
    forward_hip_indices=None,
    forward_sign=1,
    up_axis="z",
    trail_frames=60,
    auto_rotate_deg_per_frame=0.0,
    figsize=(14, 6.5),
    dpi=150,
    elev=15,
    azim=-70,
    point_size=25,
    line_width=2.5,
    bone_color="tab:blue",
    joint_color="tab:orange",
    path_color="tab:blue",
    current_color="crimson",
):
    """
    Render a video of an animated skeleton moving through 3D space.

    The output has two synchronized panels, designed specifically so that
    a small skeleton moving inside a huge bounding volume is still clearly
    visible (the bounding volume itself is never used for scaling):

      LEFT  - "local pose": the skeleton re-centered on its root every
              frame, so you clearly see the articulated motion (limbs
              moving relative to each other) at a fixed, readable scale.
      RIGHT - "global trajectory": a top-down (bird's-eye) map of the
              root's path through the world, auto-scaled to the actual
              path traveled, with a fading trail and a heading arrow, so
              you clearly see the global path (straight, U-turn, etc.).

    Parameters
    ----------
    keypoints : torch.Tensor or np.ndarray, shape (T, N, 3)
        3D keypoint positions over T frames for N joints.
    skeleton_edges : array-like, shape (M, 2)
        Pairs of joint indices (i, j) connected by a bone.
    output_path : str
        Where to save the video. '.mp4' needs ffmpeg on PATH; '.gif' uses
        Pillow and always works but is lower quality / larger for long
        sequences.
    fps : int
        Playback frame rate of the output video.
    stride : int
        Use every `stride`-th input frame (useful to speed up rendering,
        or shrink very long sequences). Note this speeds up perceived
        motion unless you also reduce `fps` to compensate.
    root_index : int or None
        Joint index to treat as the "root" for centering and for the
        trajectory panel. If None (default), the centroid of all joints
        each frame is used instead.
    forward_hip_indices : (int, int) or None
        Optional (left_joint_idx, right_joint_idx) pair (e.g. left/right
        hip or shoulder). If given, the heading arrow on the trajectory
        panel shows the body's true facing direction (perpendicular to
        the line between these two joints) instead of just the direction
        of travel. If the arrow points backwards for your skeleton, swap
        the two indices or set forward_sign=-1.
    forward_sign : {1, -1}
        Flips the computed facing direction; only relevant when
        `forward_hip_indices` is set.
    up_axis : {'x', 'y', 'z'}
        Which axis of your data is vertical. Controls the orientation of
        the 3D pose plot and which plane is used for the top-down map.
    trail_frames : int
        Number of recent frames drawn bold/opaque on the trajectory
        trail; older parts of the (fully drawn) path fade out.
    auto_rotate_deg_per_frame : float
        If nonzero, slowly orbits the camera around the local-pose plot
        by this many degrees per frame -- helpful for revealing depth
        (front/back movement) that a fixed viewpoint can hide. 0 = static
        camera (default; usually easier to read for analysis).
    figsize, dpi : matplotlib figure size / output resolution.
    elev, azim : initial viewing angle (degrees) for the local pose plot.
    point_size, line_width : joint marker size / bone line width.
    bone_color, joint_color, path_color, current_color : color overrides.

    Returns
    -------
    str
        The `output_path` the video was written to.
    """
    # ---- Normalize inputs --------------------------------------------
    kpts = _to_numpy(keypoints)
    assert kpts.ndim == 3 and kpts.shape[-1] == 3, (
        f"expected keypoints of shape (T, N, 3), got {kpts.shape}"
    )
    kpts = kpts[::stride]
    T, N, _ = kpts.shape

    edges = _to_numpy(skeleton_edges).astype(int)
    assert edges.ndim == 2 and edges.shape[1] == 2, (
        f"expected skeleton_edges of shape (M, 2), got {edges.shape}"
    )
    assert edges.min() >= 0 and edges.max() < N, "skeleton_edges index out of range"

    # ---- Root / centroid path and locally-centered pose ---------------
    if root_index is not None:
        root = kpts[:, root_index, :]
    else:
        root = kpts.mean(axis=1)
    local_kpts = kpts - root[:, None, :]

    local_disp = _reorder_for_display(local_kpts, up_axis)  # (T, N, 3)
    root_disp = _reorder_for_display(root, up_axis)  # (T, 3)

    # Precompute true facing direction if hip/shoulder indices were given
    forward_vec = None
    if forward_hip_indices is not None:
        forward_vec, norms = forward_facing(kpts, forward_hip_indices, forward_sign)

        # Also un-rotate the local pose by the body's heading each frame, so
        # the left panel shows a *facing-normalized* pose: pure articulation
        # (arm/leg swing) with global turning fully factored out into the
        # right-hand panel instead of making the body spin on screen.
        theta = np.arctan2(forward_vec[:, 1], forward_vec[:, 0])
        c, s = np.cos(-theta), np.sin(-theta)
        gx, gy = local_disp[..., 0].copy(), local_disp[..., 1].copy()
        local_disp[..., 0] = c[:, None] * gx - s[:, None] * gy
        local_disp[..., 1] = s[:, None] * gx + c[:, None] * gy

    # ---- Figure / axes setup -------------------------------------------
    axis_names = ["X", "Y", "Z"]
    up_i = _AXIS_IDX[up_axis]
    g0_name, g1_name = [axis_names[i] for i in range(3) if i != up_i]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    fig.suptitle("Skeleton Motion", fontsize=14, fontweight="bold")
    pose_title = "Local pose (centered on root"
    pose_title += ", facing-normalized)" if forward_vec is not None else ")"
    ax3d.set_title(pose_title)
    heading_label = "facing direction" if forward_vec is not None else "direction of travel"
    ax_top.set_title(f"Global trajectory (top-down)\narrow = {heading_label}")

    # Fixed, equal-aspect limits for the local skeleton view
    max_range = max(np.abs(local_disp).max() * 1.15, 1e-6)
    ax3d.set_xlim(-max_range, max_range)
    ax3d.set_ylim(-max_range, max_range)
    ax3d.set_zlim(-max_range, max_range)
    try:
        ax3d.set_box_aspect([1, 1, 1])
    except AttributeError:  # pragma: no cover - older matplotlib
        pass
    ax3d.view_init(elev=elev, azim=azim)
    if forward_vec is not None:
        ax3d.set_xlabel("forward")
        ax3d.set_ylabel("side")
    else:
        ax3d.set_xlabel(g0_name)
        ax3d.set_ylabel(g1_name)
    ax3d.set_zlabel(up_axis.upper())

    # Fixed, equal-aspect limits for the top-down trajectory view,
    # scaled to the *actual path*, never to the surrounding cage.
    gx_min, gx_max = root_disp[:, 0].min(), root_disp[:, 0].max()
    gy_min, gy_max = root_disp[:, 1].min(), root_disp[:, 1].max()
    span = max(gx_max - gx_min, gy_max - gy_min, 1e-6)
    pad = span * 0.15
    cx, cy = (gx_min + gx_max) / 2, (gy_min + gy_max) / 2
    half = span / 2 + pad
    ax_top.set_xlim(cx - half, cx + half)
    ax_top.set_ylim(cy - half, cy + half)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel(g0_name)
    ax_top.set_ylabel(g1_name)
    ax_top.grid(True, alpha=0.3)

    # ---- Artists updated every frame -----------------------------------
    bone_lines = [
        ax3d.plot([], [], [], "-", lw=line_width, color=bone_color, solid_capstyle="round")[0]
        for _ in range(len(edges))
    ]
    joint_scatter = ax3d.scatter([], [], [], s=point_size, color=joint_color, depthshade=True)

    full_path_line, = ax_top.plot([], [], "-", lw=1.0, color=path_color, alpha=0.35)
    trail_line, = ax_top.plot([], [], "-", lw=2.5, color=path_color, alpha=0.9)
    current_pt = ax_top.scatter([], [], s=70, color=current_color, zorder=5)
    heading_arrow = FancyArrowPatch(
        (0, 0), (0, 0), arrowstyle="-|>", mutation_scale=18,
        color=current_color, zorder=6, lw=1.5,
    )
    ax_top.add_patch(heading_arrow)

    frame_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10)

    fig.tight_layout(rect=[0, 0.04, 1, 0.90])

    def update(frame_idx):
        pose = local_disp[frame_idx]

        for line, (i, j) in zip(bone_lines, edges):
            _set_line3d(line, [pose[i, 0], pose[j, 0]], [pose[i, 1], pose[j, 1]], [pose[i, 2], pose[j, 2]])
        joint_scatter._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])

        if auto_rotate_deg_per_frame:
            ax3d.view_init(elev=elev, azim=azim + frame_idx * auto_rotate_deg_per_frame)

        xs = root_disp[: frame_idx + 1, 0]
        ys = root_disp[: frame_idx + 1, 1]
        full_path_line.set_data(xs, ys)

        start = max(0, frame_idx + 1 - trail_frames)
        trail_line.set_data(root_disp[start : frame_idx + 1, 0], root_disp[start : frame_idx + 1, 1])

        current_pt.set_offsets([[xs[-1], ys[-1]]])

        if forward_vec is not None:
            dx, dy = forward_vec[frame_idx]
        elif frame_idx > 0:
            back = max(0, frame_idx - 5)
            dx = root_disp[frame_idx, 0] - root_disp[back, 0]
            dy = root_disp[frame_idx, 1] - root_disp[back, 1]
        else:
            dx, dy = 0.0, 0.0
        norm = np.hypot(dx, dy)
        if norm > 1e-9:
            scale = span * 0.06
            dx, dy = dx / norm * scale, dy / norm * scale
        heading_arrow.set_positions((xs[-1], ys[-1]), (xs[-1] + dx, ys[-1] + dy))

        frame_text.set_text(f"frame {frame_idx + 1}/{T}    t = {frame_idx / fps:.2f}s")

        return bone_lines + [joint_scatter, full_path_line, trail_line, current_pt, heading_arrow, frame_text]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    # ---- Save ------------------------------------------------------------
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH but an .mp4 output was requested. "
                "Install ffmpeg, or set output_path to end in '.gif' instead."
            )
        writer = FFMpegWriter(fps=fps, bitrate=4000)

    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path

# Plotting outliers of a dataset
def plot_outlier_analysis(angles, output_path, threshold=10.0):
    """Plot angular change distributions and outlier detection"""
    fig = plt.figure(figsize=(16, 12))
    # Plot angular changes over frames
    frames = np.arange(len(angles))
    plt.plot(frames, angles, linewidth=1, alpha=0.7, color='steelblue')

    # Mark outliers
    outlier_mask = angles > threshold
    n_outliers = outlier_mask.sum().item()
    outlier_x = frames[outlier_mask]
    outlier_y = angles[outlier_mask]
    plt.scatter(outlier_x, outlier_y, color='red', s=50, zorder=5, 
                label=f'Outliers (n={n_outliers})', alpha=0.8)

    # Add threshold line
    mean_angle = angles.mean()
    max_angle = angles.max()
    plt.axhline(y=threshold, color='red', linestyle='--', 
                alpha=0.5, label=f'Threshold ({threshold:.1f}°)')
    plt.axhline(y=mean_angle, color='green', linestyle='--', 
                alpha=0.5, label=f'Mean ({max_angle:.1f}°)')

    plt.xlabel('Frame Index')
    plt.ylabel('Angular Change (degrees)')
    plt.title(f'Angular Changes Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, min(200, max_angle * 1.1))
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Plot side by side frame comparison
def plot_skeleton_comparison_plotly(frame_a, frame_b, edges,
                                     labels=('A', 'B'),
                                     colors=('royalblue', 'crimson')):
    """Interactive 3D comparison of two (N,3) keypoint frames as skeletons.

    frame_a, frame_b : (N,3) arrays of joint coordinates
    edges            : list of (i, j) index pairs defining bones
    """
    fig = go.Figure()

    for frame, color, label in zip((frame_a, frame_b), colors, labels):
        xe, ye, ze = [], [], []
        for i, j in edges:
            xe += [frame[i, 0], frame[j, 0], None]
            ye += [frame[i, 1], frame[j, 1], None]
            ze += [frame[i, 2], frame[j, 2], None]

        fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode='lines',
                                    line=dict(color=color, width=4),
                                    name=f'{label} bones', showlegend=False))
        fig.add_trace(go.Scatter3d(x=frame[:, 0], y=frame[:, 1], z=frame[:, 2],
                                    mode='markers', marker=dict(color=color, size=4),
                                    name=label))

    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=30, b=0))
    return fig

# ----------------------------------- Helpers
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
def _reorder_for_display(coords, up_axis):
    """
    Permute the last dimension (size 3) of `coords` so that column 2 is
    always the vertical/"up" axis and columns 0-1 are the ground plane.
    This lets everything downstream (3D view, top-down view) be written
    once, regardless of whether the data's vertical axis is x, y, or z.
    """
    up = _AXIS_IDX[up_axis]
    ground = [i for i in range(3) if i != up]
    return coords[..., ground + [up]]
def _set_line3d(line, xs, ys, zs):
    if hasattr(line, "set_data_3d"):
        line.set_data_3d(xs, ys, zs)
    else:  # pragma: no cover - fallback for older matplotlib
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
