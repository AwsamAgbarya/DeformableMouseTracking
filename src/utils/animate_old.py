import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

RIGID_COLOR    = "royalblue"
DEFORM_COLOR   = "tomato"
BOND_COLOR     = "mediumpurple"
RIGID_EDGE_COL = "steelblue"
NODE_SIZE      = 5

# OAT + data visualization
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

# data visualization
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
    Slice a single-node trajectory for the current frame.

    Parameters
    ----------
    coords : Tensor (T, 3)
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
    return seg[:, 0], seg[:, 1], seg[:, 2]


def _np(t):
    """Tensor → numpy, cpu-safe."""
    return t.detach().cpu().numpy()


def _edge_lines(pos2K, edge_index, edge_attr, K):
    """
    Build a list of go.Scatter3d traces (one per edge type group) for the
    skeleton edges at a single frame.

    Parameters
    ----------
    pos2K      : Tensor (2K, 3)  positions of all nodes at one frame
    edge_index : Tensor (2, E)
    edge_attr  : Tensor (E, 3)   [:, 2] = edge_type
    K          : int
    """
    edge_types = _np(edge_attr[:, 2]).astype(int)
    src = _np(edge_index[0])
    dst = _np(edge_index[1])
    pos = _np(pos2K)

    # Separate by type
    type_meta = {
        0: dict(color=RIGID_EDGE_COL, name="rigid bone",  dash="solid"),
        1: dict(color=BOND_COLOR,     name="deform bond", dash="dash"),
    }

    traces = []
    for etype, meta in type_meta.items():
        mask = edge_types == etype
        xs, ys, zs = [], [], []
        for s, d in zip(src[mask], dst[mask]):
            xs += [pos[s, 0], pos[d, 0], None]
            ys += [pos[s, 1], pos[d, 1], None]
            zs += [pos[s, 2], pos[d, 2], None]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=meta["color"], width=3, dash=meta["dash"]),
            name=meta["name"],
            showlegend=True,
        ))
    return traces


def _node_traces(pos2K, K, part_names):
    pos    = _np(pos2K)
    rigid  = pos[:K]
    deform = pos[K:]                       # ← K_active rows, not K

    r_trace = go.Scatter3d(
        x=rigid[:, 0], y=rigid[:, 1], z=rigid[:, 2],
        mode="markers",
        marker=dict(size=NODE_SIZE, color=RIGID_COLOR),
        name="rigid nodes",
        text=part_names[:K],
        hovertemplate="%{text}<extra>rigid</extra>",
        showlegend=True,
    )
    d_trace = go.Scatter3d(
        x=deform[:, 0], y=deform[:, 1], z=deform[:, 2],
        mode="markers",
        marker=dict(size=NODE_SIZE, color=DEFORM_COLOR, symbol="diamond"),
        name="deform nodes",
        text=part_names[K:],               # ← only K_active names
        hovertemplate="%{text}<extra>deform</extra>",
        showlegend=True,
    )
    return [r_trace, d_trace]


def _trail_traces(trajectory, K, k, mode, tail_len, part_names):
    palette = px.colors.qualitative.Plotly
    traces = []
    N_nodes = trajectory.shape[1]          # ← K + K_active, not 2*K
    for i in range(N_nodes):
        coords = trajectory[:, i, :]
        x, y, z = get_segment(coords, k, mode, tail_len)
        x_np, y_np, z_np = _np(x), _np(y), _np(z)
        is_rigid = i < K
        name = part_names[i]               # ← already correct length now
        color = palette[i % K]
        opacity = 0.9 if is_rigid else 0.45
        traces.append(go.Scatter3d(
            x=x_np, y=y_np, z=z_np,
            mode="lines" if len(x_np) > 1 else "markers",
            line=dict(width=2, color=color),
            marker=dict(size=2, color=color),
            opacity=opacity,
            name=name,
            showlegend=False,
        ))
    return traces


def _all_traces_at_frame(trajectory, edge_index, edge_attr, K, part_names, k, mode, tail_len):
    """Return the complete trace list for one animation frame."""
    pos2K = trajectory[k]                          # (2K, 3)
    return (
        _edge_lines(pos2K, edge_index, edge_attr, K) if edge_index is not None and edge_index is not None else None
        + _node_traces(pos2K, K, part_names)
        + _trail_traces(trajectory, K, k, mode, tail_len, part_names)
    )


def fix_aspect_ratio(trajectory, n, pad=1.0):
    all_np = _np(trajectory[:n].reshape(-1, 3))
    ranges = [(all_np[:, i].min() - pad, all_np[:, i].max() + pad) for i in range(3)]
    spans  = [r[1] - r[0] for r in ranges]
    maxr   = max(spans)
    aspect = dict(x=spans[0]/maxr, y=spans[1]/maxr, z=spans[2]/maxr)
    return aspect, list(ranges[0]), list(ranges[1]), list(ranges[2])


def single_camera(ax, positive=True, distance=2):
    d = distance if positive else -distance
    eye = {"x": 0, "y": 0, "z": 0}
    eye[ax] = d
    up = dict(x=0, y=0, z=1) if ax in ("x", "y") else dict(x=0, y=1, z=0)
    return dict(eye=eye, up=up, center=dict(x=0, y=0, z=0))


def animate(
    trajectory,
    edge_index,
    edge_attr,
    K,
    part_names=None,
    max_frames=200,
    tail_len=10,
    pad=1.0,
    direction="z",
    positive=True,
    distance=2.0,
    frame_duration=100,
):
    """
    Animate the dual-layer skeleton (rigid + deformable) over time.

    Parameters
    ----------
    trajectory     : Tensor (T, 2K, 3)   all node positions over time.
                     Use dataset.all_c for COM-centred or
                     torch.cat([dataset.rigid_coords, dataset.deformable_coords], dim=1)
                     for world-space.
    edge_index     : Tensor (2, E)        from dataset.edge_index
    edge_attr      : Tensor (E, 3)        from dataset.edge_attr
    K              : int                  number of keypoints (dataset.K)
    part_names     : list[str] | None     length K; defaults to DEFAULT_PART_NAMES
    max_frames     : int
    tail_len       : int
    pad            : float
    direction      : 'x' | 'y' | 'z'    camera view axis
    positive       : bool
    distance       : float
    frame_duration : int                 ms per animation frame
    """
    if part_names is None:
        part_names = [f"node_{i}" for i in range(K)]

    T = min(trajectory.shape[0], max_frames)

    # ── initial data (frame 0, tail mode) ──
    initial_traces = _all_traces_at_frame(
        trajectory, edge_index, edge_attr, K, part_names, 0, "tail", tail_len
    )

    # ── animation frames ──
    all_frames = []
    for mode in ("tail", "full", "current"):
        for k in range(T):
            data = _all_traces_at_frame(
                trajectory, edge_index, edge_attr, K, part_names, k, mode, tail_len
            )
            all_frames.append(go.Frame(data=data, name=f"{mode}_{k}"))

    # ── layout ──
    aspect, xr, yr, zr = fix_aspect_ratio(trajectory, T, pad)
    camera = single_camera(direction, positive, distance)

    play_args = lambda mode: [
        [f"{mode}_{k}" for k in range(T)],
        {"frame": {"duration": frame_duration, "redraw": True},
         "fromcurrent": True, "transition": {"duration": 0}},
    ]
    pause_args = [[], {"frame": {"duration": 0, "redraw": False},
                       "mode": "immediate", "transition": {"duration": 0}}]

    fig = go.Figure(
        data=initial_traces,
        layout=go.Layout(
            title=f"Dual-layer skeleton — tail={tail_len} frames",
            scene=dict(
                xaxis=dict(range=xr, autorange=False, title="X"),
                yaxis=dict(range=yr, autorange=False, title="Y"),
                zaxis=dict(range=zr, autorange=False, title="Z"),
                aspectmode="manual",
                aspectratio=aspect,
                camera=camera,
            ),
            uirevision="fixed_view",
            updatemenus=[
                dict(type="buttons", showactive=False, x=0.05, y=0.05,
                     buttons=[dict(label=f"▶ Tail ({tail_len})",
                                   method="animate", args=play_args("tail"))]),
                dict(type="buttons", showactive=False, x=0.28, y=0.05,
                     buttons=[dict(label="▶ Full trail",
                                   method="animate", args=play_args("full"))]),
                dict(type="buttons", showactive=False, x=0.52, y=0.05,
                     buttons=[dict(label="▶ Current only",
                                   method="animate", args=play_args("current"))]),
                dict(type="buttons", showactive=False, x=0.75, y=0.05,
                     buttons=[dict(label="⏸ Pause",
                                   method="animate", args=pause_args)]),
            ],
        ),
        frames=all_frames,
    )

    fig.show()

def visualize_rigid_deformable(
    rigid_coords: np.ndarray,
    deformable_coords: np.ndarray,
    joint_names: list = None,
    frame_idx: int = 0,
    title: str = "Rigid vs Deformable Joint Displacements",
) -> go.Figure:
    # ── Select frame ──────────────────────────────────────────────────────────
    if rigid_coords.ndim == 3:
        R = rigid_coords[frame_idx]       # (N, 3)
        D = deformable_coords[frame_idx]
    else:
        R = rigid_coords.copy()
        D = deformable_coords.copy()

    N = R.shape[0]
    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(N)]

    # ── Per-joint displacement ────────────────────────────────────────────────
    deltas     = D - R                                              # (N, 3)
    magnitudes = np.linalg.norm(deltas, axis=-1)                   # (N,)
    mag_norm   = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-8)
    # Cool (blue, 220°) → stiff; Warm (red, 0°) → flexible
    colors = [f"hsl({int((1 - m) * 220)}, 80%, 55%)" for m in mag_norm]

    axis_style = dict(
        showgrid=True, gridcolor="#393836", zeroline=False,
        showbackground=True, backgroundcolor="#171614", color="#7a7974",
    )

    fig = go.Figure()

    # ── Rigid anchor positions ────────────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=R[:, 0], y=R[:, 1], z=R[:, 2],
        mode="markers+text",
        name="Rigid anchor",
        marker=dict(size=7, color="#4f98a3", symbol="circle",
                    line=dict(color="white", width=1)),
        text=joint_names,
        textposition="top center",
        textfont=dict(size=10, color="#cdccca"),
        hovertemplate="<b>%{text}</b><br>Rigid: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
    ))

    # ── Deformable observed positions ─────────────────────────────────────────
    fig.add_trace(go.Scatter3d(
        x=D[:, 0], y=D[:, 1], z=D[:, 2],
        mode="markers",
        name="Deformable (observed)",
        marker=dict(size=7, color="#fdab43", symbol="diamond",
                    line=dict(color="white", width=1)),
        text=joint_names,
        hovertemplate="<b>%{text}</b><br>Deformable: (%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
    ))

    # ── Displacement arrows: rigid → deformable ───────────────────────────────
    for i in range(N):
        rx, ry, rz = R[i]
        dx, dy, dz = D[i]
        mag = magnitudes[i]

        # Arrow shaft
        fig.add_trace(go.Scatter3d(
            x=[rx, dx], y=[ry, dy], z=[rz, dz],
            mode="lines",
            name="δ vectors",
            line=dict(color=colors[i], width=4),
            hovertemplate=(
                f"<b>{joint_names[i]}</b><br>"
                f"Δx={dx-rx:.3f}  Δy={dy-ry:.3f}  Δz={dz-rz:.3f}<br>"
                f"|δ| = {mag:.4f}<extra></extra>"
            ),
            showlegend=(i == 0),
            legendgroup="arrows",
        ))

        # Arrowhead cone at deformable end
        fig.add_trace(go.Cone(
            x=[dx], y=[dy], z=[dz],
            u=[dx - rx], v=[dy - ry], w=[dz - rz],
            sizemode="absolute",
            sizeref=max(magnitudes) * 0.15 + 1e-6,
            colorscale=[[0, colors[i]], [1, colors[i]]],
            showscale=False,
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Translucent halo at rigid origin (local frame indicator) ──────────────
    fig.add_trace(go.Scatter3d(
        x=R[:, 0], y=R[:, 1], z=R[:, 2],
        mode="markers",
        name="Rigid origin halo",
        marker=dict(size=14, color="rgba(79,152,163,0.12)", symbol="circle"),
        hoverinfo="skip",
        showlegend=True,
    ))

    # ── Magnitude legend annotation ───────────────────────────────────────────
    ann_text = "<b>|δ| per joint</b><br>" + "<br>".join(
        f"  {n}: {m:.4f}" for n, m in zip(joint_names, magnitudes)
    )
    fig.add_annotation(
        text=ann_text, xref="paper", yref="paper", x=0.01, y=0.97,
        align="left", showarrow=False,
        font=dict(size=10, color="#7a7974"),
        bgcolor="rgba(28,27,25,0.85)", bordercolor="#393836", borderwidth=1,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#cdccca"), x=0.5),
        paper_bgcolor="#1c1b19",
        plot_bgcolor="#1c1b19",
        scene=dict(
            bgcolor="#1c1b19",
            xaxis=dict(**axis_style, title="X"),
            yaxis=dict(**axis_style, title="Y"),
            zaxis=dict(**axis_style, title="Z"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode="data",
        ),
        legend=dict(
            font=dict(color="#cdccca", size=11),
            bgcolor="rgba(28,27,25,0.8)",
            bordercolor="#393836", borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
    )

    return fig