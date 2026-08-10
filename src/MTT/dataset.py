import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from collections import deque

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
    (0, 1),   # head center → tail base
    (0, 2),   # head center → nose
    (0, 3),   # head center → right ear
    (0, 4),   # head center → left ear
    (1, 5),   # tail base   → tail root
    (1, 8),   # tail base   → left hind paw
    (1, 9),   # tail base   → right hind paw
    (0, 6),   # head center → left front paw
    (0, 7),   # head center → right front paw
]

DEFAULT_PART_GROUPS = {
    "head":      [0, 2, 3, 4],   # head, nose, ears
    "tail":      [1, 5],      # tail
    "paw_FL":    [6],
    "paw_FR":    [7],
    "paw_BL":    [8],
    "paw_BR":    [9],
}

# Edge type integer codes stored in edge_attr[:, 2]
EDGE_RIGID_BONE  = 0   # rigid_parent  → rigid_child       (kinematic tree)
EDGE_DEFORM_BOND = 1   # rigid_k       → deform_k          (deformation bond)


class MotionPredictionDataset(Dataset):
    """
    Sliding-window dataset for equivariant motion prediction using a dual-layer skeleton (rigid + deformable nodes).
    """ 

    def __init__(self, file_path, pred_dir, rigid_edges = DEFAULT_RIGID_EDGES, part_groups = DEFAULT_PART_GROUPS, part_names = DEFAULT_PART_NAMES, T_in = 3, T_out = 1, eps_mean=-1, eps_std=-1):
        """
        rigid_edges  : kinematic tree as (parent_idx, child_idx) tuples over the rigid part indices
        part_groups  : dict mapping group name -> list of rigid part indices
        part_names   : ordered list of part names matching the CSV "part" column
        T_in         : number of history frames per sample
        T_out        : number of future frames to predict
        """
        self.T_in  = T_in
        self.T_out = T_out
        
        raw = pd.read_csv(file_path)
        # Sort properly
        part_order = {p: i for i, p in enumerate(part_names)}
        self.N = len(part_order)
        # raw["_ord"] = raw["part"].map(part_order)
        # order = np.argsort(raw["_ord"].to_numpy().reshape(-1, self.N), axis=1)
        # raw = raw.sort_values(["time", "_ord"]).drop(columns=["_ord"])

        raw = raw.drop(columns=["part", "time"])

        # Raw coords:  (T, K, 3)
        self.rigid_coords = torch.tensor(raw.iloc[:, :3].to_numpy(), dtype=torch.float32).view(-1, self.N, 3)
        self.deformable_coords = torch.tensor(raw.iloc[:, 3:].to_numpy(), dtype=torch.float32).view(-1, self.N, 3)
        self.fixed_mask, self.active_mask = detect_fixed_nodes(self.rigid_coords, self.deformable_coords, eps_mean, eps_std)
        self.T = self.rigid_coords.shape[0]

        self.active_rigid_ids = self.active_mask.nonzero(as_tuple=True)[0]
        self.N_active = self.active_rigid_ids.shape[0]
        self.deform_active = self.deformable_coords[:, self.active_rigid_ids, :]

        # Global body motion from COM of rigid layer
        self.com          = self.rigid_coords.mean(dim=1)
        self.global_vel   = finite_diff(self.com)
        self.global_speed = self.global_vel.norm(dim=-1, keepdim=True)
        self.global_accel = finite_diff(self.global_vel)

        # COM-centred node positions
        rigid_c  = self.rigid_coords - self.com.unsqueeze(1)
        deform_c = self.deform_active - self.com.unsqueeze(1)
        self.all_c    = torch.cat([rigid_c, deform_c], dim=1)

        # Per-node kinematics
        self.node_vel       = finite_diff(self.all_c)
        self.node_speed     = self.node_vel.norm(dim=-1, keepdim=True)
        self.node_accel     = finite_diff(self.node_vel)
        self.node_accel_mag = self.node_accel.norm(dim=-1, keepdim=True)

        # Relative distances

        # Translation invariance
        self.relative_dist = self.deformable_coords - self.rigid_coords

        # Noisy Relative predictions
        file_names = os.listdir(pred_dir)
        all_relative_dists = []
        batch_idx = torch.arange(self.T).unsqueeze(1)

        for file in file_names:
            preds = torch.load(os.path.join(pred_dir, file))
            pred_deform = preds['deformable']
            pred_rigid = preds['rigid']
            
            # Extract active nodes and calculate predicted relative distance
            pred_deform_active = pred_deform#[batch_idx, order]
            pred_rigid_active = pred_rigid#[batch_idx, order]
            
            all_relative_dists.append( (pred_deform_active - pred_rigid_active)[None, :] )

        self.pred_dists = torch.cat(all_relative_dists, dim=0)
        self.pred_dataset = 0
        # Rotation invariance
        part1, part2 = ("head center", "tail base")
        i1 = part_order[part1]
        i2 = part_order[part2]

        x1 = self.rigid_coords[:, i1, 0]
        y1 = self.rigid_coords[:, i1, 1]
        x2 = self.rigid_coords[:, i2, 0]
        y2 = self.rigid_coords[:, i2, 1]

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

        R = torch.zeros((self.T, 3, 3), dtype=torch.float32)
        R[:, 2, 2] = 1.0
        R[:, 0, 0] = cos_a
        R[:, 0, 1] = -sin_a
        R[:, 1, 0] = sin_a
        R[:, 1, 1] = cos_a
        self.relative_aligned_dist = torch.einsum('fij,fpj->fpi', R, self.relative_dist)[:, self.active_rigid_ids, :]
        self.relative_dist = self.relative_dist[:, self.active_rigid_ids, :]
        self.pred_aligned_dists = torch.einsum('fij, nfpj -> nfpi', R, self.pred_dists)[:, :, self.active_rigid_ids, :]
        self.pred_dists = self.pred_dists[:, :, self.active_rigid_ids, :]

        self.normalizer = DeformationNormalizer(self.relative_aligned_dist)
        self.relative_aligned_dist_norm = self.normalizer.normalize(self.relative_aligned_dist)
        self.pred_aligned_dists_norm = self.normalizer.normalize(self.pred_aligned_dists)
        # build pruned dual-skeleton graph 
        # Mean positions tensor shaped (K + N_active, 3) for edge length calc
        mean_rigid  = self.rigid_coords.mean(dim=0)           # (K, 3)
        mean_deform = self.deform_active.mean(dim=0)          # (N_active, 3)
        mean_pos_compact = torch.cat([mean_rigid, mean_deform], dim=0)

        # Build dual-layer skeleton graph 
        self.edge_index, self.ref_bone_lengths, self.node_type, self.group_id = build_dual_skeleton_graph(self.N, self.N_active, self.active_rigid_ids, rigid_edges, part_groups, mean_pos_compact  )

        # Skeleton stats
        bond_lengths = self.relative_dist.norm(dim=-1)   # (T, N_active)
        self.bond_mean = bond_lengths.mean(dim=0)         # (N_active,)
        self.bond_std  = bond_lengths.std(dim=0)          # (N_active,)
        N_nodes = self.N + self.N_active
        self.adj, self.hop_dist = build_adjacency_matrix(self.edge_index, N_nodes, self.group_id)

        self.outlier_mask = flag_outlier_frames(self.rigid_coords, part_names, reference_edge = (0, 1), window = 5,thresh_deg = 90.0)
        self.raw_len = self.T - self.T_in - self.T_out + 1
        self.valid_indices = []
        
        for i in range(self.raw_len):
            j = i + self.T_in
            k = j + self.T_out
            window_outliers = self.outlier_mask[i:k]
            if not torch.any(window_outliers):
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)
    
    def set_current_pred(self, idx):
        if idx > self.pred_dists.shape[0]:
            idx = self.pred_dists.shape[0] - 1
        else:
            self.pred_dataset = idx
        
    def get_normalizers(self):
        return self.normalizer
    
    def get_graph(self):
        return {
            # Static dual-skeleton graph
            "edge_index":  self.edge_index,               # (2, E) dual skeleton adjacency
            "node_ids":    self.active_rigid_ids,
            "node_type":   self.node_type,                # (2K,) 0 = rigid / 1 = deformable
            "group_id":    self.group_id,                 # (2K,) GGMotion group index
            "adj_matrix":  self.adj,
            "hop_dist":    self.hop_dist,
            "N":           self.N,
            "N_active":    self.N_active,
            "bond_std":    self.bond_std,
            "bond_mean":   self.bond_mean,
            "ref_bone_lengths":  self.ref_bone_lengths,
        }
    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")
        raw_idx = self.valid_indices[idx]
        i = raw_idx
        j = raw_idx + self.T_in
        k = j + self.T_out

        return {
            # Per-node temporal features
            "node_positions":   self.all_c[i:j],               # (Tin, 2K, 3) COM-centred positions of all nodes
            "node_vel":         self.node_vel[i:j],            # (Tin, 2K, 3) frame-to-frame velocity in COM-relative frame
            "node_speed":       self.node_speed[i:j],          # (Tin, 2K, 1) ||vel||  rotation-invariant scalar
            "node_accel":       self.node_accel[i:j],          # (Tin, 2K, 3) frame-to-frame acceleration (delta vel)
            "node_accel_mag":   self.node_accel_mag[i:j],      # (Tin, 2K, 1) ||accel||  rotation-invariant scalar
            # Global body motion
            "global_vel":       self.global_vel[i:j],          # (Tin, 3) COM velocity  (locomotion carrier)
            "global_speed":     self.global_speed[i:j],        # (Tin, 1) ||COM vel||
            "global_accel":     self.global_accel[i:j],        # (Tin, 3) COM acceleration
            "com_history":      self.com[i:j],                 # (Tin, 3) raw COM per frame  (for world-space reconstruction)
            # Deformation residual
            "hist_dist":           self.relative_dist[i:j],                          # (Tin, K, 3) history of deformable - rigid  (translation invariant deformation residual)
            "input_dist":          self.pred_dists[self.pred_dataset, j:k],          # (Tin, K, 3) Noisy prediction of deformable - rigid  (translation invariantdeformation residual)
            "target_dist":         self.relative_dist[j:k],                          # (Tin, K, 3) Ground truth of deformable - rigid  (translation invariantdeformation residual)
            "hist_aligned_dist":   self.relative_aligned_dist_norm[i:j],                  # (Tin, K, 3) history of deformable - rigid  (transrotational invariant deformation residual)
            "input_aligned_dist":  self.pred_aligned_dists_norm[self.pred_dataset, j:k],  # (Tin, K, 3) Noisy prediction of deformable - rigid  (transrotational invariant deformation residual)
            "target_aligned_dist": self.relative_aligned_dist_norm[j:k],                  # (Tin, K, 3) Ground truth of deformable - rigid  (transrotational invariant deformation residual)
            # World space targets
            "target_positions": self.deform_active[j:k],       # (Tout, K, 3) future deformable coords to predict
            "target_rigid":     self.rigid_coords[j:k],        # (Tout, K, 3) future rigid coords (available at inference)
        }
        

def finite_diff(x):
    out = torch.zeros_like(x)
    out[1:] = x[1:] - x[:-1]
    return out


def build_dual_skeleton_graph(K, K_active, active_rigid_ids, rigid_edges, part_groups, mean_pos_compact):
    """
    Build the pruned dual-layer skeleton graph.
    """
    N = K + K_active
    # Build a reverse lookup: rigid index → compact deformable index (or -1)
    rigid_to_deform = torch.full((N,), -1, dtype=torch.long)
    for deform_i, rigid_k in enumerate(active_rigid_ids.tolist()):
        rigid_to_deform[rigid_k] = deform_i

    edges_list, lengths = [], []

    # Rigid bone edges (over indices 0..K-1, unchanged)
    for p, c in rigid_edges:
        edges_list.append((p, c))
        lengths.append(torch.norm(mean_pos_compact[c] - mean_pos_compact[p]).item())

    # Deform bond edges: rigid_k → compact deformable node (K + deform_i)
    for deform_i, rigid_k in enumerate(active_rigid_ids.tolist()):
        deform_node = K + deform_i
        edges_list.append((rigid_k, deform_node))
        lengths.append(torch.norm(mean_pos_compact[deform_node] - mean_pos_compact[rigid_k]).item())

    src_list, dst_list = zip(*edges_list)
    edge_index = torch.tensor([list(src_list), list(dst_list)], dtype=torch.long)

    # Hop distances via BFS on undirected graph
    adj = {i: [] for i in range(N)}
    for s, d in edges_list:
        adj[s].append(d)
        adj[d].append(s)

    lengths = torch.tensor(lengths, dtype=torch.float32)

    # Node type: 0 = rigid, 1 = deformable
    node_type = torch.zeros(N, dtype=torch.long)
    node_type[K:] = 1

    # Group ID: rigid nodes get their group as before;
    # active deformable nodes inherit the same group as their rigid parent
    group_names = list(part_groups.keys())
    name_to_id  = {name: i for i, name in enumerate(group_names)}
    group_id    = torch.zeros(N, dtype=torch.long)
    for name, part_indices in part_groups.items():
        gid = name_to_id[name]
        for p in part_indices:
            group_id[p] = gid                          # rigid node
            deform_i = rigid_to_deform[p].item()
            if deform_i >= 0:                          # only if active
                group_id[K + deform_i] = gid

    return edge_index, lengths, node_type, group_id

def detect_fixed_nodes(rigid_coords, deformable_coords, eps_mean=-1, eps_std=-1):
    """
    Identify rigid keypoints whose deformable counterpart is statistically
    identical (zero-residual), meaning the deformable twin is redundant.

    -------
    fixed_mask  : BoolTensor (K,)   True  = redundant deformable node
    active_mask : BoolTensor (K,)   True  = meaningful deformable node
    """
    residual  = deformable_coords - rigid_coords          # (T, K, 3)
    mean_norm = residual.mean(dim=0).norm(dim=-1)         # (K,)
    std_norm  = residual.std(dim=0).norm(dim=-1)          # (K,)
    fixed_mask  = (mean_norm < eps_mean) & (std_norm < eps_std)
    active_mask = ~fixed_mask
    return fixed_mask, active_mask

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


def build_adjacency_matrix(edge_index, N, group_ids):
    """
    Build a dense undirected binary adjacency matrix and a hop distance matrix.
    """
    # 1. Build Adjacency Matrix
    adj = torch.zeros(N, N, dtype=torch.uint8)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0 
    
    # 2. Initialize Hop Distance Matrix
    hop_dist = torch.full((N, N), float('inf'), dtype=torch.float32)
    hop_dist.fill_diagonal_(0)
    hop_dist[adj > 0] = 1.0
    
    # 3. Vectorized Floyd-Warshall Algorithm
    for k in range(N):
        dist_through_k = hop_dist[:, k].unsqueeze(1) + hop_dist[k, :].unsqueeze(0)
        hop_dist = torch.minimum(hop_dist, dist_through_k)
    
    group_centric_dists = torch.zeros(N, dtype=torch.uint8)
    groups = list(DEFAULT_PART_GROUPS.values())
    for n_id, g_id in enumerate(group_ids):
        parent = groups[g_id][0]
        group_centric_dists[n_id] = hop_dist[n_id, parent]
        
    return adj, group_centric_dists


class DeformationNormalizer:
    """
    Standardizes 3D coordinates/distances to have zero mean and unit variance.
    Supports broadcasting for different temporal/batch dimensions.
    """
    def __init__(self, data: torch.Tensor, min_std: float = 0.01):
        self.mean = data.mean()
        raw_std = data.std()
        self.std = torch.clamp(raw_std, min=min_std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mean

    def to(self, device):
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean)
            self.std = torch.tensor(self.std)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self