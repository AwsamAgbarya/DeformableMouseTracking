import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    B: Batch size
    N: Joints
    D: Coordinate dimension (default: 3)
    C: Hidden feature channels
    Th: Historical timesteps
"""
class Embedding(nn.Module):
    """
    Embeds motion sequences into equivariant position and velocity features,
    """
    def __init__(self, D=3, C=64, Th=10):
        super().__init__()
        self.D = D
        self.C = C
        self.Th = Th
        
        self.pos_embed = nn.Linear(Th, C, bias=False)
        self.vel_embed = nn.Linear(Th, C, bias=False)

    def forward(self, X):
        """
        Args:
            X: Input motion [B, Th, N, D]
        Returns:
            X0: Position features [B, N, D, C]
            V0: Velocity features [B, N, D, C]
        """
        B, Th, N, D = X.shape
        
        # SCenter the positions in both time and space
        X_mean = X.mean(dim=[1, 2], keepdim=True)  # [B, 1, 1, D]
        X_centered = X - X_mean  # [B, Th, N, D]
        
        # Compute velocity (temporal difference)
        velocity = X_centered[:, 1:, :, :] - X_centered[:, :-1, :, :]  # [B, Th-1, N, D]
        velocity = torch.cat([velocity, velocity[:, -1:, :, :]], dim=1)  # [B, Th, N, D]
        
        # Permute so Th is the last dimension for the Linear layer
        X_centered_perm = X_centered.permute(0, 2, 3, 1)  # [B, N, D, Th]
        vel_perm = velocity.permute(0, 2, 3, 1)           # [B, N, D, Th]
        
        # Project temporal sequence to channels
        X0 = self.pos_embed(X_centered_perm)  # [B, N, D, C]
        V0 = self.vel_embed(vel_perm)         # [B, N, D, C]
        
        # Broadcast X_mean correctly to add back the centroid
        X_mean_broadcast = X_mean.view(B, 1, D, 1)  # Reshape to broadcast over N and C
        X0 = X0 + X_mean_broadcast  # [B, N, D, C]
        
        return X0, V0

class EquivariantMLP(nn.Module):
    """
    Equivariant MLP with covariance-based attention mechanism
    
    Args:
        D: Coordinate dimension
        C: Feature channels
        n_vars: Number of variables to process (e.g., n joints or 3 for [f,r,v])
    """
    def __init__(self, D=3, C=64, n_vars=3):
        super().__init__()
        self.D = D
        self.C = C
        self.n_vars = n_vars
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(D * C, D * C, bias=False)
        self.W_k = nn.Linear(D * C, D * C, bias=False)
        self.W_v = nn.Linear(D * C, D * C, bias=False)
        
        # MLP for processing covariance matrix
        self.cov_mlp = nn.Sequential(
            nn.Linear(n_vars, n_vars),
            nn.ReLU(),
            nn.Linear(n_vars, n_vars)
        )
        
        # Output projection
        self.W_out = nn.Linear(D * C, D * C, bias=False)
    
    def forward(self, Z):
        """
        Args:
            Z: Input features [B, N, n_vars, D, C] or [B, N, D, C]
               If 4D: assumes single variable (n_vars=1)
        
        Returns:
            Z_out: Output features, same shape as input
        """
        if Z.dim() == 4:
            # Add variable dimension
            Z = Z.unsqueeze(2)  # [B, N, 1, D, C]
        
        B, N, n_vars, D, C = Z.shape
        assert n_vars == self.n_vars, f"Expected {self.n_vars} variables, got {n_vars}"
        
        # Reshape to [B, N, n_vars, D*C]
        Z_flat = Z.reshape(B, N, n_vars, D * C)
        
        # Compute Q, K, V
        Z_q = self.W_q(Z_flat).reshape(B, N, n_vars, D, C)  # [B, N, n_vars, D, C]
        Z_k = self.W_k(Z_flat).reshape(B, N, n_vars, D, C)
        Z_v = self.W_v(Z_flat).reshape(B, N, n_vars, D, C)
        
        # Compute covariance matrix: Σ = Z_q^T @ Z_k
        # For each joint, compute covariance over variables
        cov = torch.einsum('bnvdc,bnwdc->bnvw', Z_q, Z_k)  # [B, N, n_vars, n_vars]
        cov = cov / (D * C)  # Normalize
        
        # Apply MLP to covariance matrix
        cov_transformed = self.cov_mlp(cov)  # [B, N, n_vars, n_vars]
        
        # L2 normalize along row dimension
        cov_norm = F.normalize(cov_transformed, p=2, dim=-1)  # [B, N, n_vars, n_vars]
        
        # Attention-weighted combination: Z_out = Z_v @ cov_norm
        Z_out = torch.einsum('bnwdc,bnvw->bnvdc', Z_v, cov_norm)  # [B, N, n_vars, D, C]
        
        # Output projection
        Z_out_flat = Z_out.reshape(B, N, n_vars, D * C)
        Z_out_proj = self.W_out(Z_out_flat).reshape(B, N, n_vars, D, C)
        
        # If input was 4D, return 4D
        if n_vars == 1:
            Z_out_proj = Z_out_proj.squeeze(2)
        
        return Z_out_proj

class SpatioTemporalRadialField(nn.Module):
    """
    Spatio-temporal radial field for aggregating motion forces
    """
    def __init__(self, dij, adj, D=3, C=64, N=17):
        super().__init__()
        self.D = D
        self.C = C
        self.N = N
        self.dij = dij
        self.adj = adj
        
        # Spatial edge weight MLP
        self.spatial_edge_mlp = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Linear(C, D * C))
        
        # Temporal edge weight MLP
        self.temporal_edge_mlp = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Linear(C, D * C))
        
        # Edge attribute attention MLP (for hop distance)
        self.edge_attr_mlp = nn.Sequential(nn.Linear(C, C), nn.Sigmoid())
        
        # Linear layers for position differences
        self.spatial_pos_linear = nn.Linear(D, D, bias=False)
        self.temporal_pos_linear = nn.Linear(D, D, bias=False)
        
        # Learnable scaling factors (per joint, per dimension)
        self.beta_scale = nn.Parameter(torch.ones(1, N, 1, D, 1))  # Spatial scale
        self.gamma_scale = nn.Parameter(torch.ones(1, N, D, 1))      # Temporal scale
    
    def spatial(self, X, V):
        # === SPATIAL FIELD ===
        B, N, D, C = X.shape
    
        # Comput e_ij = beta_i * mlp_e (|Xi - Xj|2,col)
        # This represents the strength of physical connections between neighbors
        pos_diff = X.unsqueeze(2) - X.unsqueeze(1)  # [B, N, N, D, C]
        phi_e_out = self.spatial_edge_mlp(torch.norm(pos_diff, p=2, dim=3)).view(B, N, N, D, C) # [B, N, N, D, C]
        e_ij = self.beta_scale * phi_e_out
        # Apply adjacency mask
        adj_mask = (self.adj == 1).float().view(1, N, N, 1, 1).to(X.device)
        e_ij = e_ij * adj_mask

        # Compute e_tilde_i = Sum (mlp_att(d_{n_s1, i}))
        e_tilde_i = self.edge_attr_mlp(self.dij)  # [N, C]
        e_tilde_i = e_tilde_i.view(1, N, 1, C) 

        # compute f_i = V_i + e^_i * Sum_neighbors ( e_ij * mlp_lin(X_i, X_j))
        pos_diff_transposed = pos_diff.transpose(-1, -2)              # [B, N, N, C, D]
        phi_lin_out = self.spatial_pos_linear(pos_diff_transposed)    # [B, N, N, C, D]
        phi_lin_out = phi_lin_out.transpose(-1, -2)                   # [B, N, N, D, C]
        sum_neighbors = (e_ij * phi_lin_out).sum(dim=2)  # [B, N, D, C]
        f_spatial = V + (e_tilde_i * sum_neighbors) 

        return f_spatial
    
    def temporal(self, X, V):
        # === TEMPORAL FIELD ===
        B, N, D, C = X.shape
        
        # COmpute mi = gamma_i * mlp_m (|X_i - X_c|2, col)
        pos_diff_temporal = X - X.mean(dim=1, keepdim=True)  # [B, N, D, C]
        phi_m_out = self.temporal_edge_mlp(torch.norm(pos_diff_temporal, p=2, dim=2)) # [B, N, D*C]
        phi_m_out = phi_m_out.view(B, N, D, C) # [B, N, D, C]
        m_i = self.gamma_scale * phi_m_out
        
        # compute f_i = V_i + m_i * mlp_lin(X_i - X_c)
        # 1. Apply phi_lin to the position difference (apply to D dimension)
        pos_diff_transposed = pos_diff_temporal.transpose(-1, -2)     # [B, N, C, D]
        phi_lin_out = self.temporal_pos_linear(pos_diff_transposed)   # [B, N, C, D]
        phi_lin_out = phi_lin_out.transpose(-1, -2)                   # [B, N, D, C]
        f_temporal = V + (m_i * phi_lin_out)                          # [B, N, D, C]
        
        return f_temporal
    
    def forward(self, X, V):
        """
        Args:
            X: Position features [B, N, D, C]
            V: Velocity features [B, N, D, C]
        
        Returns:
            f: Aggregated motion forces [B, N, D, C]
        """
        
        f_spatial = self.spatial(X, V)
        f_temporal = self.temporal(X, V)
        # === COMBINE ===
        f = V + f_spatial + f_temporal  # [B, N, D, C]
        
        return f

class InterGroupInteraction(nn.Module):
    """
    Inter-group interaction module for capturing cross-body-part dependencies
    
    Args:
        D: Coordinate dimension
        C: Feature channels
        groups: List of lists, each containing joint indices for a group
        eq_mlp: Equivariant MLP module
    """
    def __init__(self, D=3, C=64, groups=None, eq_mlp=None):
        super().__init__()
        self.D = D
        self.C = C
        
        # Default Human3.6M grouping (6 groups)
        if groups is None:
            # Example grouping (adjust based on your skeleton)
            groups = [
                [0, 1, 2, 3],      # Spine
                [4, 5, 6],         # Left arm
                [7, 8, 9],         # Right arm
                [10, 11, 12],      # Left leg
                [13, 14, 15],      # Right leg
                [16, 17]           # Head
            ]
        self.groups = groups
        self.S = len(groups)
        
        # Equivariant MLP (will be provided)
        if eq_mlp is None:
            from modules import EquivariantMLP  # Placeholder
            eq_mlp = EquivariantMLP(D, C, n_vars=self.S)
        self.eq_mlp = eq_mlp
        
    def forward(self, f):
        """
        Args:
            f: Motion forces [B, N, D, C]
        
        Returns:
            f_out: Updated forces [B, N, D, C]
        """
        B, N, D, C = f.shape
        
        # Compute resultant force for each group
        group_forces = []
        for group_indices in self.groups:
            # Sum forces within each group
            group_force = f[:, group_indices, :, :].sum(dim=1)  # [B, D, C]
            group_forces.append(group_force)
        
        # Stack group forces [B, S, D, C]
        group_forces = torch.stack(group_forces, dim=1)  # [B, S, D, C]
        
        # Apply equivariant MLP to all group forces
        group_updates = self.eq_mlp(group_forces)  # [B, S, D, C]
        
        # Distribute updates back to individual joints with residual connection
        f_out = f.clone()
        for s, group_indices in enumerate(self.groups):
            for idx in group_indices:
                f_out[:, idx, :, :] = f_out[:, idx, :, :] + group_updates[:, s, :, :]
        
        return f_out
    
class IntraGroupInteraction(nn.Module):
    """
    Intra-group interaction module for within-group joint dependencies
    
    Args:
        D: Coordinate dimension
        C: Feature channels
        groups: List of lists, each containing joint indices for a group
    """
    def __init__(self, D=3, C=64, groups=None):
        super().__init__()
        self.D = D
        self.C = C
        
        # Default grouping
        if groups is None:
            groups = [
                [0, 1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17]
            ]
        self.groups = groups
        self.S = len(groups)
        
        # Separate equivariant MLP for each group
        self.group_mlps = nn.ModuleList()
        for group in groups:
            from modules import EquivariantMLP  # Placeholder
            mlp = EquivariantMLP(D, C, n_vars=len(group))
            self.group_mlps.append(mlp)
    
    def forward(self, f):
        """
        Args:
            f: Motion forces [B, N, D, C]
        
        Returns:
            f_out: Updated forces [B, N, D, C]
        """
        B, N, D, C = f.shape
        f_out = f.clone()
        
        # Process each group independently
        for s, (group_indices, group_mlp) in enumerate(zip(self.groups, self.group_mlps)):
            # Extract forces for this group
            group_f = f[:, group_indices, :, :]  # [B, k, D, C]
            
            # Apply group-specific equivariant MLP
            group_update = group_mlp(group_f)  # [B, k, D, C]
            
            # Add residual connection
            f_out[:, group_indices, :, :] = f_out[:, group_indices, :, :] + group_update
        
        return f_out

class DynamicsKinematics(nn.Module):
    """
    Parallel dynamics-kinematics propagation module
    
    Args:
        D: Coordinate dimension
        C: Feature channels
        N: Number of joints
        parent_indices: Parent joint index for each joint [N]
                       (root joint has parent -1 or self)
    """
    def __init__(self, D=3, C=64, N=22, parent_indices=None):
        super().__init__()
        self.D = D
        self.C = C
        self.N = N
        
        # Default kinematic chain (adjust for your skeleton)
        if parent_indices is None:
            # Example: simple chain, root=0
            parent_indices = torch.tensor([-1] + list(range(N-1)))
        self.register_buffer('parent_indices', parent_indices)
        
        # Equivariant MLP for dynamics (processes f, r, v)
        from modules import EquivariantMLP
        self.dynamics_mlp = EquivariantMLP(D, C, n_vars=3)
        
        # Linear layer for acceleration integration
        self.accel_linear = nn.Linear(D * C, D * C, bias=False)
    
    def forward(self, f, X, V):
        """
        Args:
            f: Motion forces [B, N, D, C]
            X: Position features [B, N, D, C]
            V: Velocity features [B, N, D, C]
        
        Returns:
            X_next: Updated positions [B, N, D, C]
            V_next: Updated velocities [B, N, D, C]
        """
        B, N, D, C = X.shape
        
        # Compute position and velocity differences from parent
        r = torch.zeros_like(X)  # Position difference
        v = torch.zeros_like(V)  # Velocity difference
        
        for j in range(N):
            parent_idx = self.parent_indices[j].item()
            if parent_idx >= 0:  # Has parent
                r[:, j] = X[:, j] - X[:, parent_idx]
                v[:, j] = V[:, j] - V[:, parent_idx]
            else:  # Root joint
                r[:, j] = X[:, j]
                v[:, j] = V[:, j]
        
        # Prepare inputs for equivariant MLP: stack [f, r, v]
        dynamics_input = torch.stack([f, r, v], dim=2)  # [B, N, 3, D, C]
        
        # Compute acceleration via equivariant MLP
        a = self.dynamics_mlp(dynamics_input)  # [B, N, 3, D, C] or [B, N, D, C]
        
        # If output is stacked, take first component (force-based acceleration)
        if a.dim() == 5:
            a = a[:, :, 0, :, :]  # [B, N, D, C]
        
        # Integrate acceleration to velocity
        a_flat = a.reshape(B, N, D * C)
        a_integrated = self.accel_linear(a_flat).reshape(B, N, D, C)
        V_next = V + a_integrated
        
        # Integrate velocity to position
        X_next = X + V_next
        
        return X_next, V_next
    
class HistoryEncoder(nn.Module):
    """
    Encodes temporal history of deformations into a fixed-size embedding.

    Input shape: (B, N, T, 3) - B batches, N joints, T time steps, 3D coords
    Output shape: (B, N, history_dim)
    """

    def __init__(self, history_dim = 64, num_layers = 1, dropout = 0.1, use_gru = True):
        """
        Args:
            history_dim: Output embedding dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
            use_gru: If True, use GRU; otherwise use LSTM
        """
        super().__init__()
        self.history_dim = history_dim
        self.use_gru = use_gru

        if use_gru:
            self.rnn = nn.GRU(input_size=3, hidden_size=history_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.LSTM(input_size=3, hidden_size=history_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, d_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d_history: (B, N, T, 3) - deformation history for N joints over T frames

        Returns:
            history_emb: (B, N, history_dim) - compressed history embeddings
        """
        B, N, T, _ = d_history.shape

        d_flat = d_history.reshape(B * N, T, 3)
        if self.use_gru:
            _, hidden = self.rnn(d_flat)
            history_emb = hidden[-1]
        else:
            _, (hidden, _) = self.rnn(d_flat)
            history_emb = hidden[-1]
        history_emb = history_emb.reshape(B, N, self.history_dim)

        return history_emb