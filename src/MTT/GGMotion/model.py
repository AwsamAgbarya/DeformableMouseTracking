import torch
import math
import torch.nn as nn
from MTT.modules import Embedding, SpatioTemporalRadialField

class GGMotionBlock(nn.Module):
    """
    Complete GGMotion block combining all modules
    
    Args:
        D: Coordinate dimension (3)
        C: Hidden feature channels
        N: Number of joints
        groups: Body group definitions
        adjacency: Skeleton adjacency matrix
        parent_indices: Parent joint indices
    """
    def __init__(self, dij, adjacency, D=3, C=64, N=22, groups=None, parent_indices=None):
        super().__init__()
        self.D = D
        self.C = C
        self.N = N
        self.dij = dij
          
        # Spatio-temporal radial field
        self.radial_field = SpatioTemporalRadialField(dij, adjacency, D, C, N)
        
        # # Inter-group interaction
        # self.inter_group = InterGroupInteraction(D, C, groups)
        
        # # Intra-group interaction
        # self.intra_group = IntraGroupInteraction(D, C, groups)
        
        # # Dynamics-kinematics
        # self.dynamics_kinematics = DynamicsKinematics(D, C, N, parent_indices)
        
        # Centroid update
        # self.centroid_update = nn.Linear(D, D, bias=False)
    
    def forward(self, X, V):
        """
        Args:
            X: Position features [B, N, D, C]
            V: Velocity features [B, N, D, C]
        
        Returns:
            X_next: Updated positions [B, N, D, C]
            V_next: Updated velocities [B, N, D, C]
        """

        f = self.radial_field(X, V)
        # f = self.inter_group(f)
        # f = self.intra_group(f)
        # X_next, V_next = self.dynamics_kinematics(f, X, V)
        
        # # Centroid update
        # X_centroid = X_next.mean(dim=(1, 3))
        # X_centroid_updated = self.centroid_update(X_centroid)
        # X_centroid_updated = X_centroid_updated.unsqueeze(1).unsqueeze(-1)  # [B, 1, D, 1]
        
        # # Re-center positions
        # X_next = X_next - X_next.mean(dim=(1, 3), keepdim=True) + X_centroid_updated
        
        return X, V, f


class GGMotion(nn.Module):
    """
    Full GGMotion network with multiple blocks
    
    Args:
        D: Coordinate dimension
        C: Hidden channels
        N: Number of joints
        Th: Historical timesteps
        Tf: Future timesteps to predict
        L: Number of blocks
        groups: Body group definitions
        adjacency: Skeleton adjacency
        parent_indices: Parent joints
    """
    def __init__(self, hop_matrix, adjacency, D=3, C=64, N=17, Th=3, L=1,
                 groups=None, parent_indices=None):
        super().__init__()
        self.D = D
        self.C = C
        self.N = N
        self.Th = Th
        self.L = L
        # Precompute the sinusoidal positional embedding
        dij = get_sinusoidal_embeddings(hop_matrix, C)
        
        self.embedding = Embedding(D, C, Th)
        self.blocks = nn.ModuleList([
            GGMotionBlock(dij, adjacency, D, C, N, groups, parent_indices)
            for _ in range(L)
        ])
        self.output_proj = nn.Linear(D * C, D, bias=False)
    
    def forward(self, X_hist):
        """
        Args:
            X_hist: Historical motion [B, Th, N, D]
        
        Returns:
            Y_pred: Predicted future motion [B, Tf, N, D]
        """
        B, Th, N, D = X_hist.shape
        
        # Embed
        X, V = self.embedding(X_hist)  # [B, C, N, D]
        # Pass through blocks
        for block in self.blocks:
            X, V, f = block(X, V)
        
        # # Output projection
        # X_flat = X.reshape(B, N, D * self.C)  # [B, N, D*C]
        # Y_pred = self.output_proj(X_flat)  # [B, N, D]
        # Y_pred = Y_pred.reshape(B, N, D)  # [B, N, D]
        
        return X, V, f

def get_sinusoidal_embeddings(positions, dim):
    half_dim = dim // 2
    div_term = torch.exp(torch.arange(half_dim, device=positions.device) * (-math.log(10000.0) / half_dim))
    scaled_pos = positions.view(-1, 1).float() * div_term.view(1, -1)
    return torch.cat([torch.sin(scaled_pos), torch.cos(scaled_pos)], dim=-1)