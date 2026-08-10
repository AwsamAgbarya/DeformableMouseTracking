import torch 
import torch.nn as nn
import sys
sys.path.append("./src")
from MTT.modules import HistoryEncoder

class SpatialTransformerCorrector(nn.Module):
    """
    Spatial Transformer-based corrector for deformation refinement.

    Input shape: (B, N, 3) for noisy deformations
                 (B, N, H) for history embeddings
                 (B, N) for joint IDs

    Output shape: (B, N, 3) for correction vectors
    """

    def __init__(self, n_joints, history_dim = 64, joint_embedding_dim = 16, transformer_dim = 128, num_heads = 4, num_layers = 2, ff_dim = 256, dropout = 0.1):
        super().__init__()
        self.n_joints = n_joints
        self.history_dim = history_dim
        self.transformer_dim = transformer_dim

        # Joint ID embedding + spatial positional encoding
        self.joint_embedding = nn.Embedding(n_joints, joint_embedding_dim)
        self.spatial_position_embedding = nn.Embedding(n_joints, transformer_dim)

        # Project input features to transformer dimension
        input_dim = 3 + history_dim + joint_embedding_dim
        self.input_projection = nn.Linear(input_dim, transformer_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: transformer_dim -> 3
        self.output_projection = nn.Linear(transformer_dim, 3)

    def forward(self, d_noisy, history_emb, joint_ids):
        """
        Args:
            d_noisy: Noisy deformation offsets (B, N, 3)
            history_emb: History embeddings (B, N, history_dim)
            joint_ids: Joint type IDs (B, N)

        Returns:
            Correction vectors (B, N, 3)
        """
        B, N, _ = d_noisy.shape

        joint_emb = self.joint_embedding(joint_ids)
        x = torch.cat([d_noisy, history_emb, joint_emb], dim=-1)
        x = self.input_projection(x)
        spatial_pos = self.spatial_position_embedding(joint_ids)
        x = x + spatial_pos

        # Apply transformer encoder with self-attention
        x = self.transformer_encoder(x)  # (B, N, transformer_dim)
        corrections = self.output_projection(x)  # (B, N, 3)

        return corrections


class DeformationCorrector(nn.Module):
    """
    Full deformation corrector pipeline:
    1. Encode history of deformations
    2. Use spatial corrector to refine noisy predictions
    """
    def __init__(self, n_joints, history_window = 5, history_dim = 64, **corrector_kwargs):
        """
        Args:
            n_joints: Number of joints/limbs
            history_window: Number of past frames to use
            history_dim: Dimension of history embedding
            corrector_type: "mlp" or "transformer"
            **corrector_kwargs: Additional arguments for the spatial corrector
        """
        super().__init__()
        self.n_joints = n_joints
        self.history_window = history_window
        self.history_dim = history_dim

        # History encoder
        self.history_encoder = HistoryEncoder(history_dim=history_dim, num_layers=1, dropout=0.1, use_gru=True)
        self.spatial_corrector = SpatialTransformerCorrector(n_joints=n_joints, history_dim=history_dim, **corrector_kwargs)


    def forward(self, d_noisy, d_history, joint_ids):
        """
        Args:
            d_noisy: Noisy deformation predictions at t+1 (B, N, 3)
            d_history: Deformation history from t-T+1 to t (B, N, T, 3)
            joint_ids: Joint type IDs (B, N)

        Returns:
            Corrected deformations (B, N, 3)
        """
        history_emb = self.history_encoder(d_history)
        delta_d = self.spatial_corrector(d_noisy=d_noisy, history_emb=history_emb, joint_ids=joint_ids)
        d_corrected = d_noisy + delta_d
        return d_corrected
