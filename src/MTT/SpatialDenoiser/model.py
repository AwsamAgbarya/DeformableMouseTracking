import torch 
import torch.nn as nn
import sys
sys.path.append("./src")
from MTT.modules import HistoryEncoder

class MLPSpatialCorrector(nn.Module):
    """
    Simple MLP-based spatial corrector for deformation refinement.
    Processes each joint independently (no spatial communication).

    Input shape: (B, N, 3) for noisy deformations
                 (B, N, H) for history embeddings
                 (B, N) for joint IDs (will be embedded)

    Output shape: (B, N, 3) for correction vectors
    """

    def __init__(self, history_dim = 64, hidden_dim = 128, num_layers = 3, dropout = 0.1):
        super().__init__()
        self.history_dim = history_dim
        # Per-joint MLP
        input_dim = 3 + history_dim

        layers = []
        prev_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, d_noisy, history_emb):
        """
        Args:
            d_noisy: Noisy deformation offsets (B, N, 3)
            history_emb: History embeddings (B, N, history_dim)
            joint_ids: Joint type IDs (B, N)

        Returns:
            Correction vectors (B, N, 3)
        """
        B, N, _ = d_noisy.shape
        # Concatenate features
        x = torch.cat([d_noisy, history_emb], dim=-1)  # (B, N, input_dim)
        x = x.reshape(B * N, -1) 
        corrections = self.mlp(x)
        corrections = corrections.reshape(B, N, 3)

        return corrections


class DeformationCorrector(nn.Module):
    """
    Full deformation corrector pipeline:
    1. Encode history of deformations
    2. Use spatial corrector to refine noisy predictions
    """
    def __init__(self, history_window = 5, **kawrgs):
        """
        Args:
            n_joints: Number of joints/limbs
            history_window: Number of past frames to use
            history_dim: Dimension of history embedding
            corrector_type: "mlp" or "transformer"
            **corrector_kwargs: Additional arguments for the spatial corrector
        """
        super().__init__()
        self.history_window = history_window
        corrector_kwargs = kawrgs['spatial_corrector']
        encoder_kwargs = kawrgs['encoder']
        # History encoder
        self.history_encoder = HistoryEncoder(**encoder_kwargs)

        # Spatial corrector
        self.spatial_corrector = MLPSpatialCorrector(history_dim=encoder_kwargs['history_dim'], **corrector_kwargs)

    def forward(self, d_noisy, d_history):
        """
        Args:
            d_noisy: Noisy deformation predictions at t+1 (B, N, 3)
            d_history: Deformation history from t-T+1 to t (B, N, T, 3)
            joint_ids: Joint type IDs (B, N)

        Returns:
            Corrected deformations (B, N, 3)
        """
        history_emb = self.history_encoder(d_history.transpose(1, 2))
        delta_d = self.spatial_corrector(d_noisy=d_noisy, history_emb=history_emb)
        d_corrected = d_noisy + delta_d
        return d_corrected
