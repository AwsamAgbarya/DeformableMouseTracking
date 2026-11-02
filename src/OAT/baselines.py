"""
TemporalCNN (baseline L2)
Config (model section of the experiment YAML):

    model:
      type: tcn
      tcn:
        channels: 832            # 832 matches the M0 parameter count at H = 32
        kernel_size: 3
        filter_widths: null      # null: smallest stack of kernel_size covering 2H+1 slots
        dropout: 0.25
        norm: batch              # batch (VideoPose3D) | layer | none
        bn_momentum: 0.1
        offset_embedding: true
        predict_uncertainty: false
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from OAT.modules import RelativeTemporalEncoding, DilatedResidualBlock

def _get(conf, key, default):
    """conf.get for dicts and DictConfigs, with None meaning 'use the default'."""
    if conf is None:
        return default
    value = conf.get(key, default)
    return default if value is None else value


def auto_filter_widths(window_size, kernel_size=3):
    """Smallest stack of equal kernels whose receptive field covers the window."""
    k = int(kernel_size)
    assert k >= 3 and k % 2 == 1, f"kernel_size must be odd and >= 3, got {k}"
    n = 1
    while k ** n < int(window_size):
        n += 1
    return [k] * n

class TemporalCNN(nn.Module):
    """
    Window in, centre-frame pose out; a drop-in replacement for TemporalModel.

    Args:
        conf          the `model` config; architecture settings under conf["tcn"]
        num_keypoints N
        window_size   2 * T_half + 1, the window the model is built for
        max_offset    largest |offset| the offset embedding distinguishes
                      (train_dataset.max_radius, as for the transformer)
    """

    def __init__(self, conf, num_keypoints, window_size, max_offset):
        super().__init__()
        c = _get(conf, "tcn", {})
        self.num_keypoints = int(num_keypoints)
        self.J = self.num_keypoints
        self.window_size = int(window_size)
        self.W = self.window_size
        self.center_idx = self.window_size // 2
        self.max_offset = int(max_offset)

        self.channels = int(_get(c, "channels", 832))
        widths = _get(c, "filter_widths", None)
        if widths is None:
            widths = auto_filter_widths(self.window_size, int(_get(c, "kernel_size", 3)))
        self.filter_widths = [int(w) for w in widths]
        assert len(self.filter_widths) >= 1, "need at least one block"
        assert all(w >= 1 and w % 2 == 1 for w in self.filter_widths), \
            f"filter_widths must be odd, got {self.filter_widths}"
        self.dilations = [math.prod(self.filter_widths[:i]) for i in range(len(self.filter_widths))]
        self.receptive_field = math.prod(self.filter_widths)
        assert self.receptive_field >= self.window_size, (
            f"receptive field {self.receptive_field} < window {self.window_size}: the centre "
            f"could not see the outer slots. Use more or wider filter_widths."
        )

        self.dropout = float(_get(c, "dropout", 0.25))
        self.norm = str(_get(c, "norm", "batch")).lower()
        bn_momentum = float(_get(c, "bn_momentum", 0.1))
        self.use_offset_embedding = bool(_get(c, "offset_embedding", True))
        self.predict_uncertainty = bool(_get(c, "predict_uncertainty", False))

        N, C = self.num_keypoints, self.channels
        self.in_features = 4 * N + 1                  # coords (3N) + visibility (N) + evidence (1)
        self.slot_proj = nn.Linear(self.in_features, C)
        self.offset_embed = RelativeTemporalEncoding(C, max_offset=self.max_offset) \
            if self.use_offset_embedding else None
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(C, w, d, dropout=self.dropout, norm=self.norm, bn_momentum=bn_momentum)
            for w, d in zip(self.filter_widths, self.dilations)
        ])
        self.head = nn.Linear(C, 3 * N)
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Linear(C, 3 * N)

    def extra_repr(self):
        return (f"window={self.window_size}, receptive_field={self.receptive_field}, "
                f"filter_widths={self.filter_widths}, dilations={self.dilations}, channels={self.channels}")

    def _slot_features(self, keypoints_window, vis, evidence):
        """(B, T, N, 3), (B, T, N), (B, T) -> (B, T, 4N + 1). Hidden joints are zeros."""
        B, T, N, _ = keypoints_window.shape
        vis = vis & evidence.unsqueeze(-1)
        coords = keypoints_window * vis.unsqueeze(-1).to(keypoints_window.dtype)
        return torch.cat([
            coords.reshape(B, T, 3 * N),
            vis.to(coords.dtype),
            evidence.to(coords.dtype).unsqueeze(-1),
        ], dim=-1)

    def forward(self, keypoints_window, occlusion_window, frame_offsets=None, pad_mask=None):
        """
        Args (straight from a TemporalWrapper batch, as for TemporalModel):
            keypoints_window (B, T, N, 3) window-frame coordinates
            occlusion_window (B, T, N) or (B, T, N, 1) bool, True = visible
            frame_offsets    (B, T) signed frame offsets from the centre
            pad_mask         (B, T) bool, True = slot holds no evidence
        """
        B, T, N, _ = keypoints_window.shape
        device = keypoints_window.device
        center = T // 2

        vis = occlusion_window.bool()
        if vis.dim() == 4:
            vis = vis[..., 0]
        if frame_offsets is None:
            frame_offsets = (torch.arange(T, device=device) - center).unsqueeze(0).expand(B, T)
        else:
            frame_offsets = frame_offsets.to(device)
        evidence = vis.any(-1)
        if pad_mask is not None:
            evidence = evidence & ~pad_mask.to(device).bool()

        # no hidden coordinate reaches the network.
        keypoints_window = keypoints_window * vis.unsqueeze(-1).to(keypoints_window.dtype)

        if T > self.receptive_field or (self.receptive_field - T) % 2:
            raise ValueError(f"window of {T} slots does not fit receptive field {self.receptive_field} "
                             f"symmetrically (needs an odd T <= receptive field)")
        pad = (self.receptive_field - T) // 2

        feats = self._slot_features(keypoints_window, vis, evidence)           # (B, T, 4N+1)
        feats = F.pad(feats, (0, 0, pad, pad))                                 # empty slots
        offsets = F.pad(frame_offsets.float(), (pad, pad))
        ev = F.pad(evidence.float(), (pad, pad))                               # (B, R)

        h = self.slot_proj(feats)                                             # (B, R, C)
        if self.offset_embed is not None:
            h = h + self.offset_embed(offsets.round().long()) * ev.unsqueeze(-1)
        h = h.transpose(1, 2)                                                  # (B, C, R)
        for block in self.blocks:
            h = block(h)
        assert h.shape[-1] == 1, f"expected one output position, got {h.shape[-1]}"
        h = h[..., 0]                                                          # (B, C)

        coords_dense = self.head(h).view(B, N, 3)
        occluded = ~vis[:, center]
        completed = torch.where(occluded.unsqueeze(-1), coords_dense, keypoints_window[:, center])

        out = {
            "coordinates": completed, 
            "coords_dense": coords_dense, 
            "occluded_mask": occluded,
        }
        if self.predict_uncertainty:
            out["log_sigma"] = self.uncertainty_head(h).view(B, N, 3)
        return out