'''
MLP: Default Submodule MLP block used for TransformerBlocks
SelfAttention: Default Submodule self-attention block used for TransformerBlocks
TransformerBlock: Submodule used for MVT encoder and OAT encoder

PoolingAttention: Default Submodule Pooling Attention block used for AttentionPoolingBlock
AttentionPoolingBlock: Submodule used for MVT encoder

GlobalAttentionPoolingBlock: Submodule used for OAT encoder

CrossViewTransformerBlock: Submodule used for MVT decoder

PositionalEncoding: Default positional encoding block used for MVT and OAT encoders

RelativeTemporalEncoding: Default module for frame indexing for OAT temporal decoder
TemporalDecoderBlock: Defauult OAT temporal decoder block
GlobalDecoderBlock: Ablation A3a, one attention over [queries; centre tokens; memory]
'''
import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        """
        Initialization function
        Args:
            in_features (int):
                Input feature dimension.
            hidden_features (int or None, optional):
                Dimension of the hidden layer. If None, defaults to `in_features`.
            out_features (int or None, optional):
                Output feature dimension. If None, defaults to `in_features`.
            drop (float, optional):
                Dropout probability applied after the second linear layer.
                Default: 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
        Returns:
            torch.Tensor: Output tensor of shape (B, N, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None):
        """
        Args:
            input_dim C (int):
                Input embedding dimension.
            num_heads (int, optional):
                Number of attention heads. Default: 8.
            qkv_bias (bool, optional):
                Whether the QKV projection layer includes a bias term. Default: False.
            qk_scale (float or None, optional):
                Optional override for the query-key scaling factor.
                If None, default is 1/sqrt(head_dim). Default: None.
            attn_drop (float, optional):
                Dropout probability applied to attention weights. Default: 0.0.
            proj_drop (float, optional):
                Dropout probability applied after the output projection. Default: 0.0.
            attn_head_dim (int or None, optional):
                Override the dimension per attention head. If None, use embed_dim // num_heads.
                Default: None.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.input_dim = input_dim
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(input_dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, output_attn=False):
        """
        Args:
            x (torch.Tensor):
                Input tensor of shape (B, N, C), where:
                - B is batch size
                - N is sequence or token length
                - C is embedding dimension
            output_attn (bool):
                Whether to output the attention matrix or not, Default: False.
        Returns:
            torch.Tensor:
                Output tensor of shape (B, N, C), same as input dimension.
        """
        B, N, C = x.shape
        # Project input into Q, K, V using a single linear layer.
        qkv = self.qkv(x)
        # Reshape and permute into (B, heads, N, head_dim).
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention.
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:  # (N, N)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            elif attn_mask.dim() == 3:  # (B, N, N)
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
            attn = attn + attn_mask  # Broadcast across heads

        # Apply softmax to get attention weights.
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Multiply weights with values and concatenate all heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)

        if output_attn:
            return (x, attn)
        else:
            return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., attn_head_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            attn_mask (torch.Tensor, optional): Attention mask of shape (B, N, N) or (N, N).
                      Should contain -inf for positions to mask out, 0 for valid positions.
        """
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class PoolingAttention(nn.Module):
    """
    Cross-Attention module for attention pooling.
    Query comes from the learnable seed; Key/Value come from the input set.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Query projection (applied to the seed)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # Key/Value projection (applied to the input keypoints)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv, kv_mask=None):
        """
        Args:
            x_q: Query tensor (Batch, 1, Dim) - derived from seed
            x_kv: Key/Value tensor (Batch, N, Dim) - derived from input keypoints
        """
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape

        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Split kv input into k and v
        kv = self.kv(x_kv).reshape(B, N_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # Standard Attention Calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if kv_mask is not None:
            attn = attn.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionPoolingBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.):
        super().__init__()

        # Learnable Seed Vector (The Query)
        self.seed = nn.Parameter(torch.randn(1, 1, output_dim))
        self.input_proj = nn.Linear(input_dim, output_dim)

        # Normalization Layers
        self.norm_seed = nn.LayerNorm(output_dim)   # Norm for the query (seed)
        self.norm_ctx = nn.LayerNorm(output_dim)    # Norm for the key/value (input points)

        # Cross Attention Module
        self.attn = PoolingAttention(output_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # MLP
        self.norm2 = nn.LayerNorm(output_dim)       # Norm before MLP
        self.mlp = Mlp(in_features=output_dim, hidden_features=int(output_dim * mlp_ratio), drop=drop)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_views, N_points, D_in).
                              B=Batch, C=Views, N=Keypoints, D=Data Dim
        Returns:
            torch.Tensor: Output tensor of shape (B, C_views, D_out).
        """
        B, C_views, N, D_in = x.shape
        x = x.view(B * C_views, N, D_in) # (B*C, N, D_in)
        x = self.input_proj(x)

        seed = self.seed.expand(B * C_views, -1, -1) # (1, 1, D_out) -> (B*C, 1, D_out)

        # Query = Seed, Key/Value = Input x
        attn_out = self.attn(x_q=self.norm_seed(seed), x_kv=self.norm_ctx(x))
        x = seed + attn_out
        x = x + self.mlp(self.norm2(x))

        x = x.squeeze(1).view(B, C_views, -1) # Shape: (B, C, D_out)

        return x


class GlobalAttentionPoolingBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()

        self.seed = nn.Parameter(torch.randn(1, 1, output_dim))
        self.input_proj = nn.Linear(input_dim, output_dim)

        self.norm_seed = nn.LayerNorm(output_dim)
        self.norm_ctx = nn.LayerNorm(output_dim)

        self.attn = PoolingAttention(
            output_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(output_dim)
        self.mlp = Mlp(in_features=output_dim, hidden_features=int(output_dim * mlp_ratio), drop=drop)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.seed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C_in)

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out)
        """
        B, N, _ = x.shape

        x = self.input_proj(x)                          # (B, N, C_out)
        seed = self.seed.expand(B, -1, -1)             # (B, 1, C_out)

        attn_out = self.attn(x_q=self.norm_seed(seed), x_kv=self.norm_ctx(x), kv_mask=mask) # (B, 1, C_out)

        x = seed + attn_out
        x = x + self.mlp(self.norm2(x))                # (B, 1, C_out)

        return x.squeeze(1)                            # (B, C_out)

class CrossViewTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, memory, occlusion_mask=None):
        """
        Args:
            query: (B*C*N, 1, D) - what we want to predict
            memory: (B*C*N, C, D) - all view encodings to attend to
            occlusion_mask: optional attention mask

        Returns:
            Updated query: (B*C*N, 1, D)
        """
        # Cross-attention: each occluded keypoint query attends to all views
        attn_output, attn_weights = self.cross_attention(
            query=query,   # (num_occluded, 1, D)
            key=memory,    # (num_occluded, C, D) - attend to all C views
            value=memory,  # (num_occluded, C, D)
            need_weights=True
        )

        # Residual connection + norm + FF
        query = self.norm1(query + attn_output)
        query = self.norm2(query + self.ffn(query))

        return query

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=100):
        super().__init__()
        self.embed_dim = embed_dim

        # Pre-compute positional encodings
        pe = torch.zeros(max_seq_len, embed_dim)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             (-np.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(pos * div_term)
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, N, D) or (B*C, N, D)
        Returns:
            Positional encodings of same shape as x
        """
        return self.pe[:, :x.size(1), :]

class RelativeTemporalEncoding(nn.Module):
    """Learned embedding for a frame's signed offset from the target frame,
    e.g. -10 ... 0 ... +10 for a 21-frame window."""
    def __init__(self, embed_dim, max_offset=32):
        super().__init__()
        self.max_offset = max_offset
        self.table = nn.Embedding(2 * max_offset + 1, embed_dim)

    def forward(self, offsets: torch.Tensor):
        idx = (offsets.clamp(-self.max_offset, self.max_offset) + self.max_offset).long()
        return self.table(idx)


class TemporalDecoderBlock(nn.Module):
    """
    One alternating-attention decoder block, matching the corrected design:

      1) Self-attention: the N per-keypoint query tokens (representing every
         keypoint of the center frame, occluded or not) self-attend TOGETHER
         WITH the center frame's own contextualized visible tokens. This lets
         occluded queries pull in full-detail, same-frame skeletal context
         directly -- not compressed through a pooled latent.

      2) Cross-attention: the same query tokens then cross-attend into a
         compact TEMPORAL memory containing exactly one latent per OTHER
         frame in the gathered window (tagged with that frame's relative
         time offset). This gives cheap, long-range temporal context without
         ever growing the memory beyond size T-1.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1, temporal=True):
        super().__init__()
        self.temporal = temporal
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        if temporal:
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, queries, center_tokens, center_pad_mask, memory=None, memory_key_padding_mask=None):
        """
        Args:
            queries: (B, N, D) one token per keypoint identity of the center frame
            center_tokens: (B, V, D) center frame's own contextualized visible tokens
            center_pad_mask: (B, V) bool, True = real (visible) token
            memory: (B, T-1, D) per-OTHER-frame latents, already tagged with
                their relative temporal offset. None (or T-1 == 0) skips the
                temporal cross-attention entirely.
            memory_key_padding_mask: (B, T-1) bool, True = valid frame

        Returns:
            queries: (B, N, D) updated query tokens
        """
        B, N, D = queries.shape
        V = center_tokens.size(1)

        # Self-attention: queries + center frame's own visible tokens ---
        combined = torch.cat([queries, center_tokens], dim=1)              # (B, N+V, D)
        # True = position should be IGNORED as a key. Query slots are always
        # valid; only padding among the center frame's visible tokens is masked.
        key_padding_mask = torch.cat([
            torch.zeros(B, N, dtype=torch.bool, device=queries.device),
            ~center_pad_mask,
        ], dim=1)                                                          # (B, N+V)

        q = self.norm1(combined)
        q2, _ = self.self_attn(q, q, q, key_padding_mask=key_padding_mask)
        combined = combined + q2
        queries = combined[:, :N]                                          # keep only query slots

        # Cross-attention: queries attend into other-frames' latents
        # Skipped entirely in single-frame mode (H = 0) or when no memory was given.
        if self.temporal and memory is not None and memory.size(1) > 0:
            if memory_key_padding_mask is None:
                valid = torch.ones(B, memory.size(1), dtype=torch.bool, device=queries.device)
            else:
                valid = memory_key_padding_mask.bool()                     # True = valid frame
            has_ctx = valid.any(dim=1)                                     # (B,)
            safe_valid = valid | ~has_ctx.unsqueeze(1)
            q = self.norm2(queries)
            q2, _ = self.cross_attn(q, memory, memory, key_padding_mask=~safe_valid)
            queries = torch.where(has_ctx.view(B, 1, 1), queries + q2, queries)

        queries = queries + self.mlp(self.norm3(queries))
        return queries


class GlobalDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hidden, dim), nn.Dropout(drop))

    def forward(self, queries, center_tokens, center_pad_mask, memory=None, memory_key_padding_mask=None):
        """
        Args: as TemporalDecoderBlock.forward
            queries: (B, N, D)
            center_tokens: (B, V, D), center_pad_mask: (B, V) bool, True = real token
            memory: (B, M, D) or None, memory_key_padding_mask: (B, M) bool, True = valid frame
        Returns:
            queries: (B, N, D)
        """
        B, N, D = queries.shape
        dev = queries.device

        q = self.norm1(queries)
        keys = [q, self.norm1(center_tokens)]
        ignore = [torch.zeros(B, N, dtype=torch.bool, device=dev), ~center_pad_mask.bool()]
        if memory is not None and memory.size(1) > 0:
            valid = (torch.ones(B, memory.size(1), dtype=torch.bool, device=dev)
                     if memory_key_padding_mask is None else memory_key_padding_mask.bool())
            keys.append(memory)
            ignore.append(~valid)
        kv = torch.cat(keys, dim=1)                                        # (B, N+V+M, D)
        key_padding_mask = torch.cat(ignore, dim=1)                        # True = ignore

        q2, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        queries = queries + q2
        queries = queries + self.mlp(self.norm3(queries))
        return queries
    

class GlobalSelfAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = float(attn_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, attn_mask):
        """
        x:         (B, L, D)
        attn_mask: (B, 1, 1, L) bool, True = this key may be attended to
        """
        B, L, D = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                   # (B, h, L, hd)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop if self.training else 0.0
        )
        x = x + self.proj_drop(self.proj(o.transpose(1, 2).reshape(B, L, D)))
        x = x + self.mlp(self.norm2(x))
        return x


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channels of a (B, C, L) tensor, at every position."""

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def _make_norm(kind, channels, momentum):
    kind = str(kind).lower()
    if kind == "batch":
        return nn.BatchNorm1d(channels, momentum=momentum)
    if kind == "layer":
        return _ChannelLayerNorm(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"norm must be 'batch', 'layer' or 'none', got {kind!r}")

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.25, norm="batch", bn_momentum=0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel widths must be odd so the centre stays aligned"
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.crop = (self.kernel_size - 1) * self.dilation // 2
        bias = str(norm).lower() == "none"            # a norm right after makes the bias redundant
        self.conv = nn.Conv1d(channels, channels, self.kernel_size, dilation=self.dilation, bias=bias)
        self.norm1 = _make_norm(norm, channels, bn_momentum)
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=bias)
        self.norm2 = _make_norm(norm, channels, bn_momentum)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x[:, :, self.crop:x.shape[2] - self.crop]
        x = self.drop(self.act(self.norm1(self.conv(x))))
        x = self.drop(self.act(self.norm2(self.pointwise(x))))
        return res + x
