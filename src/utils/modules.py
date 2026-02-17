import torch
from torch import nn
import numpy as np
import math

class GELUActivationXAI(nn.Module):
    """
    GeLU activation module.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
    
    def conservative_forward(self, input: torch.Tensor) -> torch.Tensor:
        gelu = input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
        out = input * (gelu/(input+1e-9)).detach()
        return out
    

class LayerNormXAI(nn.Module):
    """
    Layer Normalization Module.
    """
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Args:
            normalized_shape (int or tuple): 
                Input shape from an expected input of size
            eps (float):
                A value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine (bool):
                Whether to include learnable affine parameters. Default: True
        """
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, input):
        """
        Forward pass
        
        Args:
            input (torch.Tensor): Input tensor of shape (..., *normalized_shape)
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        # Compute mean and std along the last len(normalized_shape) dimensions
        # For typical usage with shape (B, N, C), this normalizes over C
        dims = list(range(-len(self.normalized_shape), 0))
        
        # Compute mean and standard deviation
        mean = input.mean(dim=dims, keepdim=True)
        std = torch.sqrt(((input - mean) ** 2).mean(dim=dims, keepdim=True))
        input_norm = (input - mean) / (std + self.eps)
        
        # Apply affine transformation if enabled
        if self.elementwise_affine:
            return input_norm * self.weight + self.bias
        else:
            return input_norm
        
        
    def conservative_forward(self, input):
        """
        Forward pass with detachment trick for explainability.
        
        Args:
            input (torch.Tensor): Input tensor of shape (..., *normalized_shape)
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        # Compute mean and std along the last len(normalized_shape) dimensions
        # For typical usage with shape (B, N, C), this normalizes over C
        dims = list(range(-len(self.normalized_shape), 0))
        
        # Compute mean and standard deviation
        mean = input.mean(dim=dims, keepdim=True)
        variance = ((input - mean) ** 2).mean(dim=dims, keepdim=True)
        std = torch.sqrt(variance)
        
        # Normalize: detach std to prevent gradients flowing through variance
        input_norm = (input - mean.detach()) / (std.detach() + self.eps)
        
        # Apply affine transformation if enabled
        if self.elementwise_affine:
            return input_norm * self.weight + self.bias
        else:
            return input_norm

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class Mlp(nn.Module):
    """
    A Multi-layer perceptron module.
    Shape:
        Input:  (B, N, C)
        Output: (B, N, out_features)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELUActivationXAI, drop=0.):
        """
        Initialization function
        Args:
            in_features (int):
                Input feature dimension.
            hidden_features (int or None, optional):
                Dimension of the hidden layer. If None, defaults to `in_features`.
            out_features (int or None, optional):
                Output feature dimension. If None, defaults to `in_features`.
            act_layer (nn.Module, optional):
                Activation function applied between the two linear layers.
                Default: GELUActivationXAI.
            drop (float, optional):
                Dropout probability applied after the second linear layer.
                Default: 0.0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        """
        Forward pass of the MLP module.
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
    """
    Multi-Head Self-Attention module.

    Shape:
        Input: (B, N, C)
        Output: (B, N, C)
    """

    def __init__(self, input_dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., attn_head_dim=None):
        """
        Initialization function
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
        Compute the forward pass of the multi-head self-attention module.

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
        
    def conservative_forward(self, x, output_attn=False):
        """
        Compute the forward pass of the multi-head self-attention module.

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

        # Apply softmax to get attention weights.
        attn = attn.softmax(dim=-1)
        # For faithful explanations
        attn = attn.detach()
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

    def forward(self, x_q, x_kv):
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
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., act_layer=GELUActivationXAI, 
                 norm_layer=LayerNormXAI, attn_head_dim=None):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                       act_layer=act_layer, drop=drop)

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
    
    def conservative_forward(self, x):
        """
        Apply attention and MLP sublayers with residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        x = x + self.attn.conservative_forward(self.norm1.conservative_forward(x))
        x = x + self.mlp(self.norm2.conservative_forward(x))
        return x
    

class AttentionPoolingBlock(nn.Module):
    """
    Transformer pooling block using Pooling by Multihead Attention (PMA).
    
    Shape:
        Input:  (B, N, C) - Batch, Number of Keypoints, Channels
        Output: (B, 1, C) - Batch, 1, Channels (Pooled latent vector)
    """
    def __init__(self, input_dim, output_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., act_layer=GELUActivationXAI, norm_layer=LayerNormXAI
                 ):
        super().__init__()
        
        # Learnable Seed Vector (The Query)
        self.seed = nn.Parameter(torch.randn(1, 1, output_dim))
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Normalization Layers
        self.norm_seed = norm_layer(output_dim)   # Norm for the query (seed)
        self.norm_ctx = norm_layer(output_dim)    # Norm for the key/value (input points)

        # Cross Attention Module
        self.attn = PoolingAttention(
            output_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
            
        # MLP
        self.norm2 = norm_layer(output_dim)       # Norm before MLP
        self.mlp = Mlp(in_features=output_dim, hidden_features=int(output_dim * mlp_ratio), act_layer=act_layer, drop=drop)

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


class CrossViewTransformerBlock(nn.Module):
    """
    Single layer of cross-attention from query (per-view predictions)
    to key/value (all view encodings).
    """
    
    def __init__(self, dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
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
    """Sinusoidal positional encoding for keypoint indices.
    
    Encodes the position/index of each keypoint in the sequence.
    Can be extended to include per-camera positional information.
    """
    
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