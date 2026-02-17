import torch
from torch import nn
from utils.modules import TransformerBlock, LayerNormXAI, GELUActivationXAI, AttentionPoolingBlock, PositionalEncoding, CrossViewTransformerBlock

class SpatialTransformerEncoder(nn.Module):
    """
    Multi-View Spatial Transformer Encoder.
    
    Takes in 2D projected keypoints from multiple views and produces
    per-view latent representations with cross-view consistency.
    
    Architecture:
    - Keypoint embedding (2D -> embed_dim)
    - Per-view positional encoding
    - Learnable view tokens (one per camera)
    - Transformer blocks with self-attention
    - Attention pooling to extract global features
    
    Args:
        conf (dict): Configuration containing:
            - embed_dim: Embedding dimension
            - depth: Number of transformer blocks
            - num_heads: Number of attention heads
            - mlp_ratio: MLP hidden dimension ratio
            - qkv_bias: Whether to use bias in QKV projections
            - qk_scale: Scaling factor for QK similarity
            - drop_rate: Dropout rate
            - attn_drop_rate: Attention dropout rate
            - proj_dim: Output projection dimension
            - enable_last_norm: Whether to apply LayerNorm at output
            - use_occlusion_mask: Whether to use occlusion masks
        num_keypoints (int): Number of keypoints (default 10)
        num_views (int): Number of camera views (default 3)
        norm_layer: Normalization layer (default LayerNormXAI)
        act_layer: Activation layer (default GELUActivationXAI)
    """

    def __init__(self, conf, num_keypoints=10, num_views=3, norm_layer=LayerNormXAI, act_layer=GELUActivationXAI):
        super(SpatialTransformerEncoder, self).__init__()

        self.num_keypoints = num_keypoints
        self.num_views = num_views
        self.embed_dim = conf['embed_dim']
        self.depth = conf['depth']
        self.num_heads = conf['num_heads']
        self.mlp_ratio = conf['mlp_ratio']
        self.qkv_bias = conf['qkv_bias']
        self.qk_scale = conf.get('qk_scale', None)
        self.drop_rate = conf['drop_rate']
        self.attn_drop_rate = conf['attn_drop_rate']
        self.proj_dim = conf['proj_dim']

        # Embedding
        self.keypoint_embed = nn.Linear(2, self.embed_dim)
        self.pos_encoding = PositionalEncoding(self.embed_dim, max_seq_len=num_keypoints + 1)
        
        # Learnable camera view latents
        self.view_tokens = nn.Parameter(torch.zeros(1, num_views, 1, self.embed_dim))
        self.view_pos_embed = nn.Parameter(torch.zeros(1, num_views, 1, self.embed_dim))

        # Transformer blocks for inter-view feature extraction
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim, 
                num_heads=self.num_heads, 
                mlp_ratio=self.mlp_ratio, 
                qkv_bias=self.qkv_bias, 
                qk_scale=self.qk_scale,
                drop=self.drop_rate, 
                attn_drop=self.attn_drop_rate, 
                act_layer=act_layer, 
                norm_layer=norm_layer,
            )
            for _ in range(self.depth - 1)
        ])
        
        # Attention pooling to aggregate all tokens into one latent variable
        self.pool = AttentionPoolingBlock(
            input_dim=self.embed_dim, 
            output_dim=self.proj_dim, 
            num_heads=self.num_heads, 
            mlp_ratio=self.mlp_ratio, 
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale, 
            drop=self.drop_rate, 
            attn_drop=self.attn_drop_rate, 
            act_layer=act_layer, 
            norm_layer=norm_layer
        )
        
        self.last_norm = norm_layer(self.proj_dim) if conf['enable_last_norm'] else nn.Identity()
        
        # Initialize weights
        self._reset_parameters()
        self._init_weights()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        nn.init.trunc_normal_(self.view_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.view_tokens, std=0.02)

    def get_num_layers(self):
        return len(self.blocks) + 2

    def forward(self, x, occlusion_mask):
        """
        Forward pass through the transformer encoder with dynamic sequence shrinking.
        
        Args:
            x (torch.Tensor): Input keypoints of shape (B, C, N, 2)
            occlusion_mask (torch.Tensor, optional): Shape (B, C, N, 1)
                            Boolean mask where True indicates visible keypoint.
        
        Returns:
            torch.Tensor: Encoded features per view, shape (B, C, proj_dim)
        """
        B, C, N, D = x.shape
        assert D == 2, f"Expected last dimension to be 2 (x,y coordinates), got {D}"
        assert N == self.num_keypoints, f"Expected {self.num_keypoints} keypoints, got {N}"
        assert C == self.num_views, f"Expected {self.num_views} views, got {C}"
        
        # Embed keypoints
        x_flat = x.view(B * C, N, D)
        x_embed = self.keypoint_embed(x_flat) + self.pos_encoding(x_flat)
        x_embed = x_embed.view(B, C, N, self.embed_dim)
        
        # Add view tokens
        view_tokens = (self.view_tokens + self.view_pos_embed).expand(B, C, 1, self.embed_dim)
        x_with_tokens = torch.cat([view_tokens, x_embed], dim=2)  # (B, C, N+1, embed_dim)
        
        # Create mask including view tokens
        mask_with_tokens = torch.cat([
            torch.ones(B, C, 1, 1, dtype=torch.bool, device=x.device), 
            occlusion_mask
        ], dim=2)  # (B, C, N+1, 1)
        
        # Reshape for processing
        x_with_tokens = x_with_tokens.view(B*C, N+1, self.embed_dim)
        mask_with_tokens = mask_with_tokens.view(B*C, N+1, 1)
        num_visible = mask_with_tokens[:, :, 0].sum(dim=1)[0].item()
        
        # Extract visible tokens PER SAMPLE 
        visible_list = []
        for i in range(B * C):
            sample_mask = mask_with_tokens[i, :, 0]  # (N+1,)
            sample_visible = x_with_tokens[i, sample_mask]
            visible_list.append(sample_visible)
        visible_data = torch.stack(visible_list)  # (B*C, num_visible, embed_dim)
        
        # Process through transformer
        for block in self.blocks:
            visible_data = block(visible_data)
        
        # Reshape back to per-view
        x_seq = visible_data.view(B, C, num_visible, self.embed_dim)
        
        # Pool per view
        x_pooled = self.pool(x_seq)
        x_pooled = self.last_norm(x_pooled)
        
        return x_pooled
        
    def _create_view_attention_mask(self, B, C, N_with_token, device):
        """
        Create block diagonal attention mask for multi-view processing.
        Each view only attends to its own tokens, not to other views.
        
        Args:
            B: Batch size
            C: Number of views
            N_with_token: Number of tokens per view (N_keypoints + 1)
            device: torch device
            
        Returns:
            Attention mask of shape (B, C*N_with_token, C*N_with_token)
            with 0.0 for allowed attention and -inf for blocked attention
        """
        total_tokens = C * N_with_token
        # Initialize with -inf
        mask = torch.full((B, total_tokens, total_tokens), float('-inf'), device=device)
        
        # Each view attends only to itself
        for c in range(C):
            start_idx = c * N_with_token
            end_idx = (c + 1) * N_with_token
            mask[:, start_idx:end_idx, start_idx:end_idx] = 0.0
        
        return mask


class SpatialTransformerDecoder(nn.Module):
    """
    Decoder with cross-attention for multi-view keypoint inpainting.
    """

    def __init__(self, conf, num_keypoints=10, num_views=3):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.num_views = num_views
        self.latent_dim = conf['proj_dim']
        self.num_heads = conf['decoder_num_heads']
        self.num_layers = conf['decoder_layers']
        self.dim_feedforward = conf['decoder_dim_feedforward']
        self.dropout = conf['decoder_dropout']
        self.predict_depth = conf['predict_depth']
        self.use_depth_conditioning = conf['use_depth_conditioning']
        self.output_confidence = conf['confidence']
        
        # Learnable queries per view and per keypoint
        self.keypoint_embeddings = nn.Parameter(torch.randn(num_keypoints, self.latent_dim))
        self.view_embeddings = nn.Parameter(torch.randn(num_views, self.latent_dim))

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossViewTransformerBlock(
                dim=self.latent_dim,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Coordinate head
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim // 2, self.latent_dim // 4),
            nn.GELU(),
            nn.Linear(self.latent_dim // 4, 2)
        )
        
        # Depth head
        if self.predict_depth:
            if self.use_depth_conditioning:
                self.depth_encoder = nn.Sequential(
                    nn.Linear(1, self.latent_dim // 4),
                    nn.GELU(),
                    nn.Linear(self.latent_dim // 4, self.latent_dim // 2)
                )
                # Project encoded features + depth encoding back to latent_dim
                self.memory_projection = nn.Linear(
                    self.latent_dim + self.latent_dim // 2, 
                    self.latent_dim
                )
            self.depth_predictor = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim // 2, self.latent_dim // 4),
                nn.GELU(),
                nn.Linear(self.latent_dim // 4, 1),
                nn.Softplus()
            )
        
        if self.output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(self.latent_dim, 1),
                nn.Sigmoid()
            ) 

        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.keypoint_embeddings, std=0.02)
        nn.init.normal_(self.view_embeddings, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, encoded_features, original_keypoints, occlusion_mask, depth_map):
        """
        Args:
            encoded_features: (B, C, proj_dim) - per-view encodings from encoder
            original_keypoints: (B, C, N, 2) - input keypoints
            occlusion_mask: (B, C, N) - True=visible, False=occluded
            depth_map (B, C, N) - depth values for keypoints (optional)
        
        Returns:
            dict with completed keypoints
        """
        B, C, N, _ = original_keypoints.shape
        D = self.latent_dim
        results = {}

        # Gather occluded positions for efficient batch processing
        occluded_mask = ~occlusion_mask
        results['occluded_mask'] = occluded_mask
        occluded_positions = torch.nonzero(occluded_mask, as_tuple=False)
        batch_idx = occluded_positions[:, 0]
        view_idx = occluded_positions[:, 1]
        keypoint_idx = occluded_positions[:, 2]

        memory = encoded_features  # (B, C, D)
        
        if self.use_depth_conditioning and depth_map is not None:
            # For each sample, aggregate depth information from visible keypoints
            depth_features = []
            for b in range(B):
                for c in range(C):
                    visible_mask = occlusion_mask[b, c]
                    if visible_mask.any():
                        vis_depths = depth_map[b, c, visible_mask].unsqueeze(-1)
                        encoded_depths = self.depth_encoder(vis_depths)  # (num_vis, D//2)
                        depth_feat = encoded_depths.mean(dim=0)  # (D//2,)
                    else:
                        depth_feat = torch.zeros(D // 2, device=encoded_features.device)
                    depth_features.append(depth_feat)
            depth_features = torch.stack(depth_features).view(B, C, D // 2)
            
            # Concatenate and project back
            memory_with_depth = torch.cat([memory, depth_features], dim=-1)  # (B, C, D + D//2)
            memory = self.memory_projection(memory_with_depth)  # (B, C, D)

        
        # Create queries only for occluded positions
        queries = self.keypoint_embeddings[keypoint_idx] + self.view_embeddings[view_idx]
        queries = queries.unsqueeze(1)  # (num_occluded_total, 1, D)
        
        # Each query should attend to ALL views to gather multi-view info
        memory_for_queries = encoded_features[batch_idx]

        # Apply cross-attention layers
        for layer in self.cross_attention_layers:
            queries = layer(queries, memory_for_queries)

        query_features = queries.squeeze(1)  # (num_occluded_total, D)
        predictions = self.predictor(query_features)  # (num_occluded_total, 2)

        # Scatter predictions back to original positions
        keypoints_completed = original_keypoints.clone()
        keypoints_completed[batch_idx, view_idx, keypoint_idx] = predictions

        results['coordinates'] = keypoints_completed

        if self.output_confidence:
            results['confidence']  = self.confidence_head(query_features) # (num_occluded_total, 1)

        if self.predict_depth:
            depth_predictions = self.depth_predictor(query_features)  # (num_occluded_total, 1)
            
            # Scatter depth predictions back to original positions
            depth_map = torch.zeros(B, C, N, 1, device=original_keypoints.device)
            depth_map[batch_idx, view_idx, keypoint_idx] = depth_predictions
            results['depth'] = depth_map.squeeze(-1)  # (B, C, N)
        
        return results

    

class MultiView3DKeypointModel(nn.Module):
    def __init__(self, conf, num_keypoints=10, num_views=3, norm_layer=LayerNormXAI, act_layer=GELUActivationXAI):
        super().__init__()

        self.encoder = SpatialTransformerEncoder(conf=conf['encoder'], num_keypoints=num_keypoints, num_views=num_views, norm_layer=norm_layer, act_layer=act_layer)
        self.decoder = SpatialTransformerDecoder(conf=conf['decoder'], num_keypoints=num_keypoints, num_views=num_views)

    def forward(self, keypoints_2d, occlusion_mask, depth_map = None):
        encoded_features = self.encoder(keypoints_2d, occlusion_mask)
        decoded = self.decoder(encoded_features, keypoints_2d, occlusion_mask, depth_map)
        decoded['encoder_features'] = encoded_features
        return decoded