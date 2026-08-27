import torch
from torch import nn
from utils.modules import GlobalAttentionPoolingBlock, TransformerBlock, PositionalEncoding, TemporalDecoderBlock, RelativeTemporalEncoding

class SnapshotEncoder(nn.Module):
    """
    Per-frame encoder. Embeds visible keypoints, contextualizes them via
    self-attention among themselves (masking out occluded points as keys),
    and pools them into a single compact per-frame latent.

    Also optionally returns the raw contextualized per-keypoint tokens.
    This distinction is central to the fixed design:
      - The CENTER frame needs its per-keypoint tokens (return_tokens=True),
        so the decoder can let occluded center-frame queries directly
        self-attend with the center frame's own visible keypoints --
        full-detail, same-frame information, not compressed through a latent.
      - Every OTHER (context) frame only needs its pooled latent
        (return_tokens=False), which keeps the temporal cross-attention
        memory to size T-1 (one token per frame) instead of (T-1)*N.
        This is what makes the design efficient and matches the original
        intended architecture: "one latent per frame" for temporal memory,
        with only the center frame exposing raw token-level detail.

    Args:
        conf (dict): Configuration containing:
            - embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale,
              drop_rate, attn_drop_rate, proj_dim, enable_last_norm
        num_keypoints (int): Number of keypoints per frame (default 10)
    """
    def __init__(self, conf, num_keypoints=10):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.embed_dim = conf['embed_dim']
        self.depth = conf['depth']
        self.num_heads = conf['num_heads']
        self.mlp_ratio = conf['mlp_ratio']
        self.qkv_bias = conf['qkv_bias']
        self.qk_scale = conf.get('qk_scale', None)
        self.drop_rate = conf['drop_rate']
        self.attn_drop_rate = conf['attn_drop_rate']
        self.proj_dim = conf['proj_dim']

        self.keypoint_embed = nn.Linear(3, self.embed_dim)
        # Learnable per-keypoint IDENTITY embedding (not sinusoidal positional
        # encoding) -- keypoints have no natural ordinal relationship, so an
        # embedding table that lets the model freely represent "which
        # keypoint" is the right tool, consistent with the decoder's identity
        # embeddings used elsewhere in this codebase.
        self.kp_identity_embed = nn.Embedding(num_keypoints, self.embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
            )
            for _ in range(self.depth - 1)
        ])

        self.pool = GlobalAttentionPoolingBlock(
            input_dim=self.embed_dim, output_dim=self.proj_dim, num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
            drop=self.drop_rate, attn_drop=self.attn_drop_rate,
        )
        self.last_norm = nn.LayerNorm(self.proj_dim) if conf.get('enable_last_norm', True) else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_num_layers(self):
        return len(self.blocks) + 2

    def forward(self, x, occlusion_mask, return_tokens=False):
        """
        Args:
            x: (B, N, 3) 3D keypoints
            occlusion_mask: (B, N, 1) or (B, N) bool, True = visible
            return_tokens: if True, also return contextualized per-keypoint
                tokens (needed only for the center frame)

        Returns:
            latent: (B, proj_dim) pooled per-frame latent
            [if return_tokens] hidden: (B, V, E) contextualized visible tokens
            [if return_tokens] pad_mask: (B, V) bool, True = real token
            [if return_tokens] kp_idx: (B, V) long, keypoint id per token (-1=pad)
        """
        B, N, D = x.shape
        vis_mask = occlusion_mask[..., 0] if occlusion_mask.dim() == 3 else occlusion_mask
        vis_mask = vis_mask.bool()

        kp_idx_all = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        x_embed = self.keypoint_embed(x) + self.kp_identity_embed(kp_idx_all)  # (B, N, E)

        visible_list, kp_idx_list = [], []
        for i in range(B):
            m = vis_mask[i]
            visible_list.append(x_embed[i, m])
            kp_idx_list.append(torch.nonzero(m, as_tuple=True)[0])

        max_visible = max(max((v.size(0) for v in visible_list), default=0), 1)
        padded = torch.zeros(B, max_visible, self.embed_dim, device=x.device, dtype=x_embed.dtype)
        pad_mask = torch.zeros(B, max_visible, dtype=torch.bool, device=x.device)
        kp_idx = torch.full((B, max_visible), -1, dtype=torch.long, device=x.device)
        for i, v in enumerate(visible_list):
            padded[i, :v.size(0)] = v
            pad_mask[i, :v.size(0)] = True
            kp_idx[i, :v.size(0)] = kp_idx_list[i]

        # Only VISIBLE tokens are usable as attention KEYS. Padding slots are
        # masked out for every query row, so they never contaminate context.
        key_invalid = ~pad_mask
        attn_mask = torch.zeros(B, max_visible, max_visible, device=x.device, dtype=padded.dtype)
        attn_mask.masked_fill_(key_invalid.unsqueeze(1), torch.finfo(padded.dtype).min)

        hidden = padded
        for block in self.blocks:
            hidden = block(hidden, attn_mask=attn_mask)

        latent = self.last_norm(self.pool(hidden))

        if return_tokens:
            return latent, hidden, pad_mask, kp_idx
        return latent

class SnapshotDecoder(nn.Module):
    """
    Decoder for single-view global 3D keypoint completion.

    Inputs:
        encoded_features: (B, D)        global latent from encoder
        original_keypoints: (B, N, 3)   input 3D pose, with visible GT values
        occlusion_mask: (B, N) or (B, N, 1), True = visible, False = occluded

    Output:
        dict with:
            coordinates: (B, N, 3) completed pose
            occluded_mask: (B, N) boolean mask of missing keypoints
    """

    def __init__(self, conf, num_keypoints=10):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.latentdim = conf["proj_dim"]
        self.dropout = conf.get("decoder_dropout", 0.1)

        hidden_dim = conf.get("decoder_dim_feedforward", self.latentdim * 2)

        # One learnable query per keypoint identity
        self.keypoint_queries = nn.Parameter(torch.randn(num_keypoints, self.latentdim))

        # Positional encoding over keypoint index
        self.pos_encoding = PositionalEncoding(self.latentdim, max_seq_len=num_keypoints)

        # Fuse [query, latent] -> decoded keypoint feature
        self.query_fuser = nn.Sequential(
            nn.Linear(self.latentdim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, self.latentdim),
            nn.GELU(),
        )

        # Predict 3D coordinates
        self.predictor = nn.Sequential(
            nn.Linear(self.latentdim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.keypoint_queries, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, encoded_features, original_keypoints, occlusion_mask):
        """
        Args:
            encoded_features:   (B, D)
            original_keypoints: (B, N, 3)
            occlusion_mask:     (B, N) or (B, N, 1), True=visible, False=occluded

        Returns:
            dict:
                coordinates:   (B, N, 3)
                occluded_mask: (B, N)
        """
        B, N, D = original_keypoints.shape
        assert D == 3, f"Expected 3D keypoints, got last dim {D}"
        assert N == self.numkeypoints, f"Expected {self.numkeypoints} keypoints, got {N}"

        if occlusion_mask.dim() == 3:
            occlusion_mask = occlusion_mask[..., 0]
        occlusion_mask = occlusion_mask.bool()

        occluded_mask = ~occlusion_mask  # True where we need prediction

        # Start from GT and only overwrite hidden points
        completed_keypoints = original_keypoints.clone()

        # Build per-keypoint query tokens with positional meaning
        query_tokens = self.keypoint_queries.unsqueeze(0).expand(B, -1, -1)   # (B, N, D)
        query_tokens = query_tokens + self.pos_encoding(query_tokens)          # (B, N, D)

        # Broadcast latent to all keypoints
        latent_tokens = encoded_features.unsqueeze(1).expand(B, N, -1)        # (B, N, D)

        # Fuse keypoint identity query with global latent
        decoded_features = self.query_fuser(
            torch.cat([query_tokens, latent_tokens], dim=-1)
        )                                                                     # (B, N, D)

        # Only predict occluded joints
        if occluded_mask.any():
            pred_3d = self.predictor(decoded_features[occluded_mask])         # (num_occ, 3)
            completed_keypoints[occluded_mask] = pred_3d

        return {
            "coordinates": completed_keypoints,
            "occluded_mask": occluded_mask,
        }

class TemporalDecoder(nn.Module):
    """
    Decodes every keypoint of the center frame (occluded ones get a genuine
    prediction; visible ones are overwritten with ground truth at the end)
    using a stack of TemporalDecoderBlocks that alternate:
      - full-detail self-attention with the center frame's own visible tokens
      - efficient cross-attention into other frames' pooled temporal memory

    Optional auxiliary heads (toggle via conf), applied only to occluded
    center-frame queries:
      - predict_uncertainty: per-keypoint aleatoric log-sigma (VGGT-style)
      - predict_bond_aux: per-keypoint rigid-to-deform offset / bond length
    """
    def __init__(self, conf, num_keypoints=10, window_size=21, max_offset=96):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.center_idx = window_size // 2
        dim = conf['proj_dim']

        self.predict_uncertainty = conf.get('predict_uncertainty', False)
        self.predict_bond_aux = conf.get('predict_bond_aux', False)
        self.bond_aux_dim = conf.get('bond_aux_dim', 3)

        # One learnable query per keypoint IDENTITY (covers all N keypoints
        # of the center frame, not just the occluded ones -- visible ones are
        # simply overwritten with GT at the end, same as before).
        self.query_embed = nn.Parameter(torch.randn(num_keypoints, dim) * 0.02)

        # Temporal encoding uses the REAL relative offset of each context
        # frame from the center frame, so non-consecutive windows (nearest
        # valid frames gathered from each direction) still get geometrically
        # meaningful positions.
        self.time_embed = RelativeTemporalEncoding(dim, max_offset=max_offset)

        self.blocks = nn.ModuleList([
            TemporalDecoderBlock(dim, conf['num_heads'], conf['mlp_ratio'], conf.get('decoder_dropout', 0.1))
            for _ in range(conf['depth'])
        ])

        hidden = conf.get('decoder_dim_feedforward', dim * 2)
        drop = conf.get('decoder_dropout', 0.1)
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 3))
        if self.predict_bond_aux:
            self.bond_aux_head = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, self.bond_aux_dim))

    def build_temporal_memory(self, context_latents, context_offsets):
        """
        Args:
            context_latents: (B, T-1, D) pooled latent for each OTHER frame
            context_offsets: (T-1,) or (B, T-1) real temporal offset from center

        Returns:
            memory: (B, T-1, D) latents tagged with their temporal offset
        """
        B, Tm1, D = context_latents.shape
        if context_offsets.dim() == 1:
            context_offsets = context_offsets.view(1, Tm1).expand(B, Tm1)
        return context_latents + self.time_embed(context_offsets.long())

    def forward(self, center_tokens, center_pad_mask, context_latents, context_offsets,
                context_valid_mask, occluded_mask_center, original_keypoints_center):
        """
        Args:
            center_tokens: (B, V, D) center frame's own contextualized visible tokens
            center_pad_mask: (B, V) bool, True = real visible token
            context_latents: (B, T-1, D) per-other-frame pooled latents
            context_offsets: (T-1,) or (B, T-1) real offsets from center frame
            context_valid_mask: (B, T-1) bool, True = frame actually usable
            occluded_mask_center: (B, N) bool, True = needs prediction
            original_keypoints_center: (B, N, 3) center frame GT

        Returns:
            dict with coordinates, occluded_mask, and optional log_sigma /
            bond_aux_pred (only for occluded keypoints).
        """
        B, N, _ = original_keypoints_center.shape
        memory = self.build_temporal_memory(context_latents, context_offsets)

        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            queries = block(queries, center_tokens, center_pad_mask, memory,
                             memory_key_padding_mask=context_valid_mask)

        completed = original_keypoints_center.clone()
        output = {"occluded_mask": occluded_mask_center}

        if occluded_mask_center.any():
            occ_queries = queries[occluded_mask_center]
            pred = self.predictor(occ_queries)
            completed[occluded_mask_center] = pred

            if self.predict_uncertainty:
                log_sigma = self.uncertainty_head(occ_queries)
                sigma_full = torch.zeros(B, self.num_keypoints, 3, device=queries.device, dtype=log_sigma.dtype)
                sigma_full[occluded_mask_center] = log_sigma
                output["log_sigma"] = sigma_full

            if self.predict_bond_aux:
                bond_pred = self.bond_aux_head(occ_queries)
                bond_full = torch.zeros(B, self.num_keypoints, self.bond_aux_dim, device=queries.device, dtype=bond_pred.dtype)
                bond_full[occluded_mask_center] = bond_pred
                output["rel_dist"] = bond_full

        output["coordinates"] = completed
        return output

class SnapshotModel(nn.Module):
    def __init__(self, conf, num_keypoints=10):
        super().__init__()

        self.encoder = SnapshotEncoder(
            conf=conf["encoder"],
            num_keypoints=num_keypoints,
        )

        self.decoder = SnapshotDecoder(
            conf=conf["decoder"],
            num_keypoints=num_keypoints,
        )

    def forward(self, keypoints_3d, occlusion_mask):
        encoded_features = self.encoder(keypoints_3d, occlusion_mask)  # (B, projdim)
        decoded = self.decoder(encoded_features, keypoints_3d, occlusion_mask)
        decoded["encoder_features"] = encoded_features
        return decoded


class TemporalModel(nn.Module):

    def __init__(self, conf, num_keypoints=10, window_size=21, max_offset=96):
        super().__init__()
        self.window_size = window_size
        self.center_idx = window_size // 2
        self.encoder = SnapshotEncoder(conf['encoder'], num_keypoints)
        self.decoder = TemporalDecoder(conf['decoder'], num_keypoints, window_size, max_offset)

    def forward(self, keypoints_window, occlusion_window, frame_offsets=None):
        """
        Args:
            keypoints_window: (B, T, N, 3)
            occlusion_window: (B, T, N) True = visible
            frame_offsets: (B, T) real temporal offset of each gathered frame
                from the center frame. Defaults to (t - center_idx) if not
                provided, i.e. assumes a consecutive window.

        Returns:
            dict with coordinates (B, N, 3), occluded_mask (B, N), and
            optional log_sigma / bond_aux_pred.
        """
        B, T, N, _ = keypoints_window.shape
        device = keypoints_window.device
        if frame_offsets is None:
            frame_offsets = torch.arange(T, device=device) - self.center_idx
            frame_offsets = frame_offsets.unsqueeze(0).expand(B, T)
        else:
            frame_offsets = frame_offsets.to(device)

        
        flat_kpts = keypoints_window.reshape(B * T, N, 3) # (B, T, N, 3) -> (B*T, N, 3)
        flat_occ = occlusion_window.reshape(B * T, N) # (B, T, N) -> (B*T, N)

        flat_latent, flat_tokens, flat_pad_mask, flat_kp_idx = self.encoder(flat_kpts, flat_occ, return_tokens=True)
        # flat_latent:   (B*T, D)
        # flat_tokens:   (B*T, V, D)   V = max visible count across the WHOLE batch*time
        # flat_pad_mask: (B*T, V)
        # flat_kp_idx:   (B*T, V)

        D = flat_latent.shape[-1]
        V = flat_tokens.shape[1]

        latent = flat_latent.view(B, T, D)
        tokens = flat_tokens.view(B, T, V, D)
        pad_mask = flat_pad_mask.view(B, T, V)
        center_tokens = tokens[:, self.center_idx]        # (B, V, D)
        center_pad_mask = pad_mask[:, self.center_idx]     # (B, V)

        keep = torch.ones(T, dtype=torch.bool, device=device)
        keep[self.center_idx] = False

        context_latents = latent[:, keep]                  # (B, T-1, D)
        context_offsets = frame_offsets[:, keep]            # (B, T-1)
        context_valid = torch.ones(B, T - 1, dtype=torch.bool, device=device)

        center_occlusion = occlusion_window[:, self.center_idx].bool()
        if center_occlusion.dim() == 3:
            center_occlusion = center_occlusion[..., 0]
        occluded_mask_center = ~center_occlusion
        center_keypoints = keypoints_window[:, self.center_idx]

        return self.decoder(center_tokens, center_pad_mask, context_latents, context_offsets,
                             context_valid, occluded_mask_center, center_keypoints)


class GlobalAttentionModel(nn.Module):
    """
   Args:
        conf (dict): Configuration containing:
            - embed_dim: Embedding dimension
            - depth: Number of global self-attention transformer blocks
            - num_heads: Number of attention heads
            - mlp_ratio: MLP hidden dim ratio
            - qkv_bias: Whether to use bias in QKV projections
            - qk_scale: Scaling factor for QK similarity
            - drop_rate: Dropout rate
            - attn_drop_rate: Attention dropout rate
            - decoder_dim_feedforward: hidden dim of the coordinate predictor head
            - decoder_dropout: dropout for the predictor head
            - enable_last_norm: whether to apply LayerNorm before prediction
            - predict_uncertainty: bool, enable per-keypoint uncertainty head
            - predict_bond_aux: bool, enable auxiliary rigid-to-deform head
            - bond_aux_dim: int, 1 = scalar bond length, 3 = full relative_dist vector
        num_keypoints (int): Number of keypoints per frame (default 10)
        window_size (int): Max number of frames T gathered per window (used only
            to size RelativeTemporalEncoding's lookup range; actual offsets can
            be non-consecutive real frame distances, not loop indices)
    """

    def __init__(self, conf, num_keypoints=10, window_size=21):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.window_size = window_size
        self.embed_dim = conf['embed_dim']
        self.depth = conf['depth']
        self.num_heads = conf['num_heads']
        self.mlp_ratio = conf['mlp_ratio']
        self.qkv_bias = conf['qkv_bias']
        self.qk_scale = conf.get('qk_scale', None)
        self.drop_rate = conf['drop_rate']
        self.attn_drop_rate = conf['attn_drop_rate']

        self.predict_uncertainty = conf.get('predict_uncertainty', False)
        self.predict_bond_aux = conf.get('predict_bond_aux', False)
        self.bond_aux_dim = conf.get('bond_aux_dim', 1)  # 1 = scalar bond length, 3 = full relative_dist vector

        # --- Token construction ---
        self.keypoint_embed = nn.Linear(3, self.embed_dim)

        # Learnable token substituted in for occluded/invalid keypoints,
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # In-frame learnale encoding
        self.kp_identity_embed = nn.Embedding(num_keypoints, self.embed_dim)

        # Temporal encoding:
        self.time_embed = RelativeTemporalEncoding(self.embed_dim, max_offset=window_size)

        # Global self-attention backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
            )
            for _ in range(self.depth)
        ])

        self.last_norm = nn.LayerNorm(self.embed_dim) if conf.get('enable_last_norm', True) else nn.Identity()

        hidden = conf.get('decoder_dim_feedforward', self.embed_dim * 2)
        dropout = conf.get('decoder_dropout', 0.1)
        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )

        if self.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(self.embed_dim, hidden), nn.GELU(),
                nn.Linear(hidden, 3)
            )

        if self.predict_bond_aux:
            self.bond_aux_head = nn.Sequential(
                nn.Linear(self.embed_dim, hidden), nn.GELU(),
                nn.Linear(hidden, self.bond_aux_dim)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.mask_token, std=0.02)

    def get_num_layers(self):
        return len(self.blocks) + 1

    def build_tokens(self, keypoints_window, occlusion_window, frame_offsets):
        """
        Args:
            keypoints_window: (B, T, N, 3) 3D keypoints for T gathered frames
                (T frames need not be temporally consecutive)
            occlusion_window: (B, T, N, 1) bool, True = visible, False = occluded
            frame_offsets: (T,) or (B, T) real temporal offset of each gathered
                frame relative to the center frame (e.g. real_frame_idx - center_frame_idx),
                NOT the loop index t

        Returns:
            tokens: (B, T*N, E) tagged token sequence
            key_valid: (B, T*N) bool, True = usable as attention KEY (visible)
            kp_idx_flat: (B, T*N) long, keypoint identity per token (for bookkeeping)
            frame_idx_flat: (B, T*N) long, which frame (0..T-1) each token came from
        """
        B, T, N, _ = keypoints_window.shape
        device = keypoints_window.device

        vis_mask = occlusion_window[..., 0] if occlusion_window.dim() == 4 else occlusion_window
        vis_mask = vis_mask.bool()  # (B, T, N) True=visible

        coord_embed = self.keypoint_embed(keypoints_window)  # (B, T, N, E)

        mask_tok = self.mask_token.expand(B, T, N, self.embed_dim)
        tokens = torch.where(vis_mask.unsqueeze(-1), coord_embed, mask_tok)  # (B, T, N, E)

        kp_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, T, N)
        tokens = tokens + self.kp_identity_embed(kp_idx)  # in-frame positional (identity) encoding

        if frame_offsets.dim() == 1:
            frame_offsets = frame_offsets.view(1, T).expand(B, T)
        offsets_per_token = frame_offsets.unsqueeze(-1).expand(B, T, N).long()  # (B, T, N)
        tokens = tokens + self.time_embed(offsets_per_token)  # temporal encoding

        tokens = tokens.reshape(B, T * N, self.embed_dim)
        key_valid = vis_mask.reshape(B, T * N)
        kp_idx_flat = kp_idx.reshape(B, T * N)
        frame_idx = torch.arange(T, device=device).view(1, T, 1).expand(B, T, N)
        frame_idx_flat = frame_idx.reshape(B, T * N)

        return tokens, key_valid, kp_idx_flat, frame_idx_flat

    def forward(self, keypoints_window, occlusion_window, frame_offsets):
        """
        Args:
            keypoints_window: (B, T, N, 3)
            occlusion_window: (B, T, N, 1) True=visible
            frame_offsets: (T,) or (B, T) real temporal distance of each gathered
                frame from the center frame (can be non-consecutive/irregular)
            center_frame_idx: int, index into the T dimension identifying which
                gathered frame is the "center" frame we are completing

        Returns:
            dict with:
                coordinates: (B, N, 3) completed center-frame keypoints
                occluded_mask: (B, N) bool, True where prediction was made
                log_sigma: (B, N, 3) [only if predict_uncertainty=True]
                bond_aux_pred: (B, N, bond_aux_dim) [only if predict_bond_aux=True]
        """
        B, T, N, _ = keypoints_window.shape
        center_frame_idx = T // 2
        tokens, key_valid, kp_idx_flat, frame_idx_flat = self.build_tokens(keypoints_window, occlusion_window, frame_offsets)

        key_invalid = ~key_valid  # (B, T*N)
        attn_mask = torch.zeros(B, T * N, T * N, device=tokens.device, dtype=tokens.dtype)
        attn_mask.masked_fill_(key_invalid.unsqueeze(1), torch.finfo(tokens.dtype).min)

        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden, attn_mask=attn_mask)
        hidden = self.last_norm(hidden)

        hidden = hidden.reshape(B, T, N, self.embed_dim)

        center_tokens = hidden[:, center_frame_idx]              # (B, N, E)
        center_occlusion = occlusion_window[:, center_frame_idx, :, 0].bool()  # (B, N)
        occluded_mask_center = ~center_occlusion

        center_keypoints = keypoints_window[:, center_frame_idx]  # (B, N, 3)
        completed = center_keypoints.clone()

        output = {"occluded_mask": occluded_mask_center}

        if occluded_mask_center.any():
            occ_queries = center_tokens[occluded_mask_center]  # (num_occ, E)

            pred = self.predictor(occ_queries)
            completed[occluded_mask_center] = pred

            if self.predict_uncertainty:
                log_sigma = self.uncertainty_head(occ_queries)   # (num_occ, 3)
                sigma_full = torch.zeros(B, self.num_keypoints, 3, device=center_tokens.device, dtype=log_sigma.dtype)
                sigma_full[occluded_mask_center] = log_sigma
                output["log_sigma"] = sigma_full

            if self.predict_bond_aux:
                bond_pred = self.bond_aux_head(occ_queries)      # (num_occ, bond_aux_dim)
                bond_full = torch.zeros(B, self.num_keypoints, self.bond_aux_dim, device=center_tokens.device, dtype=bond_pred.dtype)
                bond_full[occluded_mask_center] = bond_pred
                output["bond_aux_pred"] = bond_full

        output["coordinates"] = completed
        return output