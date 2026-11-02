import torch
from torch import nn
from OAT.modules import GlobalAttentionPoolingBlock, TransformerBlock, GlobalSelfAttentionBlock, TemporalDecoderBlock, RelativeTemporalEncoding, GlobalDecoderBlock

def _mlp_head(dim, hidden, out_dim, drop=0.0):
    return nn.Sequential(
        nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, out_dim)
    )


class SnapshotEncoder(nn.Module):
    """
    Per-frame encoder. Embeds visible keypoints, contextualizes them via
    self-attention among themselves (occluded points are never embedded at all,
    so there is no path for a hidden coordinate to leak in), and pools them into
    a single compact per-frame latent.

    Also optionally returns the raw contextualized per-keypoint tokens.
    - The CENTER frame needs its per-keypoint tokens (return_tokens=True).
    - Every OTHER (context) frame only needs its pooled latent, keeping the
      temporal memory at size T-1.

    Args:
        conf (dict): embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, proj_dim, enable_last_norm, fast_pack
        num_keypoints (int)
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
        self.kp_identity_embed = nn.Embedding(num_keypoints, self.embed_dim)

        self.no_evidence_token = nn.Parameter(torch.zeros(1, self.embed_dim))

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
        nn.init.normal_(self.no_evidence_token, std=0.02)

    def get_num_layers(self):
        return len(self.blocks) + 2

    def _pack_fast(self, x_embed, vis_mask):
        """
        Vectorized equivalent of `_pack_loop`.

        A stable argsort on (~visible) puts the visible indices first in
        ascending order, which is exactly the order boolean indexing produced,
        so the packed tensors are identical element for element.
        """
        B, N, E = x_embed.shape
        dev = x_embed.device

        n_vis = vis_mask.sum(1)                                        # (B,)
        V = max(int(n_vis.max()), 1)

        order = torch.argsort((~vis_mask).to(torch.int8), dim=1, stable=True)[:, :V]
        gathered = torch.gather(x_embed, 1, order.unsqueeze(-1).expand(B, V, E))

        slot = torch.arange(V, device=dev).unsqueeze(0)                # (1, V)
        pad_mask = slot < n_vis.unsqueeze(1)                           # (B, V)
        padded = gathered * pad_mask.unsqueeze(-1).to(gathered.dtype)
        kp_idx = torch.where(pad_mask, order, torch.full_like(order, -1))

        # A frame with nothing visible (a dropped frame) gets the learned
        # no-evidence token in slot 0, and that slot counts as real.
        empty = (n_vis == 0)
        if bool(empty.any()):
            first = empty.unsqueeze(1) & (slot == 0)                   # (B, V)
            padded = torch.where(
                first.unsqueeze(-1),
                self.no_evidence_token.view(1, 1, E).expand(B, V, E),
                padded,
            )
            pad_mask = pad_mask | first
        return padded, pad_mask, kp_idx


    def forward(self, x, occlusion_mask, return_tokens=False):
        """
        Args:
            x: (B, N, 3) 3D keypoints
            occlusion_mask: (B, N, 1) or (B, N) bool, True = visible
            return_tokens: if True, also return contextualized per-keypoint
                tokens (needed only for the center frame)

        Returns:
            latent: (B, proj_dim)
            [if return_tokens] hidden: (B, V, E), pad_mask: (B, V), kp_idx: (B, V)
        """
        B, N, D = x.shape
        vis_mask = occlusion_mask[..., 0] if occlusion_mask.dim() == 3 else occlusion_mask
        vis_mask = vis_mask.bool()

        kp_idx_all = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)
        x_embed = self.keypoint_embed(x) + self.kp_identity_embed(kp_idx_all)  # (B, N, E)

        padded, pad_mask, kp_idx = self._pack_fast(x_embed, vis_mask)

        key_invalid = ~pad_mask
        attn_mask = torch.zeros(B, padded.shape[1], padded.shape[1], device=x.device, dtype=padded.dtype)
        attn_mask.masked_fill_(key_invalid.unsqueeze(1), torch.finfo(padded.dtype).min)

        hidden = padded
        for block in self.blocks:
            hidden = block(hidden, attn_mask=attn_mask)

        latent = self.last_norm(self.pool(hidden, pad_mask))

        if return_tokens:
            return latent, hidden, pad_mask, kp_idx
        return latent

class TemporalDecoder(nn.Module):
    """
    Decodes every keypoint of the center frame using a stack of
    TemporalDecoderBlocks that alternate:
      - full-detail self-attention with the center frame's own visible tokens
      - cross-attention into the other frames' pooled temporal memory
    Single-frame decoder (H = 0)

    conf flags:
        predict_uncertainty -> log_sigma  (B, N, 3)
        block_type          "alternating" (default, M0) | "global" (ablation A3a):
                            one attention over [queries; center tokens; memory]
                            per block instead of self- then cross-attention
        relpose_embedding   False (default) | True (ablation A4c): each context
                            frame's pose relative to the reference frame,
                            (cos, sin, x, y), is linearly embedded and added to
                            its memory slot
    """

    BLOCK_TYPES = ("alternating", "global")

    def __init__(self, conf, num_keypoints=10, window_size=21, max_offset=96):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.center_idx = window_size // 2
        self.temporal = window_size > 1
        dim = conf['proj_dim']

        self.predict_uncertainty = conf.get('predict_uncertainty', False)
        self.block_type = str(conf.get('block_type', 'alternating')).lower()
        assert self.block_type in self.BLOCK_TYPES, \
            f"decoder.block_type must be one of {self.BLOCK_TYPES}, got {self.block_type!r}"
        self.use_relpose = bool(conf.get('relpose_embedding', False))

        self.query_embed = nn.Parameter(torch.randn(num_keypoints, dim) * 0.02)
        self.time_embed = RelativeTemporalEncoding(dim, max_offset=max_offset) if self.temporal else None

        if self.block_type == "global":
            self.blocks = nn.ModuleList([
                GlobalDecoderBlock(dim, conf['num_heads'], conf['mlp_ratio'], conf.get('decoder_dropout', 0.1))
                for _ in range(conf['depth'])
            ])
            self.type_embed = nn.Parameter(torch.randn(2, dim) * 0.02)    # [center token, memory]
        else:
            self.blocks = nn.ModuleList([
                TemporalDecoderBlock(dim, conf['num_heads'], conf['mlp_ratio'], conf.get('decoder_dropout', 0.1),
                                     temporal=self.temporal)
                for _ in range(conf['depth'])
            ])

        self.relpose_embed = None
        if self.use_relpose and self.temporal:
            self.relpose_embed = nn.Linear(4, dim)
            nn.init.zeros_(self.relpose_embed.weight)
            nn.init.zeros_(self.relpose_embed.bias)

        hidden = conf.get('decoder_dim_feedforward', dim * 2)
        drop = conf.get('decoder_dropout', 0.1)

        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        if self.predict_uncertainty:
            self.uncertainty_head = _mlp_head(dim, hidden, 3, drop)

    def build_temporal_memory(self, context_latents, context_offsets, context_relpose=None):
        """
        Args:
            context_latents: (B, T-1, D) pooled latent for each OTHER frame
            context_offsets: (T-1,) or (B, T-1) real temporal offset from center
            context_relpose: (B, T-1, 4) optional (cos, sin, x, y) of each
                frame's pose relative to the reference frame (A4c only)
        Returns:
            memory: (B, T-1, D) latents tagged with their temporal offset
        """
        B, Tm1, D = context_latents.shape
        if context_offsets.dim() == 1:
            context_offsets = context_offsets.view(1, Tm1).expand(B, Tm1)
        memory = context_latents + self.time_embed(context_offsets.long())
        if self.relpose_embed is not None:
            if context_relpose is None:
                raise ValueError("decoder.relpose_embedding is on but no relative pose was passed; "
                                 "the trainer must hand batch['rel_yaw'] / batch['rel_xy'] to the model")
            memory = memory + self.relpose_embed(context_relpose.to(memory.dtype))
        return memory

    def forward(self, center_tokens, center_pad_mask, context_latents, context_offsets,
                context_valid_mask, occluded_mask_center, original_keypoints_center,
                context_relpose=None):
        """
        Args:
            center_tokens: (B, V, D) center frame's own contextualized visible tokens
            center_pad_mask: (B, V) bool, True = real visible token
            context_latents: (B, T-1, D) per-other-frame pooled latents, or None
                for a single frame (H = 0)
            context_offsets: (T-1,) or (B, T-1) real offsets from center frame,
                or None for a single frame
            context_valid_mask: (B, T-1) bool, True = frame actually usable,
                or None for a single frame
            occluded_mask_center: (B, N) bool, True = needs prediction
            original_keypoints_center: (B, N, 3) center frame observation,
                occluded entries already zeroed by the caller
            context_relpose: (B, T-1, 4) or None, relative pose of each context
                frame (only used with relpose_embedding, A4c)

        Returns:
            dict with coordinates, coords_dense, occluded_mask, and the optional
            log_sigma / frame_se2.
        """
        B, N, _ = original_keypoints_center.shape

        has_context = (
            self.temporal
            and context_latents is not None
            and context_latents.size(1) > 0
        )
        if has_context:
            memory = self.build_temporal_memory(context_latents, context_offsets, context_relpose)
        else:
            # Single-frame decoding: the blocks skip the temporal cross-attention.
            memory, context_valid_mask = None, None

        if self.block_type == "global":
            center_tokens = center_tokens + self.type_embed[0]
            if memory is not None:
                memory = memory + self.type_embed[1]

        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            queries = block(queries, center_tokens, center_pad_mask, memory,
                            memory_key_padding_mask=context_valid_mask)

        coords_dense = self.predictor(queries)                             # (B, N, 3)
        completed = torch.where(
            occluded_mask_center.unsqueeze(-1), coords_dense, original_keypoints_center
        )

        out = {
            "coordinates": completed,        # observation where visible, prediction where not
            "coords_dense": coords_dense,    # the same head everywhere, for inspection
            "occluded_mask": occluded_mask_center,
        }
        if self.predict_uncertainty:
            out["log_sigma"] = self.uncertainty_head(queries)
        return out

class TemporalModel(nn.Module):
    """
    Window in, center-frame pose out.

    The arguments come straight from a TemporalWrapper batch:
        keypoints_window <- batch['window']    (B, W, N, 3), already in the
                            window's SE(2) frame
        occlusion_window <- batch['vis_mask']  (B, W, N) True = observed
        frame_offsets    <- batch['offsets']   (B, W) real frame distances
        pad_mask         <- batch['pad_mask']  (B, W) True = no evidence
        rel_yaw, rel_xy  <- batch['rel_yaw'] (B, W), batch['rel_xy'] (B, W, 2);
                            only read when decoder.relpose_embedding is on (A4c)

    decoder.center_mode
        "tokens" (default, M0): the center frame enters the decoder as its
                 per-keypoint tokens (self-attention); the memory holds the
                 OTHER frames only.
        "pooled" (ablation A3b): the center frame is pooled like every other
                 frame and enters only as z_0 + r_0 in the memory (masked when
                 the center is dropped); the queries self-attend among
                 themselves only.
    """

    CENTER_MODES = ("tokens", "pooled")

    def __init__(self, conf, num_keypoints, window_size, max_offset):
        super().__init__()
        self.window_size = window_size
        self.center_idx = window_size // 2
        self.num_keypoints = num_keypoints
        self.J = num_keypoints                       # reporting convenience
        self.W = window_size
        dec = conf['decoder']
        self.center_mode = str(dec.get('center_mode', 'tokens')).lower()
        assert self.center_mode in self.CENTER_MODES, \
            f"decoder.center_mode must be one of {self.CENTER_MODES}, got {self.center_mode!r}"
        assert not (self.center_mode == "pooled" and window_size == 1), \
            "center_mode='pooled' needs a temporal window (T_half >= 1): with H = 0 the decoder would see nothing"
        # The trainer and the evaluator pass rel_yaw / rel_xy only to models that ask for them.
        self.uses_relpose = bool(dec.get('relpose_embedding', False)) and window_size > 1
        self.encoder = SnapshotEncoder(conf['encoder'], num_keypoints)
        self.decoder = TemporalDecoder(conf['decoder'], num_keypoints, window_size, max_offset)

    def forward(self, keypoints_window, occlusion_window, frame_offsets=None, pad_mask=None,
                rel_yaw=None, rel_xy=None):
        B, T, N, _ = keypoints_window.shape
        device = keypoints_window.device
        # Center of the window actually passed in. Equal to self.center_idx when
        # T == window_size, and 0 for a single frame (H = 0), so a temporal model
        # can also be run on one frame without indexing past the end.
        center_idx = T // 2
        if frame_offsets is None:
            frame_offsets = torch.arange(T, device=device) - center_idx
            frame_offsets = frame_offsets.unsqueeze(0).expand(B, T)
        else:
            frame_offsets = frame_offsets.to(device)

        occlusion_window = occlusion_window.bool()
        if occlusion_window.dim() == 4:
            occlusion_window = occlusion_window[..., 0]

        keypoints_window = keypoints_window * occlusion_window.unsqueeze(-1).to(keypoints_window.dtype)

        flat_kpts = keypoints_window.reshape(B * T, N, 3)
        flat_occ = occlusion_window.reshape(B * T, N)

        flat_latent, flat_tokens, flat_pad_mask, flat_kp_idx = self.encoder(
            flat_kpts, flat_occ, return_tokens=True
        )

        D = flat_latent.shape[-1]
        V = flat_tokens.shape[1]

        latent = flat_latent.view(B, T, D)
        tokens = flat_tokens.view(B, T, V, D)
        padmask_kp = flat_pad_mask.view(B, T, V)
        center_tokens = tokens[:, center_idx]                      # (B, V, D)
        center_pad_mask = padmask_kp[:, center_idx]                # (B, V)

        context_relpose = None
        if T > 1:
            keep = torch.ones(T, dtype=torch.bool, device=device)
            if self.center_mode == "tokens":
                keep[center_idx] = False                           # M0: center is not in the memory
            # "pooled" (A3b): keep every slot, the center included, so the memory
            # holds z_0 + r_0; pad_mask[center] = center_dropped masks it when dropped.

            context_latents = latent[:, keep]                      # (B, T-1, D) or (B, T, D)
            context_offsets = frame_offsets[:, keep]

            if pad_mask is not None:
                context_valid = ~pad_mask.to(device).bool()[:, keep]
            else:
                context_valid = torch.ones(B, int(keep.sum()), dtype=torch.bool, device=device)

            if self.uses_relpose:
                if rel_yaw is None or rel_xy is None:
                    raise ValueError("decoder.relpose_embedding is on: pass rel_yaw=batch['rel_yaw'], "
                                     "rel_xy=batch['rel_xy'] to the model")
                rel_yaw = rel_yaw.to(device).float()
                rel = torch.cat([torch.cos(rel_yaw).unsqueeze(-1), torch.sin(rel_yaw).unsqueeze(-1),
                                 rel_xy.to(device).float()], dim=-1)          # (B, T, 4)
                context_relpose = rel[:, keep]
        else:
            # H = 0: a single frame, no other frames to attend to across time.
            context_latents = context_offsets = context_valid = None

        if self.center_mode == "pooled":
            # A3b: no per-keypoint center tokens reach the decoder.
            center_tokens = center_tokens[:, :0]                   # (B, 0, D)
            center_pad_mask = center_pad_mask[:, :0]               # (B, 0)

        occluded_mask_center = ~occlusion_window[:, center_idx]
        center_keypoints = keypoints_window[:, center_idx]

        return self.decoder(center_tokens, center_pad_mask, context_latents, context_offsets,
                            context_valid, occluded_mask_center, center_keypoints,
                            context_relpose=context_relpose)


def _get(conf, key, default):
    """conf.get for dicts and DictConfigs, with None meaning 'use the default'."""
    if conf is None:
        return default
    value = conf.get(key, default)
    return default if value is None else value


class GlobalSelfAttentionModel(nn.Module):
    """
    Window in, centre-frame pose out; a drop-in replacement for TemporalModel.

    Args:
        conf          the `model` config; architecture settings under conf["global"]
        num_keypoints N
        window_size   2 * T_half + 1 (only used for the centre index / reporting:
                      the model itself runs on any window length)
        max_offset    largest |offset| the offset embedding distinguishes
                      (train_dataset.max_radius, as for the transformer)
    """

    def __init__(self, conf, num_keypoints, window_size, max_offset):
        super().__init__()
        g = _get(conf, "global", {})
        self.num_keypoints = int(num_keypoints)
        self.J = self.num_keypoints
        self.window_size = int(window_size)
        self.W = self.window_size
        self.center_idx = self.window_size // 2
        self.max_offset = int(max_offset)

        dim = int(_get(g, "dim", 256))
        depth = int(_get(g, "depth", 9))
        heads = int(_get(g, "num_heads", 8))
        mlp_ratio = float(_get(g, "mlp_ratio", 4.0))
        qkv_bias = bool(_get(g, "qkv_bias", True))
        drop = float(_get(g, "drop_rate", 0.1))
        attn_drop = float(_get(g, "attn_drop_rate", 0.1))
        hidden = int(_get(g, "decoder_dim_feedforward", 2 * dim))
        self.predict_uncertainty = bool(_get(g, "predict_uncertainty", False))
        self.dim, self.depth = dim, depth

        N = self.num_keypoints
        # Token marking. Named *_embed so the trainer's parameter groups treat
        # them like the STT's embeddings (no weight decay).
        self.coord_embed = nn.Linear(3, dim)
        self.joint_embed = nn.Embedding(N, dim)
        self.time_embed = RelativeTemporalEncoding(dim, max_offset=self.max_offset)
        self.type_embed = nn.Parameter(torch.randn(2, dim) * 0.02)        # [query, observation]
        self.query_embed = nn.Parameter(torch.randn(N, dim) * 0.02)

        self.blocks = nn.ModuleList([
            GlobalSelfAttentionBlock(dim, heads, mlp_ratio, qkv_bias, drop, attn_drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        if self.predict_uncertainty:
            self.uncertainty_head = _mlp_head(dim, hidden, 3, drop)

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and not name.startswith(("type_embed", "query_embed", "time_embed", "joint_embed")):
                nn.init.xavier_uniform_(p)

    def extra_repr(self):
        return f"N={self.num_keypoints}, dim={self.dim}, depth={self.depth}, max_offset={self.max_offset}"

    @staticmethod
    def _pack_visible(usable):
        """
        usable: (B, S) bool over the flattened (frame, joint) slots.
        Returns the slot index of every visible token, packed to the left in
        ascending (frame, joint) order, and its validity mask:
            order (B, M) long, valid (B, M) bool, with M = max visible count.
        """
        B, S = usable.shape
        n = usable.sum(1)
        M = int(n.max()) if B > 0 else 0
        order = torch.argsort((~usable).to(torch.int8), dim=1, stable=True)[:, :M]
        valid = torch.arange(M, device=usable.device).unsqueeze(0) < n.unsqueeze(1)
        return order, valid

    def forward(self, keypoints_window, occlusion_window, frame_offsets=None, pad_mask=None,
                rel_yaw=None, rel_xy=None):
        """
        Args (straight from a TemporalWrapper batch, as for TemporalModel):
            keypoints_window (B, T, N, 3) window-frame coordinates
            occlusion_window (B, T, N) or (B, T, N, 1) bool, True = visible
            frame_offsets    (B, T) signed frame offsets from the centre
            pad_mask         (B, T) bool, True = slot holds no evidence
            rel_yaw, rel_xy  accepted and ignored (signature compatibility)
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

        # Defensive, as in TemporalModel: no hidden coordinate reaches the network.
        keypoints_window = keypoints_window * vis.unsqueeze(-1).to(keypoints_window.dtype)

        usable = vis
        if pad_mask is not None:
            usable = usable & ~pad_mask.to(device).bool().unsqueeze(-1)    # padded frames: no tokens

        # Gather the visible slots BEFORE embedding: hidden coordinates are never embedded.
        order, valid = self._pack_visible(usable.reshape(B, T * N))        # (B, M)
        M = order.shape[1]
        coords = torch.gather(keypoints_window.reshape(B, T * N, 3), 1, order.unsqueeze(-1).expand(B, M, 3))
        joint_id = order % N                                               # (B, M)
        frame_id = order // N
        offsets = torch.gather(frame_offsets.long(), 1, frame_id)          # (B, M) real offsets

        obs = (self.coord_embed(coords) + self.joint_embed(joint_id)
               + self.time_embed(offsets) + self.type_embed[1])
        obs = obs * valid.unsqueeze(-1).to(obs.dtype)                      # padding slots carry nothing

        ids = torch.arange(N, device=device)
        queries = (self.query_embed + self.joint_embed(ids)
                   + self.time_embed(torch.zeros_like(ids)) + self.type_embed[0])
        queries = queries.unsqueeze(0).expand(B, N, -1)

        x = torch.cat([queries, obs], dim=1)                               # (B, N + M, D)
        key_ok = torch.cat([torch.ones(B, N, dtype=torch.bool, device=device), valid], dim=1)
        attn_mask = key_ok[:, None, None, :]                               # (B, 1, 1, N + M)

        for block in self.blocks:
            x = block(x, attn_mask)
        q = self.norm(x[:, :N])                                            # (B, N, D)

        coords_dense = self.predictor(q)
        occluded = ~vis[:, center]
        completed = torch.where(occluded.unsqueeze(-1), coords_dense, keypoints_window[:, center])

        out = {
            "coordinates": completed,        # observation where visible, prediction where not
            "coords_dense": coords_dense,    # the head everywhere, for inspection
            "occluded_mask": occluded,
        }
        if self.predict_uncertainty:
            out["log_sigma"] = self.uncertainty_head(q)
        return out
