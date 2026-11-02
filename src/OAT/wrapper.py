"""
Occlusion Aware Transformer dataset.
"""
import torch
from torch.utils.data import Dataset
from OAT.pose_extraction import (
    rot2,
    combine_window_frames,
    procrustes_se2,
    se2_inverse_apply,
    wrap,
)

class TemporalWrapper(Dataset):
    """
    Builds a temporal window around each valid centre frame. Each window has a
    fixed size and grabs the T_half nearest USABLE frames from each side inside
    a max_radius search space.
    """

    def __init__(self, coords_world, template_xy, joint_weights, config=None, rigid_world=None, occlusion=None, normalizer=None, scale = None):
        config = config or {}
        self.config = config

        self.T_half = config.get("T_half", 8)               # frames per side
        self.max_radius = config.get("max_radius", 32)      # max search radius
        self.seed = config.get("seed", 42)
        self.require_full_context = config.get("require_full_context", True)
        self.allow_dropped_center = config.get("allow_dropped_center", True)

        # Decay lengths, in FRAMES, of the centre-decaying weights that collapse the per-frame fits into the window frame.
        self.tau_yaw = float(config.get("tau_yaw", max(0.75 * self.T_half, 1.0)))
        self.tau_xy = float(config.get("tau_xy", 1.0))
        self.use_frame_conf = bool(config.get("use_frame_conf", True))
        self.canonicalization = str(config.get("canonicalization", "window")).lower()
        assert self.canonicalization in ("window", "per_frame"), (
            f"canonicalization must be 'window' or 'per_frame', got {self.canonicalization!r}")

        self.coords_world = torch.as_tensor(coords_world, dtype=torch.float32)
        assert self.coords_world.dim() == 3 and self.coords_world.shape[-1] == 3, (f"coords_world must be (T, N, 3), got {tuple(self.coords_world.shape)}")
        self.T, self.J = self.coords_world.shape[0], self.coords_world.shape[1]

        self.rigid_world = None if rigid_world is None else torch.as_tensor(rigid_world, dtype=torch.float32)
        if self.rigid_world is not None:
            assert self.rigid_world.shape == self.coords_world.shape, (f"rigid_world must match coords_world")

        self.template_xy = torch.as_tensor(template_xy, dtype=torch.float32)
        self.joint_weights = torch.as_tensor(joint_weights, dtype=torch.float32).reshape(-1)
        assert self.template_xy.shape == (self.J, 2), (f"template must be ({self.J}, 2), got {tuple(self.template_xy.shape)}")
        assert self.joint_weights.shape[0] == self.J, (f"joint_weights has {self.joint_weights.shape[0]} entries, data has {self.J} joints")
        self.total_weight = float(self.joint_weights.sum())

        self.occlusion = occlusion
        if occlusion is not None:
            assert occlusion.T == self.T and occlusion.K == self.J, (f"occlusion is ({occlusion.T}, {occlusion.K}), data is ({self.T}, {self.J})")

        assert self.max_radius >= self.T_half, (f"max_radius ({self.max_radius}) < T_half ({self.T_half}): a full window is unreachable even with zero dropout")
        self._check_rotation_safe(normalizer)

        self.epoch = 0
        self.set_epoch(0)
        self.scale = float(scale) if scale is not None else self.fit_scale()
    @classmethod
    def from_global_dataset(cls, gd, config=None, occlusion=None, with_rigid=True, scale=None):
        """Wire a utils.dataset.GlobalDataset straight in."""
        return cls(
            coords_world=gd.deform_global,
            template_xy=gd.template_xy,
            joint_weights=gd.joint_weights,
            config=config,
            rigid_world=gd.rigid_global if with_rigid else None,
            occlusion=occlusion,
            normalizer=getattr(gd, "normalizer", None),
            scale=scale,
        )

    @staticmethod
    def _check_rotation_safe(normalizer):
        if normalizer is None:
            return
        c = getattr(normalizer, "center", None)
        if c is None:
            return
        if getattr(normalizer, "center_mode", "none") == "none":
            return
        if abs(float(c[0])) > 1e-6 or abs(float(c[1])) > 1e-6:
            raise ValueError(
                "Canonicalizing normalized data needs center_x == center_y == 0, "
                f"got center={c.tolist()}. Otherwise normalization no longer "
                "commutes with the window transform and every window picks up a "
                "pose-dependent translation. Refit with rotation_safe=True, or "
                "keep center_mode='none'."
            )
    @torch.no_grad()
    def fit_scale(self, q=0.995, max_samples=5000):
        self.scale = 1.0          # fit on unmasked, canonical windows
        samples = [self[int(i)] for i in torch.randperm(len(self))[:max_samples]]
        pts = torch.cat([s["window"][s["vis_mask"]].reshape(-1, 3) for s in samples])   # (P, 3)
        self.scale = float(pts.norm(dim=-1).quantile(q))
        return self.scale

    # epoch

    def set_epoch(self, epoch):
        """Call at the start of every training epoch, then rebuild the DataLoader."""
        self.epoch = int(epoch)

        if self.occlusion is not None:
            self.occlusion.set_epoch(self.epoch)
            self.visible = self.occlusion.visible                 # (T, J) bool
            self.frame_unusable = self.occlusion.frame_dropped    # (T,)   bool
        else:
            self.visible = torch.ones(self.T, self.J, dtype=torch.bool)
            self.frame_unusable = torch.zeros(self.T, dtype=torch.bool)

        self._fit_frames()
        self.valid_centers = self._build_valid_centers(self.frame_unusable)

    def _fit_frames(self):
        """
        Per-frame body-frame fit for the WHOLE recording, from the visible joints only.
        """
        w = self.joint_weights.unsqueeze(0) * self.visible.float()
        theta, t, conf_R, conf_t = procrustes_se2(
            self.template_xy,
            self.coords_world[..., :2],
            w,
            total_weight=self.total_weight,
        )
        self.frame_yaw_info_ratio = self.yaw_info_ratio(self.visible)
        self.frame_theta = theta          # (T,)
        self.frame_xy = t                 # (T, 2)
        self.frame_conf_R = conf_R        # (T,)
        self.frame_conf_t = conf_t        # (T,)

    def yaw_info_ratio(self, visible):
        """Visible rotational information / full-skeleton information, in [0, 1]."""
        p = self.template_xy                                              # (N, 2)
        def info(w):                                                    # w: (..., N)
            W = w.sum(-1, keepdim=True).clamp_min(1e-8)
            p_bar = (w.unsqueeze(-1) * p).sum(-2, keepdim=True) / W.unsqueeze(-1)
            return (w * ((p - p_bar) ** 2).sum(-1)).sum(-1)
        return info(self.joint_weights * visible.float()) / info(self.joint_weights)

    def __len__(self):
        return len(self.valid_centers)

    def _build_valid_centers(self, context_mask):
        if not self.require_full_context:
            ok = torch.arange(self.T, dtype=torch.long)
            if not self.allow_dropped_center:
                ok = ok[~context_mask[ok]]
            return ok

        K, H = self.max_radius, self.T_half
        valid = (~context_mask).float()
        cs = torch.cat([torch.zeros(1), valid.cumsum(0)])     # cs[i] = #valid in [0, i)

        i = torch.arange(self.T)
        past = cs[i] - cs[(i - K).clamp(min=0)]               # valid in [i-K, i)
        fut = cs[(i + K + 1).clamp(max=self.T)] - cs[i + 1]   # valid in (i, i+K]
        ok = torch.nonzero((past >= H) & (fut >= H)).squeeze(-1)

        if not self.allow_dropped_center:
            ok = ok[~context_mask[ok]]

        assert ok.numel(), (
            f"no valid centre frames at frame-drop rate "
            f"{float(context_mask.float().mean()):.1%}. Increase max_radius ({K}), "
            f"decrease T_half ({H}), or lower the occlusion frame_drop rates."
        )
        return ok.to(torch.long)

    def _gather_side(self, center, direction):
        """direction: -1 past, +1 future. All indices are GLOBAL."""
        idxs, offs = [], []
        step = 1
        while len(idxs) < self.T_half and step <= self.max_radius:
            j = center + direction * step
            if 0 <= j < self.T and not self.frame_unusable[j]:
                idxs.append(j)
                offs.append(direction * step)
            step += 1

        pad_mask = torch.zeros(self.T_half, dtype=torch.bool)
        if len(idxs) < self.T_half:
            assert not self.require_full_context, (
                "padding fired under require_full_context=True, which should be "
                "impossible. _build_valid_centers and _gather_side disagree."
            )
            n_missing = self.T_half - len(idxs)
            pad_mask[len(idxs):] = True
            fill_i = idxs[-1] if idxs else center
            fill_o = offs[-1] if offs else 0
            idxs += [fill_i] * n_missing
            offs += [fill_o] * n_missing

        if direction == -1:
            idxs, offs = idxs[::-1], offs[::-1]
            pad_mask = pad_mask.flip(0)
        return (
            torch.tensor(idxs, dtype=torch.long),
            torch.tensor(offs, dtype=torch.float32),
            pad_mask,
        )

    def __getitem__(self, idx):
        center = int(self.valid_centers[idx])
        p_i, p_o, p_m = self._gather_side(center, -1)
        f_i, f_o, f_m = self._gather_side(center, +1)
        center_dropped = bool(self.frame_unusable[center])

        window_idx = torch.cat([p_i, torch.tensor([center]), f_i])
        offsets = torch.cat([p_o, torch.tensor([0.0]), f_o])
        pad_mask = torch.cat([p_m, torch.tensor([center_dropped]), f_m])
        evidence = ~pad_mask

        window = self.coords_world[window_idx]                # (W, J, 3) world
        vis = self.visible[window_idx].clone()                # (W, J) True = observed
        vis[pad_mask] = False

        # one SE(2) frame for the whole window
        theta_i = self.frame_theta[window_idx]                # (W,)
        t_i = self.frame_xy[window_idx]                       # (W, 2)
        conf_R = self.frame_conf_R[window_idx] if self.use_frame_conf else None
        conf_t = self.frame_conf_t[window_idx] if self.use_frame_conf else None
        frame_yaw_info_ratio = self.frame_yaw_info_ratio[window_idx] if self.use_frame_conf else None

        theta_w, t_w, frame_ok = combine_window_frames(
            theta_i, t_i, offsets, self.tau_yaw, self.tau_xy,
            conf_R=frame_yaw_info_ratio, conf_t=conf_t, valid=evidence,
        )
        score = self.frame_conf_t[window_idx] * evidence.float()
        j = int(torch.argmax(score))
        if not bool(frame_ok):
            if float(score[j]) > 0:
                theta_w, t_w = theta_i[j], t_i[j]
                frame_ok = True
            else:
                frame_ok = False

        if self.canonicalization == "per_frame":
            window = se2_inverse_apply(window, theta_i, t_i) / self.scale       # (W, J, 3)
        else:
            window = se2_inverse_apply(window, theta_w, t_w)  / self.scale    # (W, J, 3)
        window = window.masked_fill(~vis.unsqueeze(-1), 0.0)
        target = se2_inverse_apply(self.coords_world[center], theta_w, t_w) / self.scale  # (J, 3)

        # per-frame pose RELATIVE to the window frame, as a feature
        rel_yaw = wrap(theta_i - theta_w)                     # (W,)
        rel_xy = torch.einsum("ij,wj->wi", rot2(-theta_w), t_i - t_w) / self.scale # (W, 2)
        rel_yaw = rel_yaw.masked_fill(pad_mask, 0.0)
        rel_xy = rel_xy.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        out = {
            "window": window,                                 # (W, J, 3) canonical
            "vis_mask": vis,                                  # (W, J) True = visible
            "offsets": offsets,                               # (W,) frame distance
            "rel_yaw": rel_yaw,                               # (W,) yaw vs window frame
            "rel_xy": rel_xy,                                 # (W, 2) xy vs window frame
            "pad_mask": pad_mask,                             # (W,) True = no evidence
            "frame_conf": torch.stack([self.frame_conf_R[window_idx], self.frame_conf_t[window_idx]], dim=-1), # (W, 2)
            "center_dropped": torch.tensor(center_dropped),
            "center_index": torch.tensor(center),
            "center_vis": vis[self.T_half],                   # (J,) True = visible
            "target_coords": target,                          # (J, 3) MAIN target
            "window_yaw": theta_w.reshape(1),                 # maps back to world
            "window_xy": t_w.reshape(2),
            "frame_ok": torch.tensor(bool(frame_ok)),
        }

        if self.rigid_world is not None:
            rigid = se2_inverse_apply(self.rigid_world[center], theta_w, t_w) / self.scale
            out["target_rigid"] = rigid                       # (J, 3)

        return out

    # report

    def epoch_report(self):
        """One dict per epoch, safe to hand straight to wandb."""
        vis = self.visible
        centers = self.valid_centers
        rep = {
            "epoch": self.epoch,
            "n_centers": int(centers.numel()),
            "frames_total": self.T,
            "frame_dropped_frac": float(self.frame_unusable.float().mean()),
            "center_dropped_frac": float(self.frame_unusable[centers].float().mean()),
            "hidden_frac": float((~vis[~self.frame_unusable]).float().mean()),
            "conf_R_mean": float(self.frame_conf_R[~self.frame_unusable].mean()),
            "conf_t_mean": float(self.frame_conf_t[~self.frame_unusable].mean()),
        }
        if self.occlusion is not None:
            rep.update({f"occ/{k}": v for k, v in self.occlusion.report().items()
                        if not isinstance(v, list)})
        return rep