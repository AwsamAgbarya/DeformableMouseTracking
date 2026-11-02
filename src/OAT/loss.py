"""
Loss for the temporal (windowed) occlusion-aware model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalKeypointLoss(nn.Module):
    """
    pred (from TemporalModel)
        coordinates   (B, N, 3)   required   observation where visible, prediction where not
        log_sigma     (B, N, 3)   optional   per-axis log sigma

    target (from the trainer's `_build_targets`)
        coordinates   (B, N, 3)   required   batch['target_coords']
        occluded_mask (B, N)      required   ~batch['center_vis'], True = scored
    """

    def __init__(self, config=None, scale_mm=1.0):
        super().__init__()
        cfg = dict(config or {})
        self.cfg = cfg

        self.main_loss = cfg.get("main_loss", "l2")              # l2 | l1 | huber
        self.use_uncertainty = bool(cfg.get("use_uncertainty", True))

        self.lambda_main = float(cfg.get("lambda_main", 1.0))
        self.beta = float(self.cfg.get("beta_nll", 0.5))
        
        aw = cfg.get("axis_weights", (1.0, 1.0, 1.0))
        self.register_buffer("axis_weights", torch.tensor(aw, dtype=torch.float32))

        self.eps = float(cfg.get("eps", 1e-6))
        self.log_sigma_clamp = tuple(cfg.get("log_sigma_clamp", (-8.0, 3.0)))
        self.huber_beta = float(cfg.get("huber_beta", 0.05))
        self.scale_mm = float(scale_mm)

    # helpers

    @staticmethod
    def _as_mask(m):
        if m.dim() == 3:
            m = m[..., 0]
        return m.bool()

    def _elementwise(self, pred, gt):
        if self.main_loss == "l1":
            return (pred - gt).abs()
        if self.main_loss == "l2":
            return (pred - gt).pow(2)
        if self.main_loss == "huber":
            return F.smooth_l1_loss(pred, gt, reduction="none", beta=self.huber_beta)
        raise ValueError(f"Unknown main_loss: {self.main_loss}")

    @staticmethod
    def _masked_mean(residual, mask):
        """residual (B, N, D) or (B, N); mask (B, N) bool. Zero-safe."""
        m = mask.unsqueeze(-1) if residual.dim() == mask.dim() + 1 else mask
        m = m.to(residual.dtype).expand_as(residual)
        n = m.sum()
        if n == 0:
            return residual.sum() * 0.0
        return (residual * m).sum() / n

    # forward

    def forward(self, pred, target):
        occluded = self._as_mask(target["occluded_mask"])
        coords_p, coords_t = pred["coordinates"], target["coordinates"]
        losses, metrics = {}, {}

        # main
        res = self._elementwise(coords_p, coords_t) * self.axis_weights.view(1, 1, 3)

        if self.use_uncertainty and "log_sigma" in pred:
            log_sigma = pred["log_sigma"].clamp(*self.log_sigma_clamp)
            if self.main_loss == "l2":
                nll = 0.5 * res * torch.exp(-2.0 * log_sigma) + log_sigma
            else:
                nll = res * torch.exp(-log_sigma) + log_sigma
            if self.beta > 0:
                nll = nll * torch.exp(2.0 * self.beta * log_sigma).detach()
            main = self._masked_mean(nll, occluded)
            losses["coord_nll"] = main
            losses["coord_raw"] = self._masked_mean(res, occluded).detach()
            with torch.no_grad():
                z2 = (res * torch.exp(-2.0 * log_sigma))          # (e/σ)² per axis
                metrics["calib_z2"]     = self._masked_mean(z2, occluded)      # should be ≈ 1.0
                metrics["nll_quad"]     = 0.5 * metrics["calib_z2"]
                metrics["nll_logsig"]   = self._masked_mean(log_sigma, occluded)
                metrics["logsig_std"]   = log_sigma[occluded].std()      
        else:
            main = self._masked_mean(res, occluded)
            losses["coord"] = main

        total = self.lambda_main * main

        # metrics
        with torch.no_grad():
            err = torch.linalg.norm(coords_p - coords_t, dim=-1)           # (B, N)
            n = occluded.sum()
            metrics["mpjpe_mm"] = (
                (err * occluded).sum() / n * self.scale_mm if n > 0 else err.sum() * 0.0
            )
            if n > 0:
                sel = err[occluded]
                metrics["pjpe_p90_mm"] = sel.quantile(0.90) * self.scale_mm
                metrics["occluded_frac"] = occluded.float().mean()
                ax = (coords_p - coords_t).pow(2)[occluded].mean(0).sqrt() * self.scale_mm
                metrics["rmse_x_mm"], metrics["rmse_y_mm"], metrics["rmse_z_mm"] = ax[0], ax[1], ax[2]
                if self.use_uncertainty and "log_sigma" in pred and sel.numel() > 2:
                    # Is the predicted sigma informative, or is it just a
                    # constant? Correlation between predicted log sigma and the
                    # realized log error; a useful head scores well above 0.
                    ls = pred["log_sigma"].clamp(*self.log_sigma_clamp).mean(-1)[occluded]
                    le = torch.log(sel + self.eps)
                    lsc, lec = ls - ls.mean(), le - le.mean()
                    metrics["sigma_err_corr"] = (lsc * lec).sum() / (
                        lsc.norm() * lec.norm()).clamp_min(self.eps)

        losses["total"] = total
        losses.update({f"metric/{k}": v for k, v in metrics.items()})
        return losses