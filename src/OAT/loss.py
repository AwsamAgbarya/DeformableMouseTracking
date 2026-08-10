import torch
import torch.nn as nn
import torch.nn.functional as F


class ModularKeypointLoss(nn.Module):
    """
    Modular multi-task loss for masked keypoint prediction.

    Main task:
        Masked coordinate regression (deformable 3D keypoints, normalized).
        Optionally re-weighted by predicted per-keypoint uncertainty
        (log-sigma parameterization), applied ONLY to occluded/masked points.

    Auxiliary tasks (each independently toggle-able via `config`):
        - use_uncertainty:   sigma = exp(log_sigma)
                              loss = residual/sigma + alpha*log_sigma
                              (inverse-confidence: higher predicted uncertainty
                              DISCOUNTS the residual penalty -> divide, not multiply)
        - use_relative_dist: masked regression on relative-distance features
                              (e.g. rigid<->deformable offset / bond length/vector)

    All losses are computed ONLY over points flagged occluded in the mask.
    Non-occluded points are treated as ground truth and never contribute,
    regardless of what the model happens to output there.

    Expected `pred` dict keys (all optional except 'coordinates'):
        coordinates : (B, N, 3)              predicted deformable coords
        log_sigma   : (B, N, 3)  [optional]  predicted log-uncertainty
        rel_dist    : (B, N, D)  [optional]  predicted relative-distance feature

    Expected `target` dict keys:
        coordinates   : (B, N, 3)             GT deformable coords
        occluded_mask : (B, N) or (B, N, 1)   bool, True == occluded == used in loss
        rel_dist      : (B, N, D) [optional]  GT relative-distance feature
    """

    def __init__(self, config = None):
        super().__init__()
        config = config or {}

        # ---- per-task toggles -----------------------------------------
        self.use_uncertainty = config.get("use_uncertainty", False)
        self.use_relative_dist = config.get("use_relative_dist", False)

        # ---- loss shape ------------------------------------------------
        self.main_loss_type = config.get("main_loss_type", "l2")       # 'l1' | 'l2' | 'smooth_l1'
        self.reldist_loss_type = config.get("reldist_loss_type", "l1")

        # ---- weights -----------------------------------------------------
        self.lambda_main = config.get("lambda_main", 1.0)
        self.lambda_uncertainty_reg = config.get("lambda_uncertainty_reg", 1.0)  # alpha
        self.lambda_reldist = config.get("lambda_reldist", 1.0)

        self.eps = config.get("eps", 1e-6)
        self.log_sigma_clamp = config.get("log_sigma_clamp", (-7.0, 7.0))  # numerical safety

    @staticmethod
    def _prep_mask(mask):
        """Collapse (B,N,1) -> (B,N) and cast to bool."""
        if mask.dim() == 3:
            mask = mask[..., 0]
        return mask.bool()

    @staticmethod
    def _elementwise_residual(pred, gt, loss_type):
        if loss_type == "l1":
            return F.l1_loss(pred, gt, reduction="none")
        elif loss_type == "l2":
            return F.mse_loss(pred, gt, reduction="none")
        elif loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, gt, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def _masked_mean(self, residual, mask):
        """
        residual: (B, N, D) elementwise loss
        mask:     (B, N) bool, True = include
        Mean over masked (points x dims); safe (zero, no NaN) if mask is all-False.
        """
        mask_exp = mask.unsqueeze(-1).expand_as(residual)
        n_valid = mask_exp.sum()
        if n_valid == 0:
            return residual.sum() * 0.0  # keeps dtype/device/graph, contributes zero
        return (residual * mask_exp).sum() / (n_valid + self.eps)

    def forward(self, pred, target):
        mask = self._prep_mask(target["occluded_mask"])  # (B, N) True = occluded = used

        losses = {}
        residual = self._elementwise_residual(pred["coordinates"], target["coordinates"], self.main_loss_type)  # (B, N, 3)

        if self.use_uncertainty and "log_sigma" in pred:
            log_sigma = pred["log_sigma"].clamp(*self.log_sigma_clamp)
            sigma = torch.exp(log_sigma)
            # inverse-confidence: divide by sigma (not Sigma-odot-residual)
            weighted_residual = residual / (sigma + self.eps) + self.lambda_uncertainty_reg * log_sigma
            main_loss = self._masked_mean(weighted_residual, mask)
            losses["coord_raw"] = self._masked_mean(residual, mask).detach()  # monitoring only
        else:
            main_loss = self._masked_mean(residual, mask)

        losses["weighted_coord"] = main_loss

        if self.use_relative_dist and "rel_dist" in pred and "rel_dist" in target:
            reldist_residual = self._elementwise_residual(pred["rel_dist"], target["rel_dist"], self.reldist_loss_type)
            losses["relative_dist"] = self._masked_mean(reldist_residual, mask)
        else:
            losses["relative_dist"] = torch.zeros((), device=pred["coordinates"].device)

        total = self.lambda_main * losses["weighted_coord"]
        if self.use_relative_dist:
            total = total + self.lambda_reldist * losses["relative_dist"]

        losses["total"] = total
        return losses