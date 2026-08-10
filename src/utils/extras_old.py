import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F
from collections import defaultdict

from MVT.models import MultiView3DKeypointModel

# Used in TestOAT
class ModelWrapper:
    def __init__(self, config, normalizer, triangulator, matcher, rigid_template):
        self.model = MultiView3DKeypointModel(config["model"]).to('cuda')
        checkpoint_path = "../checkpoints/MVT/MSE_loss/best_model.pt"
        self.load_checkpoint(checkpoint_path)

        self.normalizer = normalizer
        self.triangulator = triangulator
        self.matcher = matcher
        self.rigid_template = rigid_template.to('cuda')
        self.device='cuda'
    def __call__(self, keypoints_2d, masks, centers):
        output = self.model(
                keypoints_2d=keypoints_2d.unsqueeze(0).to('cuda'),
                occlusion_mask=masks.unsqueeze(0).to('cuda'),
            )
        pred_2d_norm = output.get('coordinates', None)
        # Denormalize prediction and add centers back
        pred_2d = self.normalizer.denormalize(pred_2d_norm.cpu())
        pred_2d = pred_2d.to(self.device).squeeze(0)
        pred_2d = pred_2d + centers.unsqueeze(1).to('cuda')
        confidences = (masks + 1).float().to('cuda')
        pred_3d_sample = self.triangulator.triangulate(pred_2d.unsqueeze(0), confidences=confidences.unsqueeze(0))
        aligned, _, _, _ = self.matcher.align(pred_3d_sample.squeeze(0), self.rigid_template, method='weighted')
        return pred_3d_sample.detach().cpu()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            best_loss = checkpoint.get('best_loss', 'unknown')
            print(f"✓ Loaded model from epoch {epoch}")
            print(f"✓ Best training loss: {best_loss}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded model weights")

# Not used yet
class BAAnalyzer:
    """
    Evaluates and compares Model, Triangulation, and Bundle Adjustment
    against ground truth, broken down by visibility group and axis.

    Usage
    -----
    analyzer = BAAnalyzer(dataset, model, ba)
    analyzer.evaluate()
    analyzer.plot_overall_mse()
    analyzer.plot_mse_per_group()
    analyzer.plot_mse_per_group_per_axis()
    analyzer.plot_visibility_distribution()
    """

    K     = 10
    AXES  = ["X", "Y", "Z"]
    N_VIS = [1, 2, 3]

    C_MODEL = "#3498db"
    C_TRI   = "#9b59b6"
    C_BA    = "#e67e22"
    C_VIS   = {1: "#e74c3c", 2: "#f39c12", 3: "#2ecc71"}

    def __init__(self, dataset, model, ba):
        self.dataset = dataset
        self.model   = model
        self.ba      = ba
        self._reset()

    # ── Storage ───────────────────────────────────────────────────────────────

    def _reset(self):
        self.mse_model_all = []
        self.mse_tri_all   = []
        self.mse_ba_all    = []

        self.pts_per_grp     = {v: [] for v in self.N_VIS}
        self.mse_model_grp   = {v: [] for v in self.N_VIS}
        self.mse_tri_grp     = {v: [] for v in self.N_VIS}
        self.mse_ba_grp      = {v: [] for v in self.N_VIS}

        self.mse_model_grp_ax = {v: {ax: [] for ax in self.AXES} for v in self.N_VIS}
        self.mse_tri_grp_ax   = {v: {ax: [] for ax in self.AXES} for v in self.N_VIS}
        self.mse_ba_grp_ax    = {v: {ax: [] for ax in self.AXES} for v in self.N_VIS}

        self.evaluated = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mse(pred, gt):
        return float(np.mean((pred - gt) ** 2))

    @staticmethod
    def _mse_per_axis(pred, gt):
        return np.mean((pred - gt) ** 2, axis=0)

    def _append_nan_group(self, v):
        for store in (self.mse_model_grp, self.mse_tri_grp, self.mse_ba_grp):
            store[v].append(np.nan)
        for store in (self.mse_model_grp_ax, self.mse_tri_grp_ax, self.mse_ba_grp_ax):
            for ax in self.AXES:
                store[v][ax].append(np.nan)

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, num_frames=None):
        self._reset()
        N = len(self.dataset) if num_frames is None else min(num_frames, len(self.dataset))

        for i in tqdm(range(N), desc="Evaluating"):
            item = self.dataset[i]
            gt           = item["deformable_current"]
            input_denorm = item["denorm_mv_coords"]
            input_norm   = item["mv_coords"]
            masks        = item["mask"]
            centers      = item["mv_centers"]

            pred    = self.model(input_norm, masks, centers).squeeze(0)
            ba_dict = self.ba(input_denorm, masks, gt)

            pred_np = pred.detach().cpu().numpy()
            gt_np   = gt.detach().cpu().numpy()
            ba_np   = ba_dict["points_3d"].numpy()
            tri_np  = ba_dict["points_3d_tri"].numpy()
            groups  = ba_dict["groups"]

            # 3. Points per group
            for v in self.N_VIS:
                self.pts_per_grp[v].append(len(groups[v]["idx"]))

            # 1. Overall MSE (recoverable points only)
            rec_mask = ~np.isnan(ba_np).any(axis=1)
            if rec_mask.sum() > 0:
                self.mse_model_all.append(self._mse(pred_np[rec_mask], gt_np[rec_mask]))
                self.mse_tri_all.append(  self._mse(tri_np[rec_mask],  gt_np[rec_mask]))
                self.mse_ba_all.append(   self._mse(ba_np[rec_mask],   gt_np[rec_mask]))
            else:
                self.mse_model_all.append(np.nan)
                self.mse_tri_all.append(np.nan)
                self.mse_ba_all.append(np.nan)

            # 2 & 4. Per-group + per-axis MSE
            for v in self.N_VIS:
                g   = groups[v]
                idx = g["idx"]

                if len(idx) == 0:
                    self._append_nan_group(v)
                    continue

                pred_g = pred_np[idx]
                gt_g   = g["gt"]

                self.mse_model_grp[v].append(self._mse(pred_g, gt_g))
                for ai, ax in enumerate(self.AXES):
                    self.mse_model_grp_ax[v][ax].append(
                        float(np.mean((pred_g[:, ai] - gt_g[:, ai]) ** 2))
                    )

                if v >= 2 and g["tri"] is not None:
                    tri_g = g["tri"]
                    ba_g  = g["ba"]
                    valid = ~np.isnan(tri_g).any(axis=1)

                    if valid.sum() > 0:
                        self.mse_tri_grp[v].append(self._mse(tri_g[valid], gt_g[valid]))
                        self.mse_ba_grp[v].append( self._mse(ba_g[valid],  gt_g[valid]))
                        ax_tri = self._mse_per_axis(tri_g[valid], gt_g[valid])
                        ax_ba  = self._mse_per_axis(ba_g[valid],  gt_g[valid])
                        for ai, ax in enumerate(self.AXES):
                            self.mse_tri_grp_ax[v][ax].append(float(ax_tri[ai]))
                            self.mse_ba_grp_ax[v][ax].append(float(ax_ba[ai]))
                    else:
                        for store in (self.mse_tri_grp, self.mse_ba_grp):
                            store[v].append(np.nan)
                        for store in (self.mse_tri_grp_ax, self.mse_ba_grp_ax):
                            for ax in self.AXES:
                                store[v][ax].append(np.nan)
                else:
                    self.mse_tri_grp[v].append(np.nan)
                    self.mse_ba_grp[v].append(np.nan)
                    for ax in self.AXES:
                        self.mse_tri_grp_ax[v][ax].append(np.nan)
                        self.mse_ba_grp_ax[v][ax].append(np.nan)

        # Convert to numpy
        self.mse_model_all = np.array(self.mse_model_all)
        self.mse_tri_all   = np.array(self.mse_tri_all)
        self.mse_ba_all    = np.array(self.mse_ba_all)

        for v in self.N_VIS:
            self.pts_per_grp[v]   = np.array(self.pts_per_grp[v])
            self.mse_model_grp[v] = np.array(self.mse_model_grp[v])
            self.mse_tri_grp[v]   = np.array(self.mse_tri_grp[v])
            self.mse_ba_grp[v]    = np.array(self.mse_ba_grp[v])
            for ax in self.AXES:
                self.mse_model_grp_ax[v][ax] = np.array(self.mse_model_grp_ax[v][ax])
                self.mse_tri_grp_ax[v][ax]   = np.array(self.mse_tri_grp_ax[v][ax])
                self.mse_ba_grp_ax[v][ax]    = np.array(self.mse_ba_grp_ax[v][ax])

        self.evaluated = True
        self._print_summary()


    def _print_summary(self):
        sep = "=" * 55
        print()
        print(sep)
        print("  Method       |    Avg MSE (recoverable)")
        print("-" * 52)
        for name, arr in [("Model", self.mse_model_all),
                           ("Tri",   self.mse_tri_all),
                           ("BA",    self.mse_ba_all)]:
            print("  {:<12} |  {:.6f}".format(name, float(np.nanmean(arr))))
        print()
        print("  Avg points per group:")
        for v in self.N_VIS:
            print("    {} view(s): {:.2f} / {} pts".format(v, float(np.mean(self.pts_per_grp[v])), self.K))
        print()
        print("  Per-group avg MSE:")
        print("  {:<8} | {:>10} | {:>10} | {:>10}".format("Group", "Model", "Tri", "BA"))
        print("-" * 45)
        for v in self.N_VIS:
            print("  {} view(s) | {:>10.6f} | {:>10.6f} | {:>10.6f}".format(
                v,
                float(np.nanmean(self.mse_model_grp[v])),
                float(np.nanmean(self.mse_tri_grp[v])),
                float(np.nanmean(self.mse_ba_grp[v])),
            ))
        print(sep)
        print()

    def plot_overall_mse(self, max_frames=None):
        self._check_evaluated()
        num_frames = len(self.mse_model_all) if max_frames is None else min(max_frames, len(self.mse_model_all))
        frames = np.arange(num_frames)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(frames, self.mse_model_all[:num_frames], label="Model",
                color=self.C_MODEL, linewidth=1.5)
        ax.plot(frames, self.mse_tri_all[:num_frames],   label="Triangulation",
                color=self.C_TRI,   linewidth=1.5, linestyle="--")
        ax.plot(frames, self.mse_ba_all[:num_frames],    label="BA",
                color=self.C_BA,    linewidth=1.5, linestyle="-.")
        ax.axhline(np.nanmean(self.mse_model_all),
                   color=self.C_MODEL, linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axhline(np.nanmean(self.mse_tri_all),
                   color=self.C_TRI,   linewidth=0.8, linestyle=":", alpha=0.6)
        ax.axhline(np.nanmean(self.mse_ba_all),
                   color=self.C_BA,    linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_title("Overall MSE over Time (recoverable points only)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_mse_per_group(self, max_frames=None):
        self._check_evaluated()
        titles = {1: "1 View (Model only)", 2: "2 Views", 3: "3 Views"}
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("MSE per Visibility Group", fontsize=13)

        for ax, v in zip(axes, self.N_VIS):
            num_frames = len(self.mse_model_grp[v]) if max_frames is None else min(max_frames, len(self.mse_model_grp[v]))
            frames = np.arange(num_frames)
            ax.plot(frames, self.mse_model_grp[v][:num_frames], label="Model",
                    color=self.C_MODEL, linewidth=1.5)
            if v >= 2:
                ax.plot(frames, self.mse_tri_grp[v][:num_frames], label="Tri",
                        color=self.C_TRI, linewidth=1.5, linestyle="--")
                ax.plot(frames, self.mse_ba_grp[v][:num_frames],  label="BA",
                        color=self.C_BA,  linewidth=1.5, linestyle="-.")
            ax.set_title(titles[v])
            ax.set_xlabel("Frame")
            ax.set_ylabel("MSE" if v == 1 else "")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_mse_per_group_per_axis(self, max_frames=None):
        self._check_evaluated()
        ax_colors = {"X": "#e74c3c", "Y": "#27ae60", "Z": "#2980b9"}
        grp_titles = {1: "1 View", 2: "2 Views", 3: "3 Views"}
        methods = [
            ("Model", self.mse_model_grp_ax),
            ("Tri",   self.mse_tri_grp_ax),
            ("BA",    self.mse_ba_grp_ax),
        ]

        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle("MSE per Group per Axis", fontsize=13)

        for row, (mname, mstore) in enumerate(methods):
            for col, v in enumerate(self.N_VIS):
                ax = axes[row][col]
                if row == 0:
                    ax.set_title(grp_titles[v], fontsize=11)
                if col == 0:
                    ax.set_ylabel(mname + "\nMSE", fontsize=10)
                if row == 2:
                    ax.set_xlabel("Frame")

                if v == 1 and row > 0:
                    ax.set_facecolor("#f0f0f0")
                    ax.text(0.5, 0.5, "N/A (1 view)", transform=ax.transAxes,
                            ha="center", va="center", fontsize=11, color="grey")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                num_frames = len(self.mse_model_grp[v]) if max_frames is None else min(max_frames, len(self.mse_model_grp[v]))
                frames = np.arange(num_frames)
                for axis_name in self.AXES:
                    ax.plot(frames, mstore[v][axis_name][:num_frames],
                            label=axis_name,
                            color=ax_colors[axis_name], linewidth=1.3)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    def _check_evaluated(self):
        if not self.evaluated:
            raise RuntimeError("Call .evaluate() first.")