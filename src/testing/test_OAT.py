from __future__ import annotations
from OAT.models import Global3DKeypointModel
from utils.geometry import linear_interpolate_keypoints, cubic_spline_interpolate_keypoints
import torch
from torch.utils.data import Dataset
import pandas as pd
from OAT.dataset import DataDrivenNormalization
from tqdm import tqdm
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from typing import Optional, Sequence
from scipy.ndimage import binary_dilation


class TestDataset(Dataset):
    def __init__(self, file_path, mask_ratio=0.3, normalize=False):
        dataset = pd.read_csv(file_path)
        self.body_parts = {part: idx for idx, part in enumerate(dataset['part'].unique())}
        dataset['p_idx'] = dataset["part"].map(self.body_parts)
        self.dataset = (
            dataset[["time"] + ["p_idx"] + dataset.columns.drop(["time", "p_idx"]).tolist()]
            .drop(columns=['part'])
        )
        self.part_count = len(self.body_parts.keys())

        self.rigid_coords = torch.tensor(
            self.dataset.iloc[:, 2:5].to_numpy(), dtype=torch.float32
        ).view(-1, self.part_count, 3)
        self.deformable_coords = torch.tensor(
            self.dataset.drop(columns=['x_r', 'y_r', 'z_r']).iloc[:, 2:5].to_numpy(),
            dtype=torch.float32,
        ).view(-1, self.part_count, 3)

        self.rigid_normalizer      = DataDrivenNormalization()
        self.deformable_normalizer = DataDrivenNormalization()

        self.outlier_mask = self.flag_outlier_frames(reference_edge=(8, 9))
        structure = np.ones(2 * 3 + 1, dtype=bool)  # e.g. [T,T,T,T,T]
        self.outlier_mask = binary_dilation(self.outlier_mask.squeeze(), structure=structure)
        self.outlier_mask[-3:] = True

        if normalize:
            self.d_centers = self.deformable_coords.mean(dim=1, keepdim=True)
            self.r_centers = self.rigid_coords.mean(dim=1, keepdim=True)
            d_coords_centered = self.deformable_coords - self.d_centers
            r_coords_centered = self.rigid_coords      - self.r_centers

            self.rigid_normalizer.fit(r_coords_centered)
            self.deformable_normalizer.fit(d_coords_centered)
            self.normalized_rigid_coords      = self.rigid_normalizer.normalize(r_coords_centered)
            self.normalized_deformable_coords = self.deformable_normalizer.normalize(d_coords_centered)

        self.set_masks(mask_ratio)

        # Pre-compute baseline interpolations over the full temporal sequence
        self._precompute_baselines()

    # ── mask helpers ─────────────────────────────────────────────────────────

    def set_masks(self, mask_ratio):
        T, P, _ = self.deformable_coords.shape
        num_masked_parts = int(P * mask_ratio)
        self.masks = torch.ones((T, P), dtype=torch.float32)
        for t in range(T):
            masked_parts = torch.randperm(P)[:num_masked_parts]
            self.masks[t, masked_parts] = 0.0

    # ── outlier detection ────────────────────────────────────────────────────

    def flag_outlier_frames(self, reference_edge=(8, 9), thresh_deg=70.0):
        p_idx, c_idx = reference_edge
        bone_vec = (
            self.rigid_coords[:, c_idx, :] - self.rigid_coords[:, p_idx, :]
        )
        norms    = bone_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit_vec = bone_vec / norms
        cos_sim  = (unit_vec[1:] * unit_vec[:-1]).sum(dim=-1).clamp(-1.0, 1.0)
        raw_angles = torch.acos(cos_sim) * (180.0 / torch.pi)
        raw_angles = torch.cat([torch.zeros(1), raw_angles])
        return raw_angles > thresh_deg

    # ── baseline pre-computation ─────────────────────────────────────────────

    def _precompute_baselines(self):
        """
        Compute linear and cubic-spline interpolations once over the full
        (T, N, 3) deformable coordinate array using the per-frame masks.

        Results are stored as torch tensors of shape (T, N, 3) in
        denormalised / original coordinate space so they can be compared
        directly against deformable_coords in the evaluator.
        """
        # masks: (T, N) — 1.0 = visible, 0.0 = occluded → bool visible mask
        mask_np = self.masks.numpy().astype(bool)   # (T, N)  True = visible
        coords_np = self.deformable_coords.numpy()  # (T, N, 3)

        linear_np = linear_interpolate_keypoints(coords_np, mask_np[..., np.newaxis],self.outlier_mask)
        spline_np = cubic_spline_interpolate_keypoints(coords_np, mask_np[..., np.newaxis], outlier_frames=self.outlier_mask)

        self.linear_baseline = torch.tensor(linear_np, dtype=torch.float32)   # (T, N, 3)
        self.spline_baseline = torch.tensor(spline_np, dtype=torch.float32)   # (T, N, 3)

    # ── dataset protocol ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.deformable_coords)

    def __getitem__(self, idx):
        return (
            self.normalized_deformable_coords[idx],
            self.normalized_rigid_coords[idx],
            self.masks[idx],
            self.deformable_coords[idx],
            self.rigid_coords[idx],
            self.outlier_mask[idx],
        )

    # ── denormalisation helpers ───────────────────────────────────────────────

    def denormalize_deformable(self, coords_norm, idx):
        coords_denorm = self.deformable_normalizer.denormalize(coords_norm)
        return coords_denorm + self.d_centers[idx]

    def denormalize_rigid(self, coords_norm, idx):
        coords_denorm = self.rigid_normalizer.denormalize(coords_norm)
        return coords_denorm + self.r_centers[idx]

class ModelWrapper(torch.nn.Module):
    def __init__(self, config, checkpoint, aligner):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.model = Global3DKeypointModel(config, numkeypoints=10)
        self.load_checkpoint(checkpoint, device='cuda')
        self.aligner = aligner
    
    def load_checkpoint(self, checkpoint_path, device):
        print(f"\n{'='*60}")
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"{'='*60}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict']) if 'model_state_dict' in checkpoint else self.model.load_state_dict(checkpoint)
        print(f"✓ Loaded model weights")


    def forward(self, deformable_coords, rigid_coords, masks):
        prediction =  self.model(deformable_coords, masks)['coordinates'].detach().cpu()
        return prediction

class Evaluator:
    """
    Three evaluation modes:
      1. evaluate_models(dataset_idx)       — compare multiple models on one dataset
      2. evaluate_datasets(method_name)     — compare one model across all datasets
      3. evaluate_with_baselines(method_name, dataset_idx)
                                            — compare one model vs linear & spline
                                              baselines on one dataset
    """

    THRESHOLDS = (0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0)
    DIM_NAMES  = ("X", "Y", "Z")

    def __init__(self, datasets, methods, dataset_names, joint_names):
        self.datasets      = datasets
        self.methods       = methods
        self.dataset_names = dataset_names or [f"Dataset_{i}" for i in range(len(datasets))]
        self.joint_names   = joint_names
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"

        self.results      = {}
        self.predictions  = {}
        self._gt_frames   = None
        self._mask_frames = None

    # ── core runner ───────────────────────────────────────────────────────────

    def _run(self, model_wrapper, dataset):
        per_frame_errs = []
        per_joint_errs = []
        per_dim_errs   = []
        pred_list      = []
        gt_list        = []
        mask_list      = []

        for idx, (deformable_coords_norm, rigid_coords_norm, masks,
                  deformable_coords, __, __) in tqdm(enumerate(dataset)):

            dnorm = deformable_coords_norm.unsqueeze(0).to(self.device)
            rnorm = rigid_coords_norm.unsqueeze(0).to(self.device)
            m     = masks.unsqueeze(0).unsqueeze(-1).bool().to(self.device)
            pred  = model_wrapper(dnorm, rnorm, m)
            pred  = dataset.denormalize_deformable(pred, idx)

            pred = pred.squeeze(0).detach().cpu()
            gt   = deformable_coords.squeeze(0)
            vis  = masks.bool()
            if vis.dim() == 2:
                vis = vis.squeeze(-1)

            pred_list.append(pred.numpy())
            gt_list.append(gt.numpy())
            mask_list.append(vis.numpy())

            joint_err = torch.norm(pred - gt, dim=-1)
            per_frame_errs.append(joint_err[~vis].mean().item())
            per_joint_errs.append(joint_err.numpy())
            per_dim_errs.append((pred - gt)[~vis].abs().mean(dim=0).numpy())

        pf  = np.array(per_frame_errs)
        pj  = np.stack(per_joint_errs, axis=0)
        pd_ = np.stack(per_dim_errs,   axis=0)
        masked_errors = np.ma.masked_equal(pj, 0)

        metrics = {
            "mean_error"       : float(pf.mean()),
            "median_error"     : float(np.median(pf)),
            "std_error"        : float(pf.std()),
            "max_error"        : float(pf.max()),
            "per_joint_mean"   : masked_errors.mean(axis=0).data,
            "per_joint_std"    : masked_errors.std(axis=0).data,
            "per_joint_median" : np.median(pj, axis=0),
            "per_joint_max"    : masked_errors.max(axis=0).data,
            "per_joint_series" : pj,
            "per_dim_mae"      : pd_.mean(axis=0),
            "within_thresh"    : {
                thr: float(np.mean(pf < thr) * 100)
                for thr in self.THRESHOLDS
            },
            "per_frame_error"  : pf,
        }
        return metrics, np.stack(pred_list), np.stack(gt_list), np.stack(mask_list)

    # ── baseline runner (uses pre-computed arrays — no iteration) ─────────────

    def _run_baseline(self, dataset, kind: str):
        """
        Compute evaluation metrics for a pre-computed interpolation baseline.

        Args:
            dataset: TestDataset instance (must have linear_baseline / spline_baseline)
            kind:    "linear" | "spline"
        """
        baseline = (
            dataset.linear_baseline if kind == "linear" else dataset.spline_baseline
        )   # (T, N, 3)

        gt      = dataset.deformable_coords          # (T, N, 3)
        masks   = dataset.masks.bool()               # (T, N)  True = visible

        T, N, _ = gt.shape

        per_frame_errs = []
        per_joint_errs = []
        per_dim_errs   = []

        for t in range(T):
            occ = ~masks[t]                          # (N,) occluded joints
            if occ.sum() == 0:
                per_frame_errs.append(0.0)
                per_joint_errs.append(np.zeros(N))
                per_dim_errs.append(np.zeros(3))
                continue

            err_vec  = baseline[t] - gt[t]          # (N, 3)
            joint_l2 = torch.norm(err_vec, dim=-1)  # (N,)

            per_frame_errs.append(joint_l2[occ].mean().item())
            per_joint_errs.append(joint_l2.numpy())
            per_dim_errs.append(err_vec[occ].abs().mean(dim=0).numpy())

        pf  = np.array(per_frame_errs)
        pj  = np.stack(per_joint_errs, axis=0)
        pd_ = np.stack(per_dim_errs,   axis=0)
        masked_errors = np.ma.masked_equal(pj, 0)

        return {
            "mean_error"       : float(pf.mean()),
            "median_error"     : float(np.median(pf)),
            "std_error"        : float(pf.std()),
            "max_error"        : float(pf.max()),
            "per_joint_mean"   : masked_errors.mean(axis=0).data,
            "per_joint_std"    : masked_errors.std(axis=0).data,
            "per_joint_median" : np.median(pj, axis=0),
            "per_joint_max"    : masked_errors.max(axis=0).data,
            "per_joint_series" : pj,
            "per_dim_mae"      : pd_.mean(axis=0),
            "within_thresh"    : {
                thr: float(np.mean(pf < thr) * 100)
                for thr in self.THRESHOLDS
            },
            "per_frame_error"  : pf,
        }

    # ── public evaluation methods ─────────────────────────────────────────────

    def evaluate_models(self, dataset_idx=0):
        """Compare all methods on datasets[dataset_idx]."""
        if dataset_idx >= len(self.datasets):
            raise IndexError(
                f"dataset_idx={dataset_idx} out of range ({len(self.datasets)} datasets)."
            )
        dataset       = self.datasets[dataset_idx]
        dataset_label = self.dataset_names[dataset_idx]

        self.results     = {}
        self.predictions = {}
        self._gt_frames  = None
        self._mask_frames = None

        with torch.no_grad():
            for method_name, model_wrapper in self.methods.items():
                print(f"\n{'='*60}\n  [{dataset_label}]  Evaluating: {method_name}\n{'='*60}")
                metrics, preds, gt_arr, mask_arr = self._run(model_wrapper, dataset)
                self.results[method_name]     = metrics
                self.predictions[method_name] = preds
                self._print_summary(method_name, metrics)
                if self._gt_frames is None:
                    self._gt_frames   = gt_arr
                    self._mask_frames = mask_arr

        return self.results

    def evaluate_datasets(self, method_name):
        """Run methods[method_name] across all datasets."""
        if method_name not in self.methods:
            raise KeyError(f"'{method_name}' not found. Available: {list(self.methods.keys())}")
        model_wrapper = self.methods[method_name]

        self.results     = {}
        self.predictions = {}
        self._gt_frames  = None
        self._mask_frames = None

        with torch.no_grad():
            for dataset_label, dataset in zip(self.dataset_names, self.datasets):
                print(f"\n{'='*60}\n  [{method_name}]  Evaluating dataset: {dataset_label}\n{'='*60}")
                metrics, preds, gt_arr, mask_arr = self._run(model_wrapper, dataset)
                self.results[dataset_label]     = metrics
                self.predictions[dataset_label] = preds
                self._print_summary(dataset_label, metrics)
                if self._gt_frames is None:
                    self._gt_frames   = gt_arr
                    self._mask_frames = mask_arr

        return self.results

    def evaluate_with_baselines(self, method_name, dataset_idx=0):
        """
        Compare a single method against the linear and cubic-spline baselines
        on datasets[dataset_idx].

        Populates self.results with keys:
            method_name, "Linear Interp", "Cubic Spline"

        Returns self.results.
        """
        if method_name not in self.methods:
            raise KeyError(f"'{method_name}' not found. Available: {list(self.methods.keys())}")
        if dataset_idx >= len(self.datasets):
            raise IndexError(
                f"dataset_idx={dataset_idx} out of range ({len(self.datasets)} datasets)."
            )

        dataset       = self.datasets[dataset_idx]
        dataset_label = self.dataset_names[dataset_idx]

        self.results     = {}
        self.predictions = {}
        self._gt_frames  = None
        self._mask_frames = None

        # --- learned model ---
        print(f"\n{'='*60}\n  [{dataset_label}]  Evaluating model: {method_name}\n{'='*60}")
        with torch.no_grad():
            metrics, preds, gt_arr, mask_arr = self._run(
                self.methods[method_name], dataset
            )
        self.results[method_name]     = metrics
        self.predictions[method_name] = preds
        self._gt_frames   = gt_arr
        self._mask_frames = mask_arr
        self._print_summary(method_name, metrics)

        # --- linear baseline (pre-computed in __init__) ---
        print(f"\n{'='*60}\n  [{dataset_label}]  Evaluating baseline: Linear Interp\n{'='*60}")
        lin_metrics = self._run_baseline(dataset, kind="linear")
        self.results["Linear Interp"] = lin_metrics
        self._print_summary("Linear Interp", lin_metrics)

        # --- spline baseline (pre-computed in __init__) ---
        print(f"\n{'='*60}\n  [{dataset_label}]  Evaluating baseline: Cubic Spline\n{'='*60}")
        spl_metrics = self._run_baseline(dataset, kind="spline")
        self.results["Cubic Spline"] = spl_metrics
        self._print_summary("Cubic Spline", spl_metrics)

        return self.results

    # ── summary printer ───────────────────────────────────────────────────────

    def _print_summary(self, name, m):
        J      = len(m["per_joint_mean"])
        jnames = self.joint_names or [f"J{i}" for i in range(J)]
        print(f"\n── Overall ({name}) ──────────────────────────────────")
        print(f"  Mean   : {m['mean_error']:.4f}")
        print(f"  Median : {m['median_error']:.4f}")
        print(f"  Std    : {m['std_error']:.4f}")
        print(f"  Max    : {m['max_error']:.4f}")
        print(f"\n── Within-threshold (%) ─────────────────────────────")
        for thr, pct in m["within_thresh"].items():
            print(f"  < {thr:.3f} : {pct:.1f}%")
        print(f"\n── Per-dimension MAE (X / Y / Z) ─────────────────────")
        for dim, err in zip(self.DIM_NAMES, m["per_dim_mae"]):
            print(f"  {dim}: {err:.4f}")
        print(f"\n── Per-joint mean ± std (worst→best) ────────────────")
        order = np.argsort(m["per_joint_mean"])[::-1]
        for i in order:
            print(
                f"  {jnames[i]:>12s}  "
                f"mean={m['per_joint_mean'][i]:.4f}  "
                f"std={m['per_joint_std'][i]:.4f}  "
                f"max={m['per_joint_max'][i]:.4f}"
            )

    # ── existing plot methods (unchanged) ────────────────────────────────────

    def plot_results(self, save_html=None):
        """
        5-panel dashboard:
          1. Overall error (mean ± std) per method
          2. Per-dimension MAE (X/Y/Z)
          3. Within-threshold curves
          4. Per-frame error time series
          5. Masked-joint mean error per method
        """
        if not self.results:
            raise RuntimeError("Call evaluate() first.")

        COLORS = ["#4f98a3", "#da7101", "#a86fdf", "#6daa45",
                  "#dd6974", "#e8af34", "#5591c7", "#d163a7"]

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Overall Error (Mean ± Std)", "Per-Dimension MAE",
                "Within-Threshold (%)", "Masked-Joint Error per Method",
                "Per-Frame Error (Time Series)", "",
            ],
            vertical_spacing=0.12, horizontal_spacing=0.10,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy", "colspan": 2}, None],
                [{"type": "xy", "colspan": 2}, None],
            ],
        )

        for idx, (mn, m) in enumerate(self.results.items()):
            c = COLORS[idx % len(COLORS)]
            fig.add_trace(go.Bar(
                x=[mn], y=[m["mean_error"]],
                error_y=dict(type="data", array=[m["std_error"]], visible=True),
                name=mn, marker_color=c, showlegend=True,
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                name=mn, x=["X", "Y", "Z"], y=m["per_dim_mae"].tolist(),
                marker_color=c, showlegend=False,
            ), row=1, col=2)

        thresholds = list(next(iter(self.results.values()))["within_thresh"].keys())
        for idx, (mn, m) in enumerate(self.results.items()):
            pcts = [m["within_thresh"][t] for t in thresholds]
            fig.add_trace(go.Scatter(
                x=[str(t) for t in thresholds], y=pcts,
                mode="lines+markers", name=mn,
                line=dict(color=COLORS[idx % len(COLORS)], width=2),
                marker=dict(size=8), showlegend=False,
            ), row=2, col=1)

        for idx, (mn, m) in enumerate(self.results.items()):
            fig.add_trace(go.Scatter(
                y=m["per_frame_error"], mode="lines", name=mn,
                line=dict(color=COLORS[idx % len(COLORS)], width=1),
                showlegend=False, opacity=0.85,
            ), row=3, col=1)

        fig.update_layout(
            title=dict(text="Evaluation Dashboard", font=dict(size=22)),
            height=1050, template="plotly_dark", barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter, sans-serif", size=12),
        )
        if save_html:
            fig.write_html(save_html)
            print(f"Dashboard saved → {save_html}")
        return fig

    def plot_per_joint(self, save_html=None):
        """
        One subplot per joint showing error-over-time for every method,
        plus a horizontal dashed line at each method's per-joint mean.
        """
        if not self.results:
            raise RuntimeError("Call evaluate() first.")

        method_names = list(self.results.keys())
        J      = next(iter(self.results.values()))["per_joint_series"].shape[1]
        jnames = self.joint_names or [f"J{i}" for i in range(J)]
        COLORS = ["#4f98a3", "#da7101", "#a86fdf", "#6daa45",
                  "#dd6974", "#e8af34", "#5591c7", "#d163a7"]

        ncols = min(3, J)
        nrows = int(np.ceil(J / ncols))

        fig = make_subplots(
            rows=nrows, cols=ncols, subplot_titles=jnames,
            shared_xaxes=True, vertical_spacing=0.08, horizontal_spacing=0.06,
        )

        traces_per_method = 2 * J
        n_methods = len(method_names)

        for m_idx, mn in enumerate(method_names):
            series  = self.results[mn]["per_joint_series"]
            means   = self.results[mn]["per_joint_mean"]
            color   = COLORS[m_idx % len(COLORS)]
            visible = (m_idx == 0)
            T       = series.shape[0]

            for j in range(J):
                row = j // ncols + 1
                col = j %  ncols + 1
                fig.add_trace(go.Scatter(
                    y=series[:, j], mode="lines", name=mn,
                    legendgroup=mn, showlegend=(j == 0),
                    line=dict(color=color, width=1), opacity=0.8, visible=visible,
                ), row=row, col=col)
                fig.add_trace(go.Scatter(
                    x=[0, T - 1], y=[means[j], means[j]], mode="lines",
                    name=f"{mn} mean", legendgroup=mn, showlegend=False,
                    line=dict(color=color, width=1.5, dash="dash"), visible=visible,
                ), row=row, col=col)

        buttons = []
        for m_idx, mn in enumerate(method_names):
            vis_flags = []
            for mi in range(n_methods):
                vis_flags += [mi == m_idx] * traces_per_method
            buttons.append(dict(
                label=mn, method="update",
                args=[{"visible": vis_flags},
                      {"title": f"Per-Joint Error Over Time — {mn}"}],
            ))

        fig.update_layout(
            title=f"Per-Joint Error Over Time — {method_names[0]}",
            height=280 * nrows, template="plotly_dark",
            font=dict(family="Inter, sans-serif", size=11),
            updatemenus=[dict(
                type="dropdown", x=1.0, y=1.12, xanchor="right",
                buttons=buttons, bgcolor="#1c1b19",
                font=dict(color="white"), bordercolor="#393836",
            )],
        )
        fig.update_yaxes(title_text="L2 Error")
        fig.update_xaxes(title_text="Frame", row=nrows)

        if save_html:
            fig.write_html(save_html)
        return fig

    # ── NEW: baseline comparison plot ─────────────────────────────────────────

    def plot_baseline_comparison(self, save_html=None):
        """
        4-panel plot comparing one method vs Linear Interp and Cubic Spline:

          Panel 1 — MSE / mean L2 bar chart (method + 2 baselines)
          Panel 2 — % of frames within error thresholds (CDF-style curves)
          Panel 3 — Per-frame error over time (all three on same axes)
          Panel 4 — Per-joint mean error: method + baselines on the same subplot
                    (grouped bars per joint, one group per method)

        Expected self.results keys: method name, "Linear Interp", "Cubic Spline"
        (as produced by evaluate_with_baselines()).
        """
        if not self.results:
            raise RuntimeError("Call evaluate_with_baselines() first.")

        names  = list(self.results.keys())
        J      = next(iter(self.results.values()))["per_joint_series"].shape[1]
        jnames = self.joint_names or [f"J{i}" for i in range(J)]

        # Colour convention: method = teal, linear = orange, spline = purple
        # Falls back to a safe palette for any extra keys.
        PALETTE = ["#4f98a3", "#da7101", "#a86fdf", "#6daa45",
                   "#dd6974", "#e8af34", "#5591c7", "#d163a7"]
        colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Mean L2 Error",
                "% Frames Within Threshold",
                "Per-Frame Error Over Time",
                "Per-Joint Mean Error",
            ],
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        # ── Panel 1: mean ± std bars ──────────────────────────────────────────
        for name in names:
            m = self.results[name]
            fig.add_trace(go.Bar(
                x=[name],
                y=[m["mean_error"]],
                error_y=dict(type="data", array=[m["std_error"]], visible=True),
                name=name,
                marker_color=colors[name],
                showlegend=True,
            ), row=1, col=1)

        # ── Panel 2: within-threshold CDF curves ─────────────────────────────
        thresholds = list(next(iter(self.results.values()))["within_thresh"].keys())
        for name in names:
            pcts = [self.results[name]["within_thresh"][t] for t in thresholds]
            fig.add_trace(go.Scatter(
                x=[str(t) for t in thresholds],
                y=pcts,
                mode="lines+markers",
                name=name,
                line=dict(color=colors[name], width=2),
                marker=dict(size=7),
                showlegend=False,
            ), row=1, col=2)

        # ── Panel 3: per-frame error time series ─────────────────────────────
        for name in names:
            pf = self.results[name]["per_frame_error"]
            fig.add_trace(go.Scatter(
                y=pf,
                mode="lines",
                name=name,
                line=dict(color=colors[name], width=1.2),
                opacity=0.85,
                showlegend=False,
            ), row=2, col=1)

        # ── Panel 4: per-joint grouped bars ──────────────────────────────────
        for name in names:
            m = self.results[name]
            fig.add_trace(go.Bar(
                x=jnames,
                y=m["per_joint_mean"].tolist(),
                error_y=dict(
                    type="data",
                    array=m["per_joint_std"].tolist(),
                    visible=True,
                ),
                name=name,
                marker_color=colors[name],
                showlegend=False,
            ), row=2, col=2)

        fig.update_layout(
            title=dict(
                text="Baseline Comparison: Model vs Interpolation Baselines",
                font=dict(size=20),
            ),
            height=800,
            template="plotly_dark",
            barmode="group",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.04,
                xanchor="right", x=1,
            ),
            font=dict(family="Inter, sans-serif", size=12),
        )
        fig.update_xaxes(title_text="Method",       row=1, col=1)
        fig.update_yaxes(title_text="Mean L2",      row=1, col=1)
        fig.update_xaxes(title_text="Threshold",    row=1, col=2)
        fig.update_yaxes(title_text="% Frames",     row=1, col=2)
        fig.update_xaxes(title_text="Frame",        row=2, col=1)
        fig.update_yaxes(title_text="L2 Error",     row=2, col=1)
        fig.update_xaxes(title_text="Joint",        row=2, col=2, tickangle=-30)
        fig.update_yaxes(title_text="Mean L2",      row=2, col=2)

        if save_html:
            fig.write_html(save_html)
            print(f"Baseline comparison saved → {save_html}")
        return fig

    # ------------------------------------------------------------------
    def animate(
        self,
        method: str | None = None,
        dataset_idx: int = 0,
        frame_idx: int = 0,
        n_frames: int = 100,
        edges: list | None = None,
        frame_duration: int = 200,
        save_html: str | None = None,
    ) -> go.Figure:
        """
        Single 3D scene animation for one method at a time, with the
        linear interpolation baseline shown alongside the model prediction.

        Point sets per frame:
          • GT visible          (teal)
          • GT masked           (gray, 50% opacity)
          • Predicted masked    (orange / method color)
          • Linear baseline     (green)
          • Error lines         (crimson, GT masked ↔ predicted)
        """
        if self._gt_frames is None:
            raise RuntimeError("Call evaluate() before animate().")

        method_names = list(self.predictions.keys())
        selected     = [method] if method else method_names

        # Pull pre-computed linear baseline from the dataset if available
        dataset       = self.datasets[dataset_idx]
        linear_baseline = (
            dataset.linear_baseline.numpy()
            if hasattr(dataset, "linear_baseline") else None
        )

        return _animate_single_scene(
            gt_xyz          = self._gt_frames,
            predictions     = {mn: self.predictions[mn] for mn in selected},
            mask            = self._mask_frames,
            linear_baseline = linear_baseline,
            joint_names     = self.joint_names,
            frame_idx       = frame_idx,
            n_frames        = n_frames,
            edges           = edges,
            frame_duration  = frame_duration,
            save_html       = save_html,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level animation helper
# ──────────────────────────────────────────────────────────────────────────────

def _animate_single_scene(
    gt_xyz: np.ndarray,
    predictions: dict,
    mask: np.ndarray,
    linear_baseline: np.ndarray | None = None,   # (T, N, 3) — pre-computed linear interp
    joint_names: Optional[Sequence[str]] = None,
    frame_idx: int = 0,
    n_frames: int = 100,
    edges: Optional[list] = None,
    frame_duration: int = 200,                   # ~5 fps
    marker_size: int = 6,
    save_html: Optional[str] = None,
) -> go.Figure:
    """
    Single-scene 3D player.

    Colors
    ------
    Teal    (#4f98a3) — GT visible joints
    Gray    (#6b7280) — GT masked joints          (opacity 0.5)
    Orange  (#da7101) — Predicted masked          (model)
    Green   (#6daa45) — Linear baseline masked    (optional)
    Crimson lines     — Error: GT masked ↔ predicted masked
    """
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32)
    mask   = np.asarray(mask,   dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0]

    T_full, J, _ = gt_xyz.shape
    jnames  = joint_names or [f"J{i}" for i in range(J)]
    t_start = int(frame_idx)
    t_end   = min(t_start + n_frames, T_full)
    T       = t_end - t_start

    gt_clip   = gt_xyz[t_start:t_end]
    mask_clip = mask[t_start:t_end]

    method_names = list(predictions.keys())
    COLORS_METHOD = ["#da7101", "#a86fdf", "#6daa45", "#dd6974", "#e8af34", "#5591c7"]

    pred_clips = {
        mn: np.asarray(predictions[mn], dtype=np.float32)[t_start:t_end]
        for mn in method_names
    }

    lin_clip = (
        np.asarray(linear_baseline, dtype=np.float32)[t_start:t_end]
        if linear_baseline is not None else None
    )

    # Stable axis limits
    all_pts = np.concatenate(
        [gt_clip.reshape(-1, 3)] +
        [v.reshape(-1, 3) for v in pred_clips.values()] +
        ([lin_clip.reshape(-1, 3)] if lin_clip is not None else []),
        axis=0,
    )
    mins   = all_pts.min(axis=0)
    maxs   = all_pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = np.max(maxs - mins) / 2 + 1e-6
    ax_range = lambda ax: [center[ax] - radius, center[ax] + radius]

    GT_VIS_COLOR  = "#4f98a3"
    GT_MASK_COLOR = "#6b7280"
    LIN_COLOR     = "#6daa45"
    ERR_COLOR     = "crimson"

    def _skeleton(pts, color="rgba(180,180,180,0.35)", width=2):
        if not edges:
            return []
        xe, ye, ze = [], [], []
        for i, j in edges:
            xe += [pts[i, 0], pts[j, 0], None]
            ye += [pts[i, 1], pts[j, 1], None]
            ze += [pts[i, 2], pts[j, 2], None]
        return [go.Scatter3d(x=xe, y=ye, z=ze, mode="lines",
                             line=dict(color=color, width=width),
                             showlegend=False, hoverinfo="skip")]

    def _points(pts, idx_mask, color, label, size=marker_size,
                opacity=1.0, show_legend=True):
        if not idx_mask.any():
            return [go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                 showlegend=False, hoverinfo="skip")]
        sel = pts[idx_mask]
        return [go.Scatter3d(
            x=sel[:, 0], y=sel[:, 1], z=sel[:, 2],
            mode="markers+text",
            text=[jnames[i] for i in np.where(idx_mask)[0]],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=size, color=color, opacity=opacity),
            name=label,
            showlegend=show_legend,
            legendgroup=label,
        )]

    def _error_lines(gt_pts, pred_pts, idx_mask):
        if not idx_mask.any():
            return [go.Scatter3d(x=[], y=[], z=[], mode="lines",
                                 showlegend=False, hoverinfo="skip")]
        xe, ye, ze = [], [], []
        for j in np.where(idx_mask)[0]:
            xe += [gt_pts[j, 0], pred_pts[j, 0], None]
            ye += [gt_pts[j, 1], pred_pts[j, 1], None]
            ze += [gt_pts[j, 2], pred_pts[j, 2], None]
        return [go.Scatter3d(x=xe, y=ye, z=ze, mode="lines",
                             line=dict(color=ERR_COLOR, width=2),
                             opacity=0.7, showlegend=False, hoverinfo="skip")]

    # ── Fixed 6-trace structure per frame ────────────────────────────────────
    #   0  skeleton (GT)
    #   1  GT visible
    #   2  GT masked          (opacity 0.5)
    #   3  predicted masked   (model, orange)
    #   4  linear baseline    (green)  — empty scatter when no baseline
    #   5  error lines
    N_TRACES = 6

    def _frame_traces(t, method_name):
        vis   = mask_clip[t]   # (J,) True = visible
        invis = ~vis
        gt    = gt_clip[t]
        pred  = pred_clips[method_name][t]
        mc    = COLORS_METHOD[method_names.index(method_name) % len(COLORS_METHOD)]

        traces = []
        traces += _skeleton(gt)                                               # 0
        traces += _points(gt,   vis,   GT_VIS_COLOR, "GT visible")            # 1
        traces += _points(gt,   invis, GT_MASK_COLOR, "GT masked",            # 2
                          size=marker_size - 1, opacity=0.5)
        traces += _points(pred, invis, mc, f"Pred ({method_name})")           # 3

        # Trace 4: linear baseline (green) or empty placeholder
        if lin_clip is not None:
            lin = lin_clip[t]
            traces += _points(lin, invis, LIN_COLOR, "Linear baseline",
                              size=marker_size - 1, opacity=0.85)             # 4
        else:
            traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                       showlegend=False, hoverinfo="skip"))   # 4 (empty)

        traces += _error_lines(gt, pred, invis)                               # 5

        # Safety pad to exactly N_TRACES
        while len(traces) < N_TRACES:
            traces.append(go.Scatter3d(x=[], y=[], z=[], mode="markers",
                                       showlegend=False, hoverinfo="skip"))
        return traces[:N_TRACES]

    # Initial figure
    init_method  = method_names[0]
    init_traces  = _frame_traces(0, init_method)
    fig = go.Figure(data=init_traces)

    # All frames
    all_frames = []
    for mn in method_names:
        for t in range(T):
            all_frames.append(go.Frame(
                data=_frame_traces(t, mn),
                name=f"{mn}_{t}",
                traces=list(range(N_TRACES)),
            ))
    fig.frames = all_frames

    def _slider_steps(mn):
        return [
            {
                "args": [
                    [f"{mn}_{t}"],
                    {"frame": {"duration": 0, "redraw": True},
                     "mode": "immediate", "transition": {"duration": 0}},
                ],
                "label": str(t + t_start),
                "method": "animate",
            }
            for t in range(T)
        ]

    method_buttons = [
        dict(
            label=mn, method="animate",
            args=[
                [f"{mn}_{t}" for t in range(T)],
                {"frame": {"duration": 0, "redraw": True},
                 "mode": "immediate", "transition": {"duration": 0}},
            ],
        )
        for mn in method_names
    ]

    scene = dict(
        xaxis=dict(range=ax_range(0), showticklabels=False, title="X"),
        yaxis=dict(range=ax_range(1), showticklabels=False, title="Y"),
        zaxis=dict(range=ax_range(2), showticklabels=False, title="Z"),
        aspectmode="cube",
        bgcolor="rgba(20,20,28,1)",
    )

    fig.update_layout(
        scene=scene,
        height=680,
        paper_bgcolor="rgba(18,18,26,1)",
        font=dict(color="white", family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=70, b=100),
        legend=dict(
            x=0.01, y=0.99, bgcolor="rgba(30,30,40,0.7)",
            bordercolor="#393836", borderwidth=1,
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0, "y": -0.13,
                "xanchor": "left", "yanchor": "top",
                "buttons": [
                    {
                        "label": "▶  Play",
                        "method": "animate",
                        "args": [
                            [f"{init_method}_{t}" for t in range(T)],
                            {"frame": {"duration": frame_duration, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 0}},
                        ],
                    },
                    {
                        "label": "⏸  Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                    },
                ],
            },
            {
                "type": "dropdown",
                "x": 1.0, "y": 1.08,
                "xanchor": "right",
                "buttons": method_buttons,
                "bgcolor": "#1c1b19",
                "font": {"color": "white"},
                "bordercolor": "#393836",
                "active": 0,
            },
        ],
        sliders=[{
            "active": 0,
            "pad": {"t": 60},
            "currentvalue": {
                "prefix": f"Frame (offset {t_start}): ",
                "font": {"color": "white"},
            },
            "steps": _slider_steps(init_method),
            "font": {"color": "white"},
        }],
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")
        print(f"Animation saved → {save_html}")

    return fig