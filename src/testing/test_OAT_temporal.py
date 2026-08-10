"""
evaluate.py — Full test-time evaluation of TemporalModel vs. Linear-Interpolation baseline.
"""
from networkx.algorithms import structuralholes
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from utils.dataset import TemporalDataset
from utils.masker import Masker
from OAT.dataset import TemporalWrapper
from OAT.models import TemporalModel

# --------------------------------------------------------------------------
# 1. LOAD CHECKPOINT
# --------------------------------------------------------------------------
CKPT_PATH = "../checkpoints/OAT/temporal_basic/best.pt"
OUT_DIR = "../checkpoints/OAT/temporal_basic/eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, best_val_loss={ckpt.get('best_val_loss')}")

# --------------------------------------------------------------------------
# 2. LOAD TEST DATASET (same wrapper classes as training)
# --------------------------------------------------------------------------
test_config = {
    "file_path": "../data/track_2_speed_change.csv",
    "T_half": 30,
    "outlier_window": 1,
    "outlier_thresh": 90.0,
    "normalize": True,
    "validate": True,
    "max_radius": 96,
    "dropout_min": 0.1,
    "dropout_max": 0.3,
    "clean_frac": 0.15,
    "pattern_weights": { "isolated": 0.35, "burst": 0.30, "center": 0.35 }
}
raw_test_dataset = TemporalDataset(file_path=test_config["file_path"], config=test_config)

T_half = test_config.get("T_half", 8)
max_radius = test_config.get("max_radius", 32)

test_dataset = TemporalWrapper(
    data_d=raw_test_dataset.data_d,
    relative_dist=raw_test_dataset.relative_dist,
    outlier_flags=raw_test_dataset.outlier_mask,
    T_half=T_half,
    max_radius=max_radius,
    dropout_min=test_config.get("dropout_min", 0.10),
    dropout_max=test_config.get("dropout_max", 0.20),
    clean_frac=test_config.get("clean_frac", 0.15),
    pattern_weights=test_config.get("pattern_weights", None)
)
test_dataset.new_epoch()

num_keypoints = raw_test_dataset.N
window_size = 2 * T_half + 1
center_idx = window_size // 2
part_names = raw_test_dataset.part_names

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)

# Deterministic masker for a fixed, reproducible occlusion ratio at test time
masking_cfg = {
    "mask_strategy": "temporal",
    "mask_min": 0.3,
    "mask_max": 0.7,
    "warmup_epochs": 50,
    "n_control": 3,
    "mixing_ratio": 0.5
}
masker = Masker(
    dimensions=(32, window_size, num_keypoints),
    mask_strategy="temporal",
    mask_min=masking_cfg.get("mask_min", 0.3),
    mask_max=masking_cfg.get("mask_max", 0.3),
    warmup_epochs=1,
    seed=123,
)

# --------------------------------------------------------------------------
# 3. TRIVIAL BASELINE: PER-KEYPOINT LINEAR INTERPOLATION OVER TIME
# --------------------------------------------------------------------------
class LinearInterpBaseline(nn.Module):
    """
    Takes the exact same inputs as TemporalModel: (window, occlusion_mask, offsets).
    For the center frame's occluded keypoints, fits a line per-keypoint-per-axis
    using that keypoint's VISIBLE frames in the window (weighted by proximity to
    the center via the real temporal offset), then evaluates at offset=0.
    Falls back to nearest-visible-value / last-known value if <2 visible points.
    """
    def __init__(self, center_idx):
        super().__init__()
        self.center_idx = center_idx

    @torch.no_grad()
    def forward(self, window, occlusion_mask, offsets):
        # window: (B, T, N, 3) | occlusion_mask: (B, T, N) True=visible | offsets: (B, T)
        B, T, N, _ = window.shape
        center_occ = occlusion_mask[:, self.center_idx].bool()   # (B, N)
        occluded_mask_center = ~center_occ
        completed = window[:, self.center_idx].clone()           # (B, N, 3)

        vis = occlusion_mask.bool()                               # (B, T, N)
        offs = offsets.float()                                    # (B, T)

        for b in range(B):
            for n in range(N):
                if not occluded_mask_center[b, n]:
                    continue
                v = vis[b, :, n]
                if v.sum() == 0:
                    continue  # nothing to interpolate from, leave as-is (GT passthrough, flagged later)
                x = offs[b, v]                       # (M,) real time offsets of visible frames
                y = window[b, v, n, :]                # (M, 3)
                if v.sum() == 1:
                    completed[b, n] = y[0]            # nearest-value fallback
                    continue
                # weighted least squares line per axis, weight by inverse |offset| distance
                w = 1.0 / (x.abs() + 1.0)
                W = torch.diag(w)
                A = torch.stack([x, torch.ones_like(x)], dim=1)   # (M, 2)
                for ax in range(3):
                    yb = y[:, ax]
                    AtW = A.T @ W
                    try:
                        coef = torch.linalg.solve(AtW @ A, AtW @ yb)
                        completed[b, n, ax] = coef[1]  # evaluate at offset = 0 (center)
                    except RuntimeError:
                        completed[b, n, ax] = yb[torch.argmin(x.abs())]

        return {"coordinates": completed, "occluded_mask": occluded_mask_center}


baseline = LinearInterpBaseline(center_idx).to(device)

# --------------------------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------------------------
model_config = {
    "encoder": {
        "embed_dim": 256,
        "depth": 4,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.1,
        "proj_dim": 256,
        "enable_last_norm": True
    },
    "decoder": {
        "proj_dim": 256,
        "window_size": 96,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "decoder_dim_feedforward": 1024,
        "decoder_dropout": 0.1,
        "predict_uncertainty": True,
        "predict_bond_aux": True,
        "bond_aux_dim": 3,
        "depth": 6
    }
}
model = TemporalModel(model_config, num_keypoints=num_keypoints,
                       window_size=window_size, max_offset=max_radius)
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

# --------------------------------------------------------------------------
# 4 & 5. RUN INFERENCE, COMPUTE ERRORS, SAVE ALL DATA
# --------------------------------------------------------------------------
records = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        window = batch["window"].to(device)          # (B, T, N, 3)
        offsets = batch["offsets"].to(device)         # (B, T)
        target_deform = batch["target_deform"].to(device).squeeze(1)  # (B, N, 3)

        B = window.shape[0]
        vis_mask = masker.get_mask(epoch=0)[:B].to(device)   # (B, T, N)
        center_vis_mask = vis_mask[:, center_idx]             # (B, N)
        occ_mask = ~center_vis_mask

        window_masked = window.clone()
        # zero out occluded points as inputs (model/baseline must predict them)
        window_masked[:, center_idx][occ_mask] = 0.0

        pred_model = model(window_masked, vis_mask, offsets)
        pred_base = baseline(window_masked, vis_mask, offsets)

        for b in range(B):
            occ = occ_mask[b].cpu().numpy()
            vis = center_vis_mask[b].cpu().numpy()
            gt = target_deform[b].cpu().numpy()
            pm = pred_model["coordinates"][b].cpu().numpy()
            pb = pred_base["coordinates"][b].cpu().numpy()

            err_model = np.linalg.norm(pm - gt, axis=-1)     # (N,)
            err_base = np.linalg.norm(pb - gt, axis=-1)      # (N,)

            for n in range(num_keypoints):
                records.append({
                    "sample_id": batch_idx * test_loader.batch_size + b,
                    "keypoint_idx": n,
                    "keypoint_name": part_names[n],
                    "visible": bool(vis[n]),
                    "occluded": bool(occ[n]),
                    "gt_x": gt[n, 0], "gt_y": gt[n, 1], "gt_z": gt[n, 2],
                    "model_pred_x": pm[n, 0], "model_pred_y": pm[n, 1], "model_pred_z": pm[n, 2],
                    "baseline_pred_x": pb[n, 0], "baseline_pred_y": pb[n, 1], "baseline_pred_z": pb[n, 2],
                    "model_err_l2": err_model[n] if occ[n] else np.nan,
                    "baseline_err_l2": err_base[n] if occ[n] else np.nan,
                    "model_err_x": abs(pm[n, 0] - gt[n, 0]) if occ[n] else np.nan,
                    "model_err_y": abs(pm[n, 1] - gt[n, 1]) if occ[n] else np.nan,
                    "model_err_z": abs(pm[n, 2] - gt[n, 2]) if occ[n] else np.nan,
                    "baseline_err_x": abs(pb[n, 0] - gt[n, 0]) if occ[n] else np.nan,
                    "baseline_err_y": abs(pb[n, 1] - gt[n, 1]) if occ[n] else np.nan,
                    "baseline_err_z": abs(pb[n, 2] - gt[n, 2]) if occ[n] else np.nan,
                    "occlusion_ratio_center": occ.mean(),
                })

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUT_DIR, "test_results_full.csv"), index=False)
print(f"Saved {len(df)} keypoint-level records to test_results_full.csv")

summary = df[df.occluded].groupby("keypoint_name")[
    ["model_err_l2", "baseline_err_l2"]
].mean().sort_values("model_err_l2", ascending=False)
summary.to_csv(os.path.join(OUT_DIR, "per_keypoint_summary.csv"))
print(summary)

# --------------------------------------------------------------------------
# 6. COMPREHENSIVE ERROR COMPARISON PLOTS (incl. per-axis XYZ)
# --------------------------------------------------------------------------
occ_df = df[df.occluded].copy()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) Overall L2 error distribution: model vs baseline
axes[0, 0].hist(occ_df["model_err_l2"].dropna(), bins=50, alpha=0.6, label="Model")
axes[0, 0].hist(occ_df["baseline_err_l2"].dropna(), bins=50, alpha=0.6, label="Linear Interp")
axes[0, 0].set_title("Overall L2 Error Distribution")
axes[0, 0].set_xlabel("L2 error"); axes[0, 0].legend()

# (b) Per-axis MAE comparison
axis_err = pd.DataFrame({
    "Model": [occ_df["model_err_x"].mean(), occ_df["model_err_y"].mean(), occ_df["model_err_z"].mean()],
    "Baseline": [occ_df["baseline_err_x"].mean(), occ_df["baseline_err_y"].mean(), occ_df["baseline_err_z"].mean()],
}, index=["X", "Y", "Z"])
axis_err.plot(kind="bar", ax=axes[0, 1])
axes[0, 1].set_title("Per-Axis Mean Absolute Error")
axes[0, 1].set_ylabel("MAE")

# (c) Per-keypoint mean L2 error
kp_err = occ_df.groupby("keypoint_name")[["model_err_l2", "baseline_err_l2"]].mean()
kp_err.plot(kind="bar", ax=axes[0, 2])
axes[0, 2].set_title("Per-Keypoint Mean L2 Error")
axes[0, 2].tick_params(axis="x", rotation=45)

# (d) Error vs. occlusion ratio of center frame
bins = pd.cut(occ_df["occlusion_ratio_center"], bins=8)
ratio_err = occ_df.groupby(bins)[["model_err_l2", "baseline_err_l2"]].mean()
ratio_err.plot(kind="line", marker="o", ax=axes[1, 0])
axes[1, 0].set_title("Error vs. Center-Frame Occlusion Ratio")
axes[1, 0].tick_params(axis="x", rotation=45)

# (e) Boxplot: model vs baseline per axis
melt = occ_df.melt(value_vars=["model_err_x", "model_err_y", "model_err_z",
                                "baseline_err_x", "baseline_err_y", "baseline_err_z"],
                    var_name="metric", value_name="err")
melt["model"] = melt["metric"].apply(lambda m: "Model" if "model" in m else "Baseline")
melt["axis"] = melt["metric"].apply(lambda m: m[-1].upper())
import seaborn as sns
sns.boxplot(data=melt, x="axis", y="err", hue="model", ax=axes[1, 1], showfliers=False)
axes[1, 1].set_title("Per-Axis Error Spread (Model vs Baseline)")

# (f) Model improvement (or regression) over baseline per keypoint
improvement = (kp_err["baseline_err_l2"] - kp_err["model_err_l2"]) / kp_err["baseline_err_l2"] * 100
improvement.plot(kind="bar", ax=axes[1, 2], color=np.where(improvement > 0, "green", "red"))
axes[1, 2].set_title("% Improvement of Model over Baseline")
axes[1, 2].axhline(0, color="black", linewidth=0.8)
axes[1, 2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "error_comparison_full.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# 7. INTERACTIVE PER-SAMPLE WINDOW VIEWER (visible/occluded + both preds)
# --------------------------------------------------------------------------
def plot_sample_window(sample_id, dataset=test_dataset, model=model, baseline=baseline,
                        device=device, center_idx=center_idx, masker=masker):
    """
    Slider-based viewer: scroll through the T+1 window frames for a given
    dataset sample_id. Shows which keypoints are visible/occluded at each
    frame; at the center frame, overlays GT, model prediction, and baseline
    prediction for occluded points.
    """
    sample = dataset[sample_id]
    window = sample["window"].unsqueeze(0).to(device)      # (1, T, N, 3)
    offsets = sample["offsets"].unsqueeze(0).to(device)
    gt_center = sample["target_deform"].squeeze(0).cpu().numpy()

    vis_mask = masker.get_mask(epoch=0)[:1].to(device)
    center_vis = vis_mask[0, center_idx].cpu().numpy()
    occ = ~center_vis

    window_masked = window.clone()
    window_masked[:, center_idx][~vis_mask[:, center_idx]] = 0.0

    with torch.no_grad():
        pm = model(window_masked, vis_mask, offsets)["coordinates"][0].cpu().numpy()
        pb = baseline(window_masked, vis_mask, offsets)["coordinates"][0].cpu().numpy()

    T = window.shape[1]
    win_np = window[0].cpu().numpy()
    vis_np = vis_mask[0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.2)
    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(slider_ax, "Frame (offset)", 0, T - 1, valinit=center_idx, valstep=1)

    def draw(frame_idx):
        ax.clear()
        pts = win_np[frame_idx]
        v = vis_np[frame_idx]
        ax.scatter(pts[v, 0], pts[v, 1], c="blue", label="Visible", s=60)
        ax.scatter(pts[~v, 0], pts[~v, 1], c="gray", marker="x", label="Occluded (input)", s=60)

        if frame_idx == center_idx:
            ax.scatter(gt_center[occ, 0], gt_center[occ, 1], c="black", marker="*",
                       label="GT (occluded)", s=90, zorder=5)
            ax.scatter(pm[occ, 0], pm[occ, 1], c="red", marker="^",
                       label="Model pred", s=70, zorder=4)
            ax.scatter(pb[occ, 0], pb[occ, 1], c="green", marker="v",
                       label="Baseline pred", s=70, zorder=4)
            ax.set_title(f"Sample {sample_id} — CENTER frame (offset 0)")
        else:
            real_offset = offsets[0, frame_idx].item()
            ax.set_title(f"Sample {sample_id} — context frame (offset {real_offset:+.0f})")

        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        fig.canvas.draw_idle()

    draw(center_idx)
    slider.on_changed(lambda val: draw(int(val)))
    plt.show()
    return fig, slider

# Example usage: plot_sample_window(100)

# --------------------------------------------------------------------------
# 8. FAILURE MODE ANALYSIS
# --------------------------------------------------------------------------
failure_records = []

# (a) Worst-performing samples overall
sample_err = occ_df.groupby("sample_id")["model_err_l2"].mean().sort_values(ascending=False)
worst_samples = sample_err.head(20)
worst_samples.to_csv(os.path.join(OUT_DIR, "worst_samples.csv"))

# (b) Keypoint-specific failure rate: how often model err > baseline err
occ_df["model_worse"] = occ_df["model_err_l2"] > occ_df["baseline_err_l2"]
worse_rate = occ_df.groupby("keypoint_name")["model_worse"].mean().sort_values(ascending=False)
worse_rate.to_csv(os.path.join(OUT_DIR, "model_worse_than_baseline_rate.csv"))

# (c) Error vs. distance-to-nearest-visible-context (extensibility for non-consecutive gaps)
# Requires offsets from the sample; approximate via largest gap in visible offsets seen per sample
gap_records = []
for i in range(len(test_dataset)):
    s = test_dataset[i]
    offs = s["offsets"].numpy()
    gap_records.append({"sample_id": i, "max_abs_offset": np.abs(offs).max()})
gap_df = pd.DataFrame(gap_records)
merged = occ_df.merge(gap_df, on="sample_id", how="left")
gap_bins = pd.cut(merged["max_abs_offset"], bins=6)
gap_err = merged.groupby(gap_bins)[["model_err_l2", "baseline_err_l2"]].mean()
gap_err.to_csv(os.path.join(OUT_DIR, "error_vs_context_gap.csv"))

fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
gap_err.plot(kind="line", marker="o", ax=ax2[0])
ax2[0].set_title("Error vs. Max Temporal Gap to Context")
ax2[0].tick_params(axis="x", rotation=45)

worst_kp = worse_rate.head(10)
worst_kp.plot(kind="barh", ax=ax2[1], color="crimson")
ax2[1].set_title("Keypoints Where Model Loses to Baseline Most Often")
ax2[1].set_xlabel("Fraction of occluded instances model_err > baseline_err")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "failure_mode_analysis.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)

print("Evaluation complete. All outputs saved to:", OUT_DIR)