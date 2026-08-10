
import os
import random
import time
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import wandb
from torchinfo import summary as torch_summary


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """Linear warmup -> cosine annealing over the remaining steps."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, model, loss_fn, masker, train_dataset, val_dataset,
                 raw_train_dataset, raw_val_dataset, logger, config, device=None, seed=42):
        self.config = config
        self.seed = seed
        set_seed(self.seed)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.masker = masker
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.raw_train_dataset = raw_train_dataset
        self.raw_val_dataset = raw_val_dataset
        self.py_logger = logger

        run_name = config.get("run_name", time.strftime("run_%Y%m%d_%H%M%S"))
        self.ckpt_dir = Path(config.get("checkpoint_root")) 
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.val_print_path = self.ckpt_dir / "val_predictions.txt"

        self.grad_clip_norm = config.get("grad_clip_norm", 1.0)
        self.log_interval = config.get("log_interval", 50)
        self.ckpt_interval = config.get("checkpoint_interval", 5)
        self.visualize_interval = config.get("visualize_interval", 10)
        self.global_step = 0
        self.start_epoch = 0

        self._init_logger()
        self._init_wandb()

        es_cfg = config.get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.patience = es_cfg.get("patience", 15)
        self.min_delta = es_cfg.get("min_delta", 1e-4)
        self.epochs_no_improve = 0
        self.best_val_loss = float("inf")

        self.train_loss_history = defaultdict(list)
        self.val_loss_history = defaultdict(list)
        self.lr_history = []
        dl_cfg = config.get("dataloader", {})
        self.batch_size = dl_cfg.get("batch_size", 32)
        self.num_workers = dl_cfg.get("num_workers", 4)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )

        opt_cfg = config.get("optimizer", {})
        self.lr = opt_cfg.get("lr", 3e-4)
        self.weight_decay = opt_cfg.get("weight_decay", 1e-2)
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
            betas=opt_cfg.get("betas", (0.9, 0.999)),
        )

        sched_cfg = config.get("scheduler", {})
        self.epochs = config.get("epochs", 100)
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.epochs * self.steps_per_epoch
        self.warmup_steps = sched_cfg.get(
            "warmup_steps", int(sched_cfg.get("warmup_frac", 0.05) * self.total_steps)
        )
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer, self.warmup_steps, self.total_steps,
            min_lr_ratio=sched_cfg.get("min_lr_ratio", 0.01),
        )

        self._log_model_summary()

        if config.get("resume_from"):
            self.load_checkpoint(config["resume_from"])

    def _init_logger(self):
        log_path = self.ckpt_dir / "train.log"
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)

        if self.py_logger is None:
            self.py_logger = logging.getLogger(f"trainer_{self.ckpt_dir.name}")
            self.py_logger.setLevel(logging.INFO)
            self.py_logger.propagate = False
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(fmt)
            self.py_logger.addHandler(stream_handler)

        self.py_logger.addHandler(file_handler)
        self.py_logger.setLevel(logging.INFO)
        self.py_logger.info(f"Checkpoint dir: {self.ckpt_dir}")

    def _init_wandb(self):
        self.use_wandb = self.config.get("use_wandb", False) and wandb is not None
        if self.use_wandb:
            wandb.init(
                project=self.config.get("wandb_project", "mouse-keypoint-transformer"),
                name=self.ckpt_dir.name,
                config=self.config,
                dir=str(self.ckpt_dir),
            )

    def _log_model_summary(self):
        self.py_logger.info(f"Model: {self.model.__class__.__name__}")
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.py_logger.info(f"Trainable parameters: {n_params:,}")
        if torch_summary is not None:
            try:
                sample = self.train_dataset[0]
                window = sample["window"].unsqueeze(0).to(self.device)
                B, T1, N, _ = window.shape
                occ = torch.ones(B, T1, N, dtype=torch.bool, device=self.device)
                offs = sample["offsets"].unsqueeze(0).to(self.device)
                info = torch_summary(
                    self.model, input_data=(window, occ, offs), verbose=0,
                )
                self.py_logger.info(f"\n{info}")
            except Exception as e:
                self.py_logger.warning(f"torchinfo summary failed: {e}")
        else:
            self.py_logger.warning("torchinfo not installed; skipping architecture summary.")

    def save_checkpoint(self, epoch, is_best=False, tag=None):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(state, best_path)
            self.py_logger.info(f"Saved new best checkpoint: {best_path}")
        else:
            name = tag or f"epoch_{epoch:04d}"
            path = self.ckpt_dir / f"{name}.pt"
            torch.save(state, path)
            self.py_logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        self.global_step = state.get("global_step", 0)
        self.start_epoch = state.get("epoch", 0) + 1
        self.py_logger.info(f"Resumed from {path} at epoch {self.start_epoch}")

    def _build_targets(self, batch, center_vis_mask):
        return {
            "coordinates": batch["target_deform"].squeeze(1),   # (B, N, 3)
            "rel_dist": batch["target_rel"].squeeze(1),          # (B, N, 3)
            "occluded_mask": ~center_vis_mask,                   # True = occluded = used in loss
        }
    def _batch_to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train_one_epoch(self, epoch):
        self.model.train()
        running = defaultdict(float)
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._batch_to_device(batch)
            B = batch["window"].shape[0]
            vis_mask = self.masker.get_mask(epoch=epoch, batch_size=B).to(self.device, non_blocking=True)

            center_idx = batch["window"].shape[1] // 2
            center_vis_mask = vis_mask[:, center_idx, :]  # (B, N)

            pred = self.model(batch["window"], vis_mask, batch["offsets"])
            target = self._build_targets(batch, center_vis_mask)
            losses = self.loss_fn(pred, target)
            total_loss = losses["total"]

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            for k, v in losses.items():
                running[k] += v.item() if torch.is_tensor(v) else v

            if batch_idx % self.log_interval == 0:
                lr_now = self.scheduler.get_last_lr()[0]
                self.py_logger.info(
                    f"[Epoch {epoch}] Step {batch_idx}/{n_batches} | "
                    f"loss={total_loss.item()+7.0:.6f} | " + " | ".join([f"{k}:{v.item():.6f}" for k,v in losses.items() if k != "total"]) +
                    f" | lr={lr_now:.2e}"
                )

        epoch_metrics = {k: v / n_batches for k, v in running.items()}
        return epoch_metrics

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.model.eval()
        all_records = []
        running = defaultdict(float)
        n_batches = len(self.val_loader)
        printed_samples = []

        for batch_idx, batch in enumerate(self.val_loader):
            batch = self._batch_to_device(batch)
            B = batch["window"].shape[0]
            vis_mask = self.masker.get_mask(epoch=epoch, batch_size=B).to(self.device, non_blocking=True)

            center_idx = batch["window"].shape[1] // 2
            center_vis_mask = vis_mask[:, center_idx, :]

            pred = self.model(batch["window"], vis_mask, batch["offsets"])
            target = self._build_targets(batch, center_vis_mask)
            losses = self.loss_fn(pred, target)

            for k, v in losses.items():
                running[k] += v.item() if torch.is_tensor(v) else v

            if batch_idx == 0:
                err = (pred["coordinates"] - target["coordinates"]).norm(dim=-1)  # (B, N)
                occ = target["occluded_mask"]
                mean_err_occ = err[occ].mean().item() if occ.any() else float("nan")
                printed_samples.append(
                    f"Epoch {epoch} | sample occluded-point mean L2 error: {mean_err_occ:.5f}"
                )

            # Book-keeping
            err_full = (pred["coordinates"] - target["coordinates"]).norm(dim=-1)  # (B, N)
            occ_full = target["occluded_mask"]                                    # (B, N)
            per_sample_err = torch.where(
                occ_full, err_full, torch.zeros_like(err_full)
            ).sum(-1) / occ_full.sum(-1).clamp(min=1)

            for b in range(B):
                all_records.append({
                    "error": per_sample_err[b].item(),
                    "xy_true": target["coordinates"][b, :, :2].cpu().numpy(),
                    "xy_pred": pred["coordinates"][b, :, :2].cpu().numpy(),
                    "vis_mask": center_vis_mask[b].cpu().numpy(),
                    "occ_mask": occ_full[b].cpu().numpy(),
                })

        epoch_metrics = {k: v / n_batches for k, v in running.items()}

        with open(self.val_print_path, "a") as f:
            f.write("\n".join(printed_samples) + "\n")
            f.write(f"  metrics: {json.dumps(epoch_metrics)}\n")

        # vizualize
        if getattr(self, "visualize_interval", 0) and epoch % self.visualize_interval == 0 and all_records:
            all_records.sort(key=lambda r: r["error"])
            best = all_records[0]
            median = all_records[len(all_records) // 2]
            worst = all_records[-1]
            best["label"], median["label"], worst["label"] = "Best", "Median", "Worst"
            plot_best_worst_median(epoch, [best, median, worst], self.ckpt_dir)

        return epoch_metrics

    def log_epoch(self, epoch, train_metrics, val_metrics, epoch_time):
        for k, v in train_metrics.items():
            self.train_loss_history[k].append(v)
        for k, v in val_metrics.items():
            self.val_loss_history[k].append(v)
        lr_now = self.scheduler.get_last_lr()[0]
        self.lr_history.append(lr_now)

        self.py_logger.info(
            f"Epoch {epoch} done in {epoch_time:.1f}s | "
            f"train_total={train_metrics.get('total', float('nan')):.6f} | "
            f"val_total={val_metrics.get('total', float('nan')):.6f} | lr={lr_now:.2e}"
        )

        if self.use_wandb:
            log_dict = {"epoch": epoch, "lr": lr_now}
            for k, v in train_metrics.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
            wandb.log(log_dict)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()

            self.train_dataset.new_epoch()
            self.val_dataset.new_epoch()

            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)
            epoch_time = time.time() - t0

            self.log_epoch(epoch, train_metrics, val_metrics, epoch_time)

            val_total = val_metrics.get("total", float("inf"))
            is_best = val_total < (self.best_val_loss - self.min_delta)
            if is_best:
                self.best_val_loss = val_total
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if is_best or (epoch % self.ckpt_interval == 0):
                self.save_checkpoint(epoch, is_best=is_best)

            if self.early_stopping_enabled and self.epochs_no_improve >= self.patience:
                self.py_logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs)."
                )
                break

        self.save_checkpoint(epoch, is_best=False, tag="final")
        if self.use_wandb:
            wandb.finish()
        self.py_logger.info("Training complete.")

def plot_best_worst_median(epoch, samples, ckpt_dir, keypoint_names=None):
    """
    samples: list of exactly 3 dicts, ordered [best, median, worst], each with:
        'label'    : str, e.g. "Best" / "Median" / "Worst"
        'error'    : float, the per-sample mean L2 error used to rank it
        'xy_true'  : (N, 2) ndarray, ground-truth center-frame XY (deformable target)
        'xy_pred'  : (N, 2) ndarray, model prediction, only meaningful at occluded idx
        'vis_mask' : (N,) bool, True = visible in center frame
        'occ_mask' : (N,) bool, True = occluded in center frame (== ~vis_mask)
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, s in zip(axes, samples):
        xy_true, xy_pred = s["xy_true"], s["xy_pred"]
        vis, occ = s["vis_mask"], s["occ_mask"]

        # visible keypoints: solid blue circles
        ax.scatter(xy_true[vis, 0], xy_true[vis, 1],
                   c="blue", marker="o", s=60, alpha=1.0,
                   label="Visible", zorder=3)

        # occluded keypoints (ground truth): blue X, opacity 0.5
        ax.scatter(xy_true[occ, 0], xy_true[occ, 1],
                   c="blue", marker="x", s=70, alpha=0.5,
                   label="Occluded (GT)", zorder=3)

        # predictions of occluded keypoints: red triangles
        ax.scatter(xy_pred[occ, 0], xy_pred[occ, 1],
                   c="red", marker="^", s=60, alpha=1.0,
                   label="Predicted", zorder=4)

        # dashed connector between each occluded GT point and its prediction
        occ_idx = np.nonzero(occ)[0]
        for i in occ_idx:
            ax.plot([xy_true[i, 0], xy_pred[i, 0]],
                    [xy_true[i, 1], xy_pred[i, 1]],
                    linestyle="--", color="gray", linewidth=1.0,
                    alpha=0.7, zorder=2)

        if keypoint_names is not None:
            for i in range(len(xy_true)):
                ax.annotate(keypoint_names[i], xy_true[i], fontsize=6, alpha=0.6)

        ax.set_title(f"{s['label']} (err={s['error']:.4f})", fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"Validation reconstructions — epoch {epoch}", y=1.14, fontsize=13)
    fig.tight_layout()

    out_path = ckpt_dir / f"val_vis_epoch_{epoch:04d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path