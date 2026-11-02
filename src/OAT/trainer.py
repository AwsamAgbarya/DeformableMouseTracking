import json
import logging
import math
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    import wandb  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    wandb = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


__all__ = [
    "TemporalTrainer",
    "Trainer",
    "set_seed",
    "build_warmup_cosine_scheduler",
    "hidden_bout_lengths",
    "ModelEMA",
    "plot_predictions",
]

def set_seed(seed, deterministic=False):
    """Seed python, numpy and torch. Returns the seed, for logging."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """Linear warmup -> cosine annealing over the remaining steps."""
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), warmup_steps + 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def hidden_bout_lengths(visible):
    vis = torch.as_tensor(visible).bool()
    hid = ~vis
    T, K = hid.shape
    start = torch.zeros_like(hid)
    start[0] = hid[0]
    start[1:] = hid[1:] & ~hid[:-1]
    run_id = start.long().cumsum(0) * hid.long()           # 0 where visible
    out = torch.zeros(T, K, dtype=torch.long)
    for k in range(K):
        ids = run_id[:, k]
        nz = ids > 0
        if not bool(nz.any()):
            continue
        counts = torch.bincount(ids[nz])
        out[nz, k] = counts[ids[nz]]
    return out


class ModelEMA:
    """Exponential moving average of the weights, with the usual warmup ramp."""

    def __init__(self, model, decay=0.999, warmup=1000):
        self.decay = float(decay)
        self.warmup = int(warmup)
        self.updates = 0
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / max(self.warmup, 1)))
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow.state_dict()}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.updates = sd["updates"]
        self.shadow.load_state_dict(sd["shadow"])


class MetricAccumulator:
    """
    Per-joint / per-condition MPJPE, accumulated as (sum of errors, count) so
    the result is a true mean over joints and not a mean of per-batch means.
    """

    HIDDEN_BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    BOUT_BUCKETS = [(1, 1), (2, 3), (4, 8), (9, 20), (21, 10 ** 9)]

    def __init__(self, n_joints, scale_mm=1.0, part_names=None,
                 bout_lengths=None, behaviour=None):
        self.J = int(n_joints)
        self.scale_mm = float(scale_mm)
        self.part_names = list(part_names) if part_names is not None else \
            [f"j{i}" for i in range(self.J)]
        self.bout_lengths = bout_lengths          # (T, J) long or None
        self.behaviour = behaviour                # (T,) array of labels or None
        self.reset()

    def reset(self):
        self.joint_sum = torch.zeros(self.J, dtype=torch.float64)
        self.joint_cnt = torch.zeros(self.J, dtype=torch.float64)
        self.group = defaultdict(lambda: [0.0, 0.0])      # key -> [sum, count]
        self.total = [0.0, 0.0]

    @staticmethod
    def _bucket(value, buckets, fmt):
        for lo, hi in buckets:
            if lo <= value <= hi:
                return fmt(lo, hi)
        return "other"

    @torch.no_grad()
    def update(self, pred, target, batch):
        cp = pred["coordinates"].detach().float().cpu()
        ct = target["coordinates"].detach().float().cpu()
        occ = pred["occluded_mask"].detach().cpu().bool()
        err = (cp - ct).norm(dim=-1) * self.scale_mm              # (B, J)

        self.joint_sum += (err * occ).sum(0).double()
        self.joint_cnt += occ.sum(0).double()
        self.total[0] += float((err * occ).sum())
        self.total[1] += float(occ.sum())

        centers = batch.get("center_index")
        centers = centers.detach().cpu().long() if centers is not None else None
        dropped = batch.get("center_dropped")
        dropped = dropped.detach().cpu().bool() if dropped is not None else None

        frac = occ.float().mean(1)                                 # (B,)
        B = err.shape[0]
        for b in range(B):
            e_b, m_b = err[b], occ[b]
            n = float(m_b.sum())
            if n == 0:
                continue
            s = float((e_b * m_b).sum())

            key = self._bucket(float(frac[b]), self.HIDDEN_BUCKETS,
                               lambda lo, hi: f"hidden/{lo:.1f}-{min(hi, 1.0):.1f}")
            g = self.group[key]; g[0] += s; g[1] += n

            if dropped is not None:
                key = "center/dropped" if bool(dropped[b]) else "center/present"
                g = self.group[key]; g[0] += s; g[1] += n

            if self.behaviour is not None and centers is not None:
                lab = str(self.behaviour[int(centers[b])])
                g = self.group[f"behaviour/{lab}"]; g[0] += s; g[1] += n

            if self.bout_lengths is not None and centers is not None:
                bl = self.bout_lengths[int(centers[b])]            # (J,) long
                for j in torch.nonzero(m_b).flatten().tolist():
                    L = int(bl[j])
                    if L <= 0:
                        continue
                    key = self._bucket(L, self.BOUT_BUCKETS,
                                       lambda lo, hi: f"bout/{lo}-{hi if hi < 10**9 else 'inf'}f")
                    g = self.group[key]; g[0] += float(e_b[j]); g[1] += 1.0

    def result(self):
        out = {}
        if self.total[1] > 0:
            out["mpjpe_mm"] = self.total[0] / self.total[1]
        cnt = self.joint_cnt.clamp_min(1)
        per_joint = (self.joint_sum / cnt).tolist()
        out["per_joint_mm"] = {
            n: v for n, v, c in zip(self.part_names, per_joint, self.joint_cnt.tolist()) if c > 0
        }
        for k, (s, c) in sorted(self.group.items()):
            if c > 0:
                out[k] = s / c
                out[f"{k}__n"] = int(c)
        return out



class Trainer:
    MONITOR_DEFAULT = "metric/mpjpe_mm"

    def __init__(self, model, loss_fn, train_dataset, val_dataset, config,
                edges=None, part_names=None,
                 behaviour_labels=None, logger=None, device=None):
        self.config = dict(config or {})
        self.seed = set_seed(config.seed)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.edges = edges
        self.part_names = part_names
        self.behaviour_labels = behaviour_labels

        self.scale_mm = float(getattr(self.loss_fn, "scale_mm", 1.0))
        run_name = self.config.get("run_name", time.strftime("run_%Y%m%d_%H%M%S"))
        self.ckpt_dir = Path(self.config.get("checkpoint_root", "./runs")) / run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.grad_clip_norm = float(self.config.get("grad_clip_norm", 1.0))
        self.accum_steps = max(int(self.config.get("grad_accum_steps", 1)), 1)
        self.log_interval = int(self.config.get("log_interval", 50))
        self.ckpt_interval = int(self.config.get("checkpoint_interval", 5))
        self.visualize_interval = int(self.config.get("visualize_interval", 10))
        self.epochs = int(self.config.get("epochs", 100))
        self.global_step = 0
        self.start_epoch = 0
        self.skipped_steps = 0
        self.last_breakdown = {}

        self.py_logger = logger
        self._init_logger()
        self._init_wandb()
        self._init_amp()

        dl = self.config.get("dataloader", {})
        self.batch_size = int(dl.get("batch_size", 64))
        self.num_workers = int(dl.get("num_workers", 4))

        # The validation occlusion field is frozen: built once, never resampled.
        self.val_dataset.set_epoch(0)
        self.val_loader = self._make_loader(self.val_dataset, shuffle=False, drop_last=False)
        self._log_val_field()

        opt = self.config.get("optimizer", {})
        self.optimizer = AdamW(
            self._param_groups(float(opt.get("weight_decay", 0.05))),
            lr=float(opt.get("lr", 3e-4)),
            betas=tuple(opt.get("betas", (0.9, 0.95))),
        )

        sch = self.config.get("scheduler", {})
        self.train_dataset.set_epoch(0)
        self.steps_per_epoch = max(len(self.train_dataset) // self.batch_size, 1) // self.accum_steps
        self.total_steps = max(self.epochs * max(self.steps_per_epoch, 1), 1)
        warmup = sch.get("warmup_steps", int(float(sch.get("warmup_frac", 0.03)) * self.total_steps))
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer, warmup, self.total_steps,
            min_lr_ratio=float(sch.get("min_lr_ratio", 0.01)),
        )

        ema_cfg = self.config.get("ema", {})
        self.ema = (
            ModelEMA(self.model, float(ema_cfg.get("decay", 0.999)), int(ema_cfg.get("warmup", 1000)))
            if ema_cfg.get("enabled", True) else None
        )

        es = self.config.get("early_stopping", {})
        self.early_stopping_enabled = bool(es.get("enabled", True))
        self.patience = int(es.get("patience", 20))
        self.min_delta = float(es.get("min_delta", 1e-4))
        self.monitor = es.get("monitor", self.MONITOR_DEFAULT)
        self.epochs_no_improve = 0
        self.best_score = float("inf")

        self.history = {"train": defaultdict(list), "val": defaultdict(list), "lr": []}

        self._log_setup()

        if self.config.get("resume_from"):
            self.load_checkpoint(self.config["resume_from"])

    def _init_logger(self):
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        if self.py_logger is None:
            self.py_logger = logging.getLogger(f"trainer_{self.ckpt_dir.name}")
            self.py_logger.setLevel(logging.INFO)
            self.py_logger.propagate = False
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self.py_logger.addHandler(sh)
        fh = logging.FileHandler(self.ckpt_dir / "train.log")
        fh.setFormatter(fmt)
        self.py_logger.addHandler(fh)
        self.py_logger.setLevel(logging.INFO)
        self.py_logger.info(f"Checkpoint dir: {self.ckpt_dir}")

    def _init_wandb(self):
        self.use_wandb = bool(self.config.get("use_wandb", False)) and wandb is not None
        if self.use_wandb:
            wandb.init(
                project=self.config.get("wandb_project", "oat-mouse-keypoints"),
                name=self.ckpt_dir.name, config=self.config, dir=str(self.ckpt_dir),
            )

    def _init_amp(self):
        mode = str(self.config.get("amp", "bf16")).lower()
        cuda = self.device == "cuda"
        if mode == "off" or not cuda:
            self.amp_dtype, self.scaler = None, None
        elif mode == "bf16" and torch.cuda.is_bf16_supported():
            self.amp_dtype, self.scaler = torch.bfloat16, None
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
        self.py_logger.info(f"AMP: {self.amp_dtype}")

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device, dtype=self.amp_dtype)

    def _param_groups(self, weight_decay):
        """No weight decay on norms, biases, embeddings or learned tokens."""
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() <= 1 or "embed" in name or "token" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _make_loader(self, ds, shuffle, drop_last=False):
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=(self.device == "cuda"),
            drop_last=drop_last, persistent_workers=False,
        )

    def _log_val_field(self):
        rep = self.val_dataset.epoch_report()
        self.py_logger.info(
            "Frozen validation occlusion: "
            + " | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in rep.items())
        )

    def _log_setup(self):
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        dec = getattr(self.model, "decoder", None)
        heads = ["coordinates"]
        if getattr(dec, "predict_uncertainty", False):
            heads.append("log_sigma")
        if getattr(dec, "predict_frame", False):
            heads.append("frame_se2")
        self.py_logger.info(f"Model {self.model.__class__.__name__} | {n:,} trainable parameters")
        self.py_logger.info(
            f"window={getattr(self.model, 'window_size', '?')} "
            f"joints={getattr(self.model, 'num_keypoints', self.val_dataset.J)} "
            f"heads=[{', '.join(heads)}] | 1 unit = {self.scale_mm:.3f} mm"
        )

    #  batch

    def _to_device(self, batch):
        return {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    def _forward(self, model, batch):
        """TemporalModel takes the window explicitly, not the batch dict."""
        extra = {}
        if getattr(model, "uses_relpose", False):
            # A4c only: each slot's pose relative to the reference frame
            extra = {"rel_yaw": batch["rel_yaw"], "rel_xy": batch["rel_xy"]}
        return model(
            batch["window"],          # (B, W, J, 3), already in the window frame
            batch["vis_mask"],        # (B, W, J) True = observed
            frame_offsets=batch["offsets"],
            pad_mask=batch["pad_mask"],
            **extra,
        )

    @torch.no_grad()
    def _build_targets(self, batch):
        """
        The scored mask is the DATASET's, always. Occlusion is a property of
        the recording, sampled once per epoch by `RecordingOcclusion`; nothing
        in the model or the trainer is allowed to add to it, so `center_vis` is
        the single source of truth for what the model could not see.
        """
        occ = ~batch["center_vis"].bool()
        if "frame_ok" in batch:
            occ = occ & batch["frame_ok"].view(-1, 1)
        target = {
            "coordinates": batch["target_coords"],
            "occluded_mask": occ,
        }

        return target

    #  epochs

    @staticmethod
    def _accumulate(running, counts, losses):
        for k, v in losses.items():
            x = float(v.detach()) if torch.is_tensor(v) else float(v)
            if math.isfinite(x):
                running[k] += x
                counts[k] += 1

    def train_one_epoch(self, epoch, loader):
        self.model.train()
        running, counts = defaultdict(float), defaultdict(int)
        n_batches = len(loader)
        self.optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            batch = self._to_device(batch)
            target = self._build_targets(batch)

            with self._autocast():
                pred = self._forward(self.model, batch)
                losses = self.loss_fn(pred, target)
                loss = losses["total"] / self.accum_steps

            if not torch.isfinite(loss):
                self.skipped_steps += 1
                self.optimizer.zero_grad(set_to_none=True)
                self.py_logger.warning(f"[Epoch {epoch}] non-finite loss at step {i}; skipped")
                continue

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.model)
                running["grad_norm"] += float(gnorm)
                counts["grad_norm"] += 1

            self._accumulate(running, counts, losses)

            if self.log_interval and i % self.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                terms = " | ".join(
                    f"{k}={float(v.detach()):.4f}" for k, v in losses.items()
                    if k != "total" and not k.startswith("metric/")
                )
                self.py_logger.info(
                    f"[Epoch {epoch}] {i}/{n_batches} | total={float(losses['total'].detach()):.4f} | "
                    f"{terms} | mpjpe={float(losses.get('metric/mpjpe_mm', float('nan'))):.3f}mm "
                    f"| lr={lr:.2e}"
                )

        return {k: running[k] / max(counts[k], 1) for k in running}

    @torch.no_grad()
    def validate(self, epoch, model=None, collect=True):
        model = model or (self.ema.shadow if self.ema is not None else self.model)
        model.eval()
        running, counts = defaultdict(float), defaultdict(int)
        acc = MetricAccumulator(
            n_joints=getattr(self.model, "J", self.val_dataset.J),
            scale_mm=self.scale_mm,
            part_names=self.part_names,
            bout_lengths=self._val_bouts(),
            behaviour=self.behaviour_labels,
        )
        records = []

        for bi, batch in enumerate(self.val_loader):
            batch = self._to_device(batch)
            target = self._build_targets(batch)
            with self._autocast():
                pred = self._forward(model, batch)
            pred = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                    for k, v in pred.items()}
            losses = self.loss_fn(pred, target)

            self._accumulate(running, counts, losses)
            acc.update(pred, target, batch)

            if collect and self.visualize_interval and epoch % self.visualize_interval == 0:
                records.extend(self._records(pred, target, batch))

        metrics = {k: running[k] / max(counts[k], 1) for k in running}
        self.last_breakdown = acc.result()
        metrics["metric/mpjpe_mm"] = self.last_breakdown.get("mpjpe_mm", float("nan"))

        if records:
            records.sort(key=lambda r: r["error"])
            picks = [records[0], records[len(records) // 2], records[-1]]
            for p, lab in zip(picks, ("Best", "Median", "Worst")):
                p["label"] = lab
            plot_predictions(epoch, picks, self.ckpt_dir, self.edges, self.part_names)

        return metrics

    def _val_bouts(self):
        """Hidden-bout lengths of the frozen validation field. Computed once."""
        if hasattr(self, "_val_bout_cache"):
            return self._val_bout_cache
        vis = getattr(self.val_dataset, "visible", None)
        self._val_bout_cache = None if vis is None else hidden_bout_lengths(vis.cpu())
        return self._val_bout_cache

    @staticmethod
    def _records(pred, target, batch):
        cp = pred["coordinates"].detach().float().cpu()
        ct = target["coordinates"].detach().float().cpu()
        occ = pred["occluded_mask"].detach().cpu().bool()
        err = (cp - ct).norm(dim=-1)
        per = (err * occ).sum(-1) / occ.sum(-1).clamp(min=1)
        out = []
        for b in range(cp.shape[0]):
            out.append({
                "error": float(per[b]),
                "pred": cp[b].numpy(),
                "true": ct[b].numpy(),
                "occ": occ[b].numpy(),
            })
        return out

    # main loop

    def train(self):
        last_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()

            # Resample the recording-wide occlusion, then REBUILD the loader:
            # the number of valid centre frames changes with the field.
            self.train_dataset.set_epoch(epoch)
            train_loader = self._make_loader(self.train_dataset, shuffle=True, drop_last=True)

            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t_train = time.time()
            train_metrics = self.train_one_epoch(epoch, train_loader)
            # Cost columns of T3 / T11: time per epoch and peak training memory.
            train_metrics["epoch_train_time_s"] = time.time() - t_train
            if self.device == "cuda":
                train_metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            val_metrics = self.validate(epoch)
            val_str = 'VAL Metrics ' + str(epoch) + ':'
            for k,v in val_metrics.items():
                val_str += k + '=' + str(v) + ' | '
            self.py_logger.info(val_str)

            self.log_epoch(epoch, train_metrics, val_metrics, time.time() - t0)

            score = val_metrics.get(self.monitor, float("inf"))
            if not math.isfinite(score):
                score = float("inf")
            is_best = score < self.best_score - self.min_delta
            if is_best:
                self.best_score = score
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if is_best:
                self.save_checkpoint(epoch, is_best=True)
            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                self.save_checkpoint(epoch, tag="latest")

            last_epoch = epoch
            if self.early_stopping_enabled and self.epochs_no_improve >= self.patience:
                self.py_logger.info(
                    f"Early stopping at epoch {epoch}: no improvement in {self.monitor} "
                    f"for {self.patience} epochs (best {self.best_score:.5f})."
                )
                break

        self.save_checkpoint(last_epoch, tag="final")
        self.write_report()
        if self.use_wandb:
            wandb.finish()
        self.py_logger.info(f"Training complete. Best {self.monitor} = {self.best_score:.5f}")
        return self.best_score

    def log_epoch(self, epoch, train_metrics, val_metrics, dt):
        for k, v in train_metrics.items():
            self.history["train"][k].append(v)
        for k, v in val_metrics.items():
            self.history["val"][k].append(v)
        lr = self.scheduler.get_last_lr()[0]
        self.history["lr"].append(lr)

        occ = self.train_dataset.epoch_report()
        nan = float("nan")
        msg = (
            f"Epoch {epoch} [{dt:.1f}s] "
            f"train_total={train_metrics.get('total', nan):.4f} "
            f"val_total={val_metrics.get('total', nan):.4f} "
            f"val_mpjpe={val_metrics.get('metric/mpjpe_mm', nan):.4f}mm "
            f"val_p90={val_metrics.get('metric/pjpe_p90_mm', nan):.4f}mm "
        )
        if "metric/frame_yaw_deg" in val_metrics:
            # Printed next to the estimator it is correcting, so a head that is
            # not beating the window frame is visible immediately.
            msg += (
                f"val_yaw={val_metrics['metric/frame_yaw_deg']:.3f}deg"
                f"(base {val_metrics.get('metric/frame_yaw_deg_base', nan):.3f}) "
                f"val_fxy={val_metrics.get('metric/frame_xy_mm', nan):.3f}mm"
                f"(base {val_metrics.get('metric/frame_xy_mm_base', nan):.3f}) "
            )
        if "metric/sigma_err_corr" in val_metrics:
            msg += f"sigma_corr={val_metrics['metric/sigma_err_corr']:.3f} "
        if "peak_mem_gb" in train_metrics:
            msg += f"peak_mem={train_metrics['peak_mem_gb']:.2f}GB "
        msg += (
            f"| hidden={occ.get('hidden_frac', nan):.3f} "
            f"centres={occ.get('n_centers', 0)} | lr={lr:.2e}"
        )
        self.py_logger.info(msg)

        if self.use_wandb:
            log = {"epoch": epoch, "lr": lr, "skipped_steps": self.skipped_steps}
            log.update({f"train/{k}": v for k, v in train_metrics.items()})
            log.update({f"val/{k}": v for k, v in val_metrics.items()})
            log.update({f"data/{k}": v for k, v in occ.items() if isinstance(v, (int, float))})
            log.update({
                f"val_breakdown/{k}": v
                for k, v in self.last_breakdown.items()
                if isinstance(v, (int, float))
            })
            wandb.log(log)

    # checkpoints

    def save_checkpoint(self, epoch, is_best=False, tag=None):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "ema_state": self.ema.state_dict() if self.ema is not None else None,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "best_score": self.best_score,
            "monitor": self.monitor,
            "config": self.config,
            "scale_mm": self.scale_mm,
        }
        path = self.ckpt_dir / ("best.pt" if is_best else f"{tag or f'epoch_{epoch:04d}'}.pt")
        torch.save(state, path)
        self.py_logger.info(f"Saved {'BEST ' if is_best else ''}checkpoint: {path}")

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        if self.ema is not None and state.get("ema_state"):
            self.ema.load_state_dict(state["ema_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        if self.scaler is not None and state.get("scaler_state"):
            self.scaler.load_state_dict(state["scaler_state"])
        self.best_score = state.get("best_score", float("inf"))
        self.global_step = state.get("global_step", 0)
        self.start_epoch = state.get("epoch", -1) + 1
        self.py_logger.info(f"Resumed from {path} at epoch {self.start_epoch}")

    # report

    def write_report(self):
        best = self.ckpt_dir / "best.pt"
        if best.exists():
            state = torch.load(best, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state"])
            if self.ema is not None and state.get("ema_state"):
                self.ema.load_state_dict(state["ema_state"])
        metrics = self.validate(epoch=-1, collect=False)
        report = {
            "monitor": self.monitor,
            "best_score": self.best_score,
            "val_metrics": {k: float(v) for k, v in metrics.items()},
            "breakdown": self.last_breakdown,
            "val_occlusion": self.val_dataset.epoch_report(),
        }
        (self.ckpt_dir / "final_report.json").write_text(json.dumps(report, indent=2, default=str))

        lines = ["| condition | MPJPE (mm) | n |", "|---|---|---|"]
        for k, v in sorted(self.last_breakdown.items()):
            if k.endswith("__n") or k == "per_joint_mm":
                continue
            n = self.last_breakdown.get(f"{k}__n", "")
            lines.append(f"| {k} | {v:.3f} | {n} |")
        lines += ["", "| joint | MPJPE (mm) |", "|---|---|"]
        for k, v in sorted(self.last_breakdown.get("per_joint_mm", {}).items(),
                           key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v:.3f} |")
        (self.ckpt_dir / "final_report.md").write_text("\n".join(lines))
        self.py_logger.info(f"Wrote final report to {self.ckpt_dir}")
        return report

def plot_predictions(epoch, samples, ckpt_dir, edges=None, part_names=None):
    """
    Two rows (top view XY, side view XZ) by three columns (best / median /
    worst). Bones are drawn, because a scatter of 25 dots tells you nothing
    about whether a limb is anatomically possible.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if edges is not None:
        edges = np.asarray(torch.as_tensor(edges).cpu())

    fig, axes = plt.subplots(2, len(samples), figsize=(5 * len(samples), 9), squeeze=False)
    planes = [((0, 1), "top (x, y)"), ((0, 2), "side (x, z)")]

    for col, s in enumerate(samples):
        true, pred, occ = s["true"], s["pred"], s["occ"]
        vis = ~occ
        for row, ((a, b), name) in enumerate(planes):
            ax = axes[row][col]
            if edges is not None:
                for p, c in edges:
                    ax.plot([true[p, a], true[c, a]], [true[p, b], true[c, b]],
                            color="0.75", lw=1.5, zorder=1)
                    ax.plot([pred[p, a], pred[c, a]], [pred[p, b], pred[c, b]],
                            color="tab:red", lw=1.0, alpha=0.6, zorder=2)
            ax.scatter(true[vis, a], true[vis, b], c="tab:blue", marker="o", s=45,
                       label="visible (copied)", zorder=4)
            ax.scatter(true[occ, a], true[occ, b], c="tab:blue", marker="x", s=55,
                       alpha=0.6, label="hidden (GT)", zorder=4)
            ax.scatter(pred[occ, a], pred[occ, b], c="tab:red", marker="^", s=45,
                       label="hidden (pred)", zorder=5)
            for i in np.nonzero(occ)[0]:
                ax.plot([true[i, a], pred[i, a]], [true[i, b], pred[i, b]],
                        ls="--", color="0.4", lw=0.9, zorder=3)
            if row == 1:
                ax.axhline(0.0, color="0.3", lw=0.8, ls=":")
            if part_names is not None and row == 0:
                for i in np.nonzero(occ)[0]:
                    ax.annotate(part_names[i], true[i, [a, b]], fontsize=6, alpha=0.7)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(
                f"{s['label']} - {name}" + (f"  err={s['error']:.4f}" if row == 0 else ""),
                fontsize=10,
            )

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3)
    fig.suptitle(f"Validation reconstructions - epoch {epoch}", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ckpt_dir / f"val_vis_epoch_{epoch:04d}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out