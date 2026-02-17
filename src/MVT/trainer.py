import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import wandb
from tqdm import tqdm
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os

class Trainer:

    def __init__(self, model, loss, curriculum, train_metrics, val_metrics, ema, train_dataset, val_dataset, logger, config, device='cuda'):
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.epoch = 0
        self.global_step = 0
        self.visualize_interval = config.get('vizualize_interval', 10)
        self.train_losses = []
        self.val_losses = []

        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.logger = logger
        self.loss_fn = loss
        self.curriculum = curriculum
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.metrics_ema = ema

        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
            )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler
        total_epochs = config.get('epochs', 50)
        warmup_epochs = config.get('warmup_epochs', 5)
        
        warmup_scheduler = LinearLR(self.optimizer, start_factor=config.get('warmup_start_factor', 0.01), total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs,eta_min=config.get('min_learning_rate', 1e-6),)
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        
        
        # WandB
        self.use_wandb = config['logging'].get('use_wandb', False)
        if self.use_wandb and wandb is not None:
            wandb.init(
                project=config['logging'].get('wandb_project', 'MVTransformer'),
                entity=config['logging'].get('wandb_entity', None),
                config=config,
                name=config['logging'].get('run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            )
            self.logger.info("W&B logging enabled")

    def train_epoch(self):
        """Train for one epoch with curriculum learning."""
        self.model.train()
        self.train_metrics.reset()
        loss_weights = self.curriculum.get_loss_weights(self.epoch)
        masking_ratio = self.curriculum.get_masking_ratio(self.epoch)
        curriculum_info = self.curriculum.get_curriculum_info(self.epoch)
        self.train_dataset.set_occlusion(masking_ratio)
        
        self.logger.info(
            f"Epoch {self.epoch + 1} | mask={masking_ratio:.3f} | "
            f"active_losses={curriculum_info['num_active_losses']} | "
            f"weights={loss_weights} | warmup={curriculum_info['warmup_losses']}"
        )
        
        epoch_loss = 0.0
        
        for batch_idx, (keypoints_2d, depths, masks) in enumerate(self.train_loader):
            keypoints_2d = keypoints_2d.to(self.device)
            depths = depths.to(self.device)
            masks = masks.to(self.device).bool()
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(
                keypoints_2d=keypoints_2d,
                occlusion_mask=masks,
                depth_map=depths
            )
            pred_2d = output.get('coordinates', None)
            pred_depths = output.get('depth', None)
            confidence = output.get('confidence', None)

            # Compute loss with curriculum weights
            loss_dict = self.loss_fn(
                keypoints_2d_pred=pred_2d,
                keypoints_2d_gt=keypoints_2d,
                masks=masks,
                depths_pred=pred_depths,
                depths_gt=depths,
                confidence_pred=confidence,
            )
            
            # Apply curriculum weighting
            total_loss = 0.0
            weighted_losses = {}
            for loss_name, loss_value in loss_dict.items():
                if loss_name == 'total':
                    continue
                curriculum_weight = loss_weights.get(loss_name, 0.0)
                weighted_loss = loss_value * curriculum_weight
                total_loss += weighted_loss
                weighted_losses[loss_name] = loss_value.item()

            # Backward pass
            total_loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()

            # Update metrics and accumulate loss
            self.train_metrics.update(loss_dict, batch_size=keypoints_2d.size(0))
            epoch_loss += total_loss.item()
            self.global_step += 1
            
            # Batch logging (every log_interval)
            if (batch_idx + 1) % self.config.get('log_interval', 10) == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                loss_str = ", ".join([f"{k}: {v:.6f}" for k, v in weighted_losses.items()])
                self.logger.info(
                    f"[{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={avg_loss:.7f} | {loss_str} | grad_norm={total_norm:.4f}"
                )
            
            # Per-batch WandB logging (optional)
            if self.use_wandb and self.config.get('log_per_batch', False):
                wandb.log({
                    'train/batch_loss': total_loss.item(),
                    **{f'train/batch/{k}': v for k, v in weighted_losses.items()},
                    'global_step': self.global_step,
                }, step=self.global_step)

        self.scheduler.step()
        
        # Final epoch summary
        epoch_metrics = self.train_metrics.get_average_metrics()
        epoch_avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(epoch_avg_loss)
        self.metrics_ema.update(epoch_metrics)
        ema_metrics = self.metrics_ema.get_ema_metrics()
        
        self._log_epoch(
            mode='train', 
            metrics=epoch_metrics, 
            ema_metrics=ema_metrics,
            curriculum_info=curriculum_info,
            loss_weights=loss_weights,
            masking_ratio=masking_ratio
        )
        
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"Epoch {self.epoch + 1} done | train_loss={epoch_avg_loss:.6f} | LR={lr:.2e}")
        return epoch_avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        if self.val_loader is None:
            self.logger.warning("No validation dataset provided")
            return None
        
        self.model.eval()
        self.val_metrics.reset()
        
        loss_weights = self.curriculum.get_loss_weights(self.epoch)
        
        total_val_loss = 0.0
        num_batches = 0
        visualize = hasattr(self, 'visualize_interval') and (self.epoch + 1) % self.visualize_interval == 0
        if visualize:
            vis_examples = []

        for batch_idx, (keypoints_2d, depths, masks) in enumerate(self.val_loader):
            keypoints_2d = keypoints_2d.to(self.device)
            depths = depths.to(self.device)
            masks = masks.to(self.device).bool()
            
            output = self.model(keypoints_2d=keypoints_2d, occlusion_mask=masks, depth_map=depths)
            pred_2d = output.get('coordinates', None)
            pred_depths = output.get('depth', None)
            confidence = output.get('confidence', None)

            if visualize and len(vis_examples) < 3 and batch_idx == 0:
                B = keypoints_2d.size(0)
                for i in range(min(3, B)):
                    vis_examples.append({
                        "keypoints_gt": keypoints_2d[i].detach().cpu(),
                        "keypoints_pred": pred_2d[i].detach().cpu() if pred_2d is not None else None,
                        "mask": masks[i].squeeze(-1).detach().cpu().bool(),
                    })

                   
            loss_dict = self.loss_fn(
                keypoints_2d_pred=pred_2d,
                keypoints_2d_gt=keypoints_2d,
                masks=masks,
                depths_pred=pred_depths,
                depths_gt=depths,
                confidence_pred=confidence,
            )
            
            # Compute weighted validation loss
            batch_loss = 0.0
            for loss_name, loss_value in loss_dict.items():
                if loss_name == 'total':
                    continue
                curriculum_weight = loss_weights.get(loss_name, 0.0)
                batch_loss += loss_value * curriculum_weight
            
            self.val_metrics.update(loss_dict, batch_size=keypoints_2d.size(0))
            total_val_loss += batch_loss.item()
            num_batches += 1
            
        # Create visualization plots
        if visualize and vis_examples:
            self._create_prediction_plot(vis_examples)

        # Final validation metrics
        epoch_metrics = self.val_metrics.get_average_metrics()
        val_loss = total_val_loss / num_batches
        self.val_losses.append(val_loss)
        epoch_metrics['loss/total'] = val_loss

        self._log_epoch(mode='val', metrics=epoch_metrics)
        self.logger.info(f"Epoch {self.epoch + 1} | val_loss={val_loss:.6f}")
        
        if hasattr(self, 'best_val_loss') and self.use_wandb:
            self.logger.info(f"Best val loss: {self.best_val_loss:.6f}")
        
        return val_loss

    def _create_prediction_plot(self, examples):
        """
        3x3 grid: 3 examples (rows) x 3 views (columns).
        - Blue: Visible GT
        - Green: Occluded GT
        - Red: Predicted Occluded
        - Dashed Line: Connects Green (GT) to Red (Pred) to show error
        
        Assumes shapes: [V, N, 2] for keypoints and [V, N] for mask
        """
        num_examples = min(len(examples), 3)
        num_views = 3

        fig, axes = plt.subplots(num_examples, num_views, figsize=(15, 5 * num_examples))
        
        # Handle single example case to ensure axes is always 2D
        if num_examples == 1:
            axes = np.array(axes).reshape(1, -1)
        elif num_examples == 0:
            return # Safety check

        # 1. Collect all coordinates to set consistent axis limits
        all_x, all_y = [], []
        for ex in examples[:num_examples]:
            gt = ex["keypoints_gt"].numpy()    # [V, N, 2]
            pred = ex["keypoints_pred"]        
            if pred is not None: 
                pred = pred.numpy()            # [V, N, 2]
                all_x.extend(pred[..., 0].flatten())
                all_y.extend(pred[..., 1].flatten())
            
            all_x.extend(gt[..., 0].flatten())
            all_y.extend(gt[..., 1].flatten())

        if not all_x: return # Safety if empty

        # Compute global limits with padding
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        x_padding = (x_max - x_min) * 0.1 if x_max > x_min else 0.1
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1

        # 2. Plotting Loop
        for row_idx, ex in enumerate(examples[:num_examples]):
            gt = ex["keypoints_gt"].numpy()               # [V, N, 2]
            pred = ex["keypoints_pred"]                   
            pred = pred.numpy() if pred is not None else None
            mask = ex["mask"].numpy().astype(bool)        # [V, N] (True=Visible)

            V, N, _ = gt.shape

            for view_idx in range(num_views):
                ax = axes[row_idx, view_idx]
                
                # Handle cases with fewer than 3 views in data
                if view_idx >= V:
                    ax.axis('off')
                    continue

                # --- Extract View Data ---
                gt_view = gt[view_idx, :, :]              # [N, 2]
                mask_view = mask[view_idx, :]             # [N]
                pred_view = pred[view_idx, :, :] if pred is not None else None # [N, 2]

                # --- Identify Indices ---
                # np.where returns a tuple, take [0] to get the array of indices
                visible_indices = np.where(mask_view)[0]
                occluded_indices = np.where(~mask_view)[0]

                # --- Plot Points ---
                
                # 1. Visible GT (Blue)
                if len(visible_indices) > 0:
                    vis_pts = gt_view[visible_indices]
                    ax.scatter(vis_pts[:, 0], vis_pts[:, 1], 
                            c='blue', s=50, alpha=1.0, label='Visible (GT)', zorder=3)

                # 2. Occluded GT (Green)
                if len(occluded_indices) > 0:
                    occ_pts = gt_view[occluded_indices]
                    ax.scatter(occ_pts[:, 0], occ_pts[:, 1], 
                            c='green', s=50, alpha=0.5, label='Occluded (GT)', zorder=2)

                # 3. Predicted Occluded (Red) & Error Lines
                if pred_view is not None and len(occluded_indices) > 0:
                    pred_occ_pts = pred_view[occluded_indices]
                    
                    # Plot Red Dots
                    ax.scatter(pred_occ_pts[:, 0], pred_occ_pts[:, 1], 
                            c='red', s=50, alpha=0.5, label='Predicted', zorder=2)
                    
                    # Plot Dashed Lines (Connect Green -> Red)
                    for i, k_idx in enumerate(occluded_indices):
                        # Get specific points for this keypoint index
                        start = gt_view[k_idx]      # Green point
                        end = pred_view[k_idx]      # Red point
                        
                        ax.plot([start[0], end[0]], [start[1], end[1]], 
                                color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

                # --- Formatting ---
                ax.set_xlim(x_min - x_padding, x_max + x_padding)
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
                ax.set_aspect('equal')
                ax.invert_yaxis()
                ax.set_title(f'Ex {row_idx + 1} | View {view_idx + 1}', fontsize=10)
                ax.grid(True, alpha=0.3)

                # Legend on first plot only
                if row_idx == 0 and view_idx == 0:
                    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        plt.tight_layout()

        # Save
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        save_path = os.path.join(self.checkpoint_dir, f'predictions_epoch_{self.epoch + 1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"Saved prediction plot to {save_path}")



    def _log_epoch(self, mode='train', metrics=None, ema_metrics=None, 
                curriculum_info=None, loss_weights=None, masking_ratio=None):
        """Log metrics to console and W&B with consistent organization."""
        if metrics is None:
            return
        
        if self.use_wandb and wandb is not None:
            log_dict = {}
            loss_metrics = {k: v for k, v in metrics.items() if k.startswith('loss/')}
            other_metrics = {k: v for k, v in metrics.items() if not k.startswith('loss/')}

            log_dict.update({f"{mode}/{k}": v for k, v in loss_metrics.items()})
            log_dict.update({f"{mode}/{k}": v for k, v in other_metrics.items()})
            
            # EMA metrics
            if ema_metrics:
                log_dict.update({f"{mode}_ema/{k}": v for k, v in ema_metrics.items()})
            
            # Curriculum (train only)
            if mode == 'train' and curriculum_info:
                log_dict['curriculum/masking_ratio'] = masking_ratio
                log_dict['curriculum/num_active_losses'] = curriculum_info['num_active_losses']
                if loss_weights:
                    log_dict.update({f'curriculum/weight_{k}': v for k, v in loss_weights.items()})
            
            # Standard metadata
            log_dict['epoch'] = self.epoch
            log_dict['lr'] = self.optimizer.param_groups[0]['lr']
            
            wandb.log(log_dict, step=self.epoch)

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        best_path = self.checkpoint_dir / "best_model.pt"
        
        state = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if is_best:
            torch.save(state, best_path)
            self.logger.info(f"✓ Saved best model to {best_path}")
        else:
            torch.save(state, checkpoint_path)
            self.logger.info(f"✓ Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load from checkpoint."""
        state = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.epoch = state['epoch']
        self.best_val_loss = state['best_val_loss']
        
        self.logger.info(f"✓ Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """
        Full training loop with early stopping.
        """
        # Seed generation
        seed = random.randint(0, 2**32 - 1)
        self._set_seed(seed)
        seed_file = self.checkpoint_dir / "seed.json"
        with open(seed_file, 'w') as f:
            json.dump({"seed": seed}, f, indent=2)
        self.logger.info(f"Random seed {seed} saved to {seed_file}")
        if self.use_wandb and wandb is not None:
            wandb.config.update({"seed": seed}, allow_val_change=True)

        num_epochs = self.config.get('epochs', 100)
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.val_dataset.set_occlusion(0.3)
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            val_loss = None
            if self.val_loader is not None:
                val_loss = self.validate()
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.save_checkpoint(is_best=True)
                    wandb.run.summary["best_val_loss"] = self.best_val_loss
                    wandb.run.summary["best_val_epoch"] = self.best_epoch
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= self.patience:
                        self.logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(best epoch: {self.best_epoch})"
                        )
                        break
            
            # Regular checkpoint save
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint()
        
        self.logger.info(f"Training complete. Best epoch: {self.best_epoch}")
        
        if self.use_wandb and wandb is not None:
            wandb.finish()
    
    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_metrics_summary(self):
        """Get summary statistics."""
        return {
            'total_epochs_trained': self.epoch + 1,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'train_loss_history': self.train_losses,
            'val_loss_history': self.val_losses,
        }
