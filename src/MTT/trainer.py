import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

import numpy as np
from torch.utils.data import DataLoader

class Trainer:
    """
    Generic trainer for DeformationCorrector-style models.

    Expected batch format from dataloaders:
        batch = {
            "d_noisy":   (B, N, 3)   noisy deformations at t+1
            "d_history": (B, N, T, 3) history deformations t-T+1..t
            "d_gt":      (B, N, 3)   ground-truth deformations at t+1
        }

    """

    def __init__(self, model, train_dataset, val_dataset, optimizer, loss_fn, scheduler, batch_size = 32, device = "cuda", output_dir = "./outputs", log_wandb = True, patience = 10, grad_clip = 1.0, min_delta = 0.0, joint_edges = None):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.loss_fn = loss_fn
        self.patience = patience
        self.min_delta = min_delta
        self.joint_edges = joint_edges  # for skeleton plotting
        self.grad_clip = grad_clip

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0

    def train(self, num_epochs):
        """
        Main training loop with validation, early stopping, and checkpointing.
        """
        for epoch in range(1, num_epochs + 1):
            d_idx = torch.randint(0, 10, (1,)).item()
            self.train_dataset.set_current_pred(d_idx)
            train_loss_dict, grad_norm = self._train_one_epoch()
            val_loss_dict = self._validate(epoch)
            train_loss = train_loss_dict['total']
            val_loss = val_loss_dict['total']
            self.scheduler.step()

            # Check for improvement
            if val_loss + self.min_delta < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
            print(f"Epoch {epoch:3d}:\nTrain: {"| ".join([f"{k}: {v:.5f}" for k, v in train_loss_dict.items()])}\nValidation:{"| ".join([f"{k}: {v:.5f}" for k, v in val_loss_dict.items()])} \nGrad norm:{grad_norm:.5f} | Dataset index:{d_idx} | Best epoch:{self.best_epoch} ")
            # Save "last" checkpoint every epoch
            self._save_checkpoint(epoch, is_best=False)

            if self.log_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/coord_loss":train_loss_dict['coord'],
                        "train/skeleton_loss":train_loss_dict['skeleton'],
                        "train/velocity_loss":train_loss_dict['velocity'],
                        "val/loss": val_loss,
                        "val/coord_loss": train_loss_dict['coord'],
                        "val/skeleton_loss": train_loss_dict['skeleton'],
                        "val/velocity_loss": train_loss_dict['velocity'],
                        "train/grad_norm": grad_norm,
                        "train/lr": self.optimizer.param_groups[0]['lr'],
                        "early_stopping/epochs_without_improvement": self.epochs_without_improvement,
                        "early_stopping/best_epoch": self.best_epoch,
                        "early_stopping/best_val_loss": self.best_val_loss,
                    }
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(
                    f"Early stopping triggered at epoch {epoch}, "
                    f"best epoch was {self.best_epoch} with val_loss={self.best_val_loss:.6f}"
                )
                break

        print(
            f"Training finished. Best epoch: {self.best_epoch}, "
            f"best val_loss={self.best_val_loss:.6f}"
        )

    def _train_one_epoch(self):
        self.model.train()
        total_norm = 0.0
        n_batches = 0
        weighted_losses = {'total':0.0,
                           'coord':0.0,
                           'skeleton':0.0,
                           'velocity':0.0}
        for batch in self.train_loader:
            d_noisy, d_history, d_gt = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()
            d_corrected = self.model(d_noisy, d_history)
            loss_dict = self.loss_fn(d_corrected, d_gt)
            loss = loss_dict['total']
            loss.backward()
            total_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            self.optimizer.step()

            for loss_name, loss_value in loss_dict.items():
                weighted_losses[loss_name] += loss_value
            n_batches += 1
        
        for loss_name, loss_value in loss_dict.items():
            weighted_losses[loss_name] /= max(1, n_batches)
        avg_norm = total_norm / max(1, n_batches)
        return weighted_losses, avg_norm

    def _validate(self, epoch):
        self.model.eval()
        n_batches = 0
        weighted_losses = {'total':0.0,
                           'coord':0.0,
                           'skeleton':0.0,
                           'velocity':0.0}

        first_batch_for_plot = None

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                d_noisy, d_history, d_gt = self._move_batch_to_device(batch)

                d_corrected = self.model(d_noisy, d_history)
                loss_dict = self.loss_fn(d_corrected, d_gt)
                n_batches += 1

                for loss_name, loss_value in loss_dict.items():
                    weighted_losses[loss_name] += loss_value

                if first_batch_for_plot is None:
                    first_batch_for_plot = (
                        d_noisy.detach().cpu(),
                        d_corrected.detach().cpu(),
                        d_gt.detach().cpu(),
                    )
        # Plot predictions for qualitative monitoring
        if epoch%10==0 and first_batch_for_plot is not None:
            self._plot_predictions(first_batch_for_plot, epoch)

        for loss_name, loss_value in loss_dict.items():
            weighted_losses[loss_name] /= max(1, n_batches)
        return weighted_losses

    def _move_batch_to_device(self, batch):
        # Expecting dict-like batch with specific keys.
        d_noisy = batch["input_aligned_dist"].squeeze(1).to(self.device)    # (B, N, 3)
        d_history = batch["hist_aligned_dist"].to(self.device)   # (B, N, T, 3)
        d_gt = batch["target_aligned_dist"].squeeze(1).to(self.device)      # (B, N, 3)
        return d_noisy, d_history, d_gt

    def _save_checkpoint(self, epoch: int, is_best: bool):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state"] = self.scheduler.state_dict()

        # Last checkpoint
        last_path = os.path.join(self.output_dir, "checkpoints", "last.pt")
        torch.save(ckpt, last_path)

        # Best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoints", "best.pt")
            torch.save(ckpt, best_path)

    def _plot_predictions(self, batch_data, epoch):
        d_noisy, d_corrected, d_gt = batch_data
        d_noisy_np = d_noisy.cpu().numpy()     # (B, N, 3)
        d_corr_np = d_corrected.cpu().numpy()  # (B, N, 3)
        d_gt_np = d_gt.cpu().numpy()           # (B, N, 3)
        
        plot_path = os.path.join(self.output_dir, "plots", f"val_predictions_epoch_{epoch:04d}.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        
        # Plot best and worst
        plot_best_and_worst_predictions(
            d_noisy_np, 
            d_corr_np, 
            d_gt_np, 
            epoch=epoch,
            use_3d=False,  # Switch to True if you want Z-axis depth
            save_path=plot_path
        )

        if self.log_wandb:
            import wandb
            wandb.log({"val/predictions_plot": wandb.Image(plot_path), "epoch": epoch})

def plot_best_and_worst_predictions(d_noisy_batch, d_pred_batch, d_gt_batch, epoch, use_3d=False, save_path=None):
    """
    Finds and plots the best and worst predictions in a batch based on MSE.
    
    Args:
        d_noisy_batch: numpy array (B, N, 3)
        d_pred_batch: numpy array (B, N, 3) 
        d_gt_batch: numpy array (B, N, 3)
        epoch: int - Current epoch for the title
        use_3d: bool - Plot in 3D if True, else 2D (XY plane)
        save_path: str - Path to save the figure
    """
    # 1. Calculate MSE per sample to find best and worst
    # Average across joints (axis=1) and coordinates (axis=2)
    mse_per_sample = np.mean((d_pred_batch - d_gt_batch)**2, axis=(1, 2))
    
    best_idx = np.argmin(mse_per_sample)
    worst_idx = np.argmax(mse_per_sample)
    
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Epoch {epoch}: Best vs Worst Validation Predictions", fontsize=16)
    
    # 2. Helper function to plot a single sample on a given matplotlib Axis
    def _plot_on_ax(ax, nx_pts, pr_pts, gt_pts, title):
        N = nx_pts.shape[0]
        cmap = plt.get_cmap('tab10')
        
        # Plot Anchor at Origin
        if use_3d:
            ax.scatter(0, 0, 0, c='black', marker='P', s=200, label='Anchor', zorder=5)
        else:
            ax.scatter(0, 0, c='black', marker='P', s=200, label='Anchor', zorder=5)

        for i in range(N):
            color = cmap(i % 10)
            nx, ny, nz = nx_pts[i]
            px, py, pz = pr_pts[i]
            gx, gy, gz = gt_pts[i]
            
            if use_3d:
                ax.scatter(nx, ny, nz, c=[color], marker='x', s=60, alpha=0.6, label='Noisy Input' if i==0 else "")
                ax.scatter(px, py, pz, c=[color], marker='o', s=60, label='Denoised Pred' if i==0 else "")
                ax.scatter(gx, gy, gz, c=[color], marker='*', s=120, edgecolor='black', linewidth=0.5, label='Ground Truth' if i==0 else "")
                
                ax.plot([nx, px], [ny, py], [nz, pz], color=color, linestyle='-', alpha=0.7)
                ax.plot([px, gx], [py, gy], [pz, gz], color=color, linestyle=':', alpha=0.4)
            else:
                ax.scatter(nx, ny, c=[color], marker='x', s=60, alpha=0.6, label='Noisy Input' if i==0 else "")
                ax.scatter(px, py, c=[color], marker='o', s=60, label='Denoised Pred' if i==0 else "")
                ax.scatter(gx, gy, c=[color], marker='*', s=120, edgecolor='black', linewidth=0.5, label='Ground Truth' if i==0 else "")
                
                ax.annotate('', xy=(px, py), xytext=(nx, ny),
                            arrowprops=dict(arrowstyle="->", color=color, alpha=0.8, lw=1.5))
                ax.plot([px, gx], [py, gy], color=color, linestyle=':', alpha=0.4)

        ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        if use_3d:
            ax.set_zlabel('Z Coordinate')
        else:
            ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, linestyle='--', alpha=0.5)

    # 3. Create subplots and plot
    ax1 = fig.add_subplot(121, projection='3d' if use_3d else None)
    _plot_on_ax(ax1, d_noisy_batch[best_idx], d_pred_batch[best_idx], d_gt_batch[best_idx], 
                title=f"BEST Prediction (MSE: {mse_per_sample[best_idx]:.5f})")
                
    ax2 = fig.add_subplot(122, projection='3d' if use_3d else None)
    _plot_on_ax(ax2, d_noisy_batch[worst_idx], d_pred_batch[worst_idx], d_gt_batch[worst_idx], 
                title=f"WORST Prediction (MSE: {mse_per_sample[worst_idx]:.5f})")

    # 4. Global Legend and Save
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return fig
