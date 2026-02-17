
import torch
from collections import defaultdict
import numpy as np
class MetricsTracker:
    """
    Tracks loss components and per-point/per-loss metrics.
    Enables analysis of which losses/points contribute most to training.
    """
    def __init__(self, num_keypoints=10, num_views=3):
        self.num_keypoints = num_keypoints
        self.num_views = num_views
        
        # Overall metrics
        self.loss_history = defaultdict(list)  # loss_name -> [values]
        self.lr_history = []
        
        # Per-keypoint metrics (track error for each point)
        self.keypoint_errors = defaultdict(lambda: np.zeros(num_keypoints))
        self.keypoint_counts = np.zeros(num_keypoints)
        
        # Per-view metrics
        self.view_errors = defaultdict(lambda: np.zeros(num_views))
        self.view_counts = np.zeros(num_views)
        
        # Loss magnitude tracking (for adaptive weighting)
        self.loss_magnitudes = defaultdict(list)
        
    def update(self, loss_dict, batch_size=1):
        """
        Update metrics from loss dict.
        
        Args:
            loss_dict: Dict from KeypointLoss.forward() with all loss components
            batch_size: Batch size for averaging
        """
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            
            self.loss_history[loss_name].append(loss_value)
            self.loss_magnitudes[loss_name].append(abs(loss_value))
    
    def update_keypoint_errors(self, pred_2d, gt_2d, masks):
        """
        Track per-keypoint 2D regression errors.
        
        Args:
            pred_2d: [B, V, N, 2]
            gt_2d: [B, V, N, 2]
            masks: [B, V, N, 1]
        """
        masks = ~masks
        # Compute L2 error per point
        error = torch.sqrt(((pred_2d - gt_2d) ** 2).sum(dim=-1))  # [B, V, N]
        
        # Mask invalid points
        error = error * masks.squeeze(-1).float()
        
        # Aggregate per keypoint
        for n in range(self.num_keypoints):
            valid = masks[:, :, n, 0].sum().item()
            if valid > 0:
                self.keypoint_errors['2d_regression'][n] += error[:, :, n].sum().item()
                self.keypoint_counts[n] += valid
    
    def update_view_errors(self, pred, gt, masks, metric_name='2d_regression'):
        """Track per-view metrics."""
        masks = ~masks
        error = torch.abs(pred - gt)
        for v in range(self.num_views):
            valid = masks[:, v].sum().item()
            if valid > 0:
                self.view_errors[metric_name][v] += error[:, v].sum().item()
                self.view_counts[v] += valid
    
    def get_average_metrics(self):
        """Get averaged metrics."""
        avg_metrics = {}
        
        # Average loss components
        for loss_name, values in self.loss_history.items():
            if values:
                avg_metrics[f"loss/{loss_name}"] = np.mean(values)
        
        # # Average keypoint errors
        # for loss_name, errors in self.keypoint_errors.items():
        #     for n in range(self.num_keypoints):
        #         if self.keypoint_counts[n] > 0:
        #             avg_metrics[f"keypoint_error/{loss_name}/pt_{n}"] = \
        #                 errors[n] / self.keypoint_counts[n]
        
        # # Average view errors
        # for metric_name, errors in self.view_errors.items():
        #     for v in range(self.num_views):
        #         if self.view_counts[v] > 0:
        #             avg_metrics[f"view_error/{metric_name}/view_{v}"] = \
        #                 errors[v] / self.view_counts[v]
        
        return avg_metrics
    
    def get_loss_magnitudes(self):
        """Get average magnitude of each loss component."""
        mags = {}
        for loss_name, values in self.loss_magnitudes.items():
            if values:
                mags[loss_name] = np.mean(values)
        return mags
    
    def reset(self):
        """Reset for next epoch."""
        self.loss_history.clear()
        self.keypoint_errors.clear()
        self.keypoint_counts = np.zeros(self.num_keypoints)
        self.view_errors.clear()
        self.view_counts = np.zeros(self.num_views)
        self.loss_magnitudes.clear()

class MetricsEMA:
    """EMA for smoothing loss curves without affecting model weights."""
    
    def __init__(self, decay=0.99):
        self.decay = decay
        self.ema_dict = {}
    
    def update(self, metric_dict):
        """Update EMA with new metrics."""
        for key, value in metric_dict.items():
            if key not in self.ema_dict:
                self.ema_dict[key] = value
            else:
                self.ema_dict[key] = self.decay * self.ema_dict[key] + \
                                     (1 - self.decay) * value
    
    def get_ema_metrics(self):
        return self.ema_dict.copy()
    
    def reset(self):
        self.ema_dict.clear()