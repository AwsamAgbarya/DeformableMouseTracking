import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.geometry import project_points, unproject_points, triangulate_points

class KeypointLoss(nn.Module):

    def __init__(self, projection_matrices, normalizer, loss_config = None, device= 'cuda'):
        """
        Args:
            projection_matrices: Camera projection matrices
            loss: Dictionary with loss component names as keys and config dicts as values.
                        Each config dict can have 'enabled' (bool) and 'weight' (float).
                        Example: {'depth': {'enabled': True, 'weight': 0.5}}
            huber_delta: Delta parameter for Huber loss
            confidence_loss_type: Type of confidence loss
            device: Device for computation
        """
        super().__init__()
        self.huber_delta = loss_config['huber_delta']
        # Default configuration
        default_config = {
            '2d': {'enabled': True, 'weight': 1.0},
            'depth': {'enabled': True, 'weight': 0.5},
            'confidence': {'enabled': False, 'weight': 0.3, 'loss_type': 'expectation_mismatch'},
            'diff_reprojection': {'enabled': False, 'weight': 1.0},
            'cycle_consistency': {'enabled': False, 'weight': 1.0},
            'triangulation_residual': {'enabled': False, 'weight': 1.0},
            'triangulation_diff': {'enabled': False, 'weight': 1.0},
        }
        
        # Merge user config with defaults
        if loss_config['components'] is not None:
            for key, value in loss_config['components'].items():
                if key in default_config:
                    default_config[key].update(value)
        
        self.loss_config = default_config
        print(f"Loss config:\n{self.loss_config}")
        self.num_views = projection_matrices.shape[0]
        self.confidence_loss_type = self.loss_config['confidence']['loss_type']
        self.device = device
        
        # Huber loss for robustness
        self.huber = nn.HuberLoss(delta=self.huber_delta, reduction='none')

        self.projection = projection_matrices.to(device)
        self.normalizer = normalizer

    
    def forward(self, keypoints_2d_pred, keypoints_2d_gt, masks, depths_pred = None, depths_gt = None, confidence_pred = None):
        """Compute combined keypoint loss."""
        losses = {}
        masks = ~masks # masks should indicate True = occluded

        # 2D Keypoint Loss
        loss_2d, kpt_point_loss = self.compute_2d_loss(keypoints_2d_pred, keypoints_2d_gt, masks)
        losses['2d'] = loss_2d

        if self.loss_config['depth']['enabled']:
            # Depth Loss
            if depths_pred is not None:
                depths_pred = depths_pred.unsqueeze(-1)
                loss_depth, depth_point_loss = self.compute_depth_loss(depths_pred, depths_gt, masks)
                losses['depth'] = loss_depth

        if self.loss_config['confidence']['enabled']:
            # Confidence Loss
            if confidence_pred is not None:
                loss_conf = self.compute_confidence_loss(confidence_pred, masks, kpt_point_loss, depth_point_loss)
                losses['confidence'] = loss_conf
        
        if self.loss_config['triangulation_residual']['enabled']:
            # Triangulation Residual Loss
            loss_tri_residual = self.triangulation_residual_loss(keypoints_2d_pred)
            losses['triangulation_residual'] = loss_tri_residual

        if self.loss_config['triangulation_diff']['enabled']:
            # Triangulation error
            loss_tri = self.compute_differential_triangulation_loss(keypoints_2d_pred,keypoints_2d_gt)
            losses['triangulation_diff'] = loss_tri

        if self.loss_config['diff_reprojection']['enabled']:
            # Differential Reprojection Loss
            if depths_pred is not None:
                loss_diff, loss_depth_diff = self.compute_relative_reprojection_loss(keypoints_2d_pred, depths_pred, keypoints_2d_gt, depths_gt, masks)
                losses['diff_coord'] = loss_diff
                losses['depth_diff'] = loss_depth_diff

        if self.loss_config['cycle_consistency']['enabled']:
            # Cycle Consistency
            if depths_pred is not None:
                cycle_coord, cycle_depth = self.compute_cycle_consistency_loss(keypoints_2d_pred, depths_pred)
                losses['cycle_coord'] = cycle_coord
                losses['cycle_depth'] = cycle_depth

        return losses
        
    def compute_2d_loss(self, keypoints_2d_pred, keypoints_2d_gt, masks):
        """Compute 2D keypoint regression loss (MSE/Huber)."""
        B, V, N, _ = keypoints_2d_pred.shape
        pred = keypoints_2d_pred.view(-1, N, 2) # [B*C, N, 2]
        gt = keypoints_2d_gt.view(-1, N, 2)
        mask = masks.view(-1, masks.shape[2], 1)

        loss = self.huber(pred, gt).mean(dim=-1, keepdim=True)
        masked_loss = (loss * mask)
        num_occluded = mask.sum()

        return masked_loss.sum() / num_occluded, masked_loss.sum(dim=-1) / (mask.squeeze(-1) + 1e-8)
    
    def compute_depth_loss(self, depths_pred, depths_gt, masks):
        """Compute depth regression loss."""
        pred = depths_pred.view(-1, depths_pred.shape[2]) # [B * C, N]
        gt = depths_gt.view(-1, depths_gt.shape[2])
        mask = masks.view(-1, masks.shape[2])
        num_occluded = mask.sum()

        loss =  F.smooth_l1_loss(pred * mask, gt * mask, reduction='none')
        return loss.sum() / num_occluded, loss
    
    def compute_confidence_loss(self, confidence_pred, masks, kpt_error, depth_error):
        """Confidence calibration loss."""
        conf = confidence_pred.view(-1, confidence_pred.shape[2])
        mask = masks.view(-1, masks.shape[2])
        num_occluded = mask.sum()
        total_error = kpt_error + 0.1 * depth_error
        
        if self.confidence_loss_type == 'expectation_mismatch':
            conf_loss = torch.abs(conf - (1.0 / (1.0 + total_error.detach())))
        else:
            conf_loss = conf * total_error + (1 - conf) * torch.log(1 + total_error)
        
        return (conf_loss * mask).sum() / num_occluded
    
    
    def triangulation_residual_loss(self, keypoints_2d_pred):
        """
        Triangulation Residual (TR) Loss:
        For each joint j, build the triangulation matrix A_j from all views and
        minimize its smallest singular value sigma_min(A_j).

        Args:
            keypoints_2d_pred: [B, V, N, 2]
        Returns:
            scalar TR loss
        """
        B, V, N, _ = keypoints_2d_pred.shape
        # Requires denormalized keypoints
        keypoints_2d_pred = self.normalizer.denormalize(keypoints_2d_pred)
        u = keypoints_2d_pred[..., 0:1]   # [B, V, N, 1]
        v = keypoints_2d_pred[..., 1:2]   # [B, V, N, 1]

        # Expand P per point: [B, V, N, 3, 4]
        P = self.projection.unsqueeze(0).unsqueeze(2).expand(-1, -1, N, -1, -1)
        P0 = P[..., 0, :]
        P1 = P[..., 1, :]
        P2 = P[..., 2, :]

        A_u = u * P2 - P0 
        A_v = v * P2 - P1
        A = torch.stack([A_u, A_v], dim=3)  # [B, V, N, 2, 4]

        # Collapse views and the 2 rows into a 2V x 4 matrix per (B, N)
        A = A.permute(0, 2, 1, 3, 4).reshape(B, N, 2 * V, 4)
        A_flat = A.reshape(B * N, 2 * V, 4)

        # SVD: A = U S Vh, smallest singular value is last in S
        _, S, _ = torch.linalg.svd(A_flat)  # S: [(B*N), 4]

        sigma_min = S[..., -1]              # [(B*N)]
        sigma_min = sigma_min.reshape(B, N) # [B, N]
        return torch.clamp(sigma_min, min=1e-4, max=10.0).mean()
    
    def compute_absolute_reprojection_loss(self, keypoints_2d_pred, depths_pred, keypoints_2d_gt, depths_gt, masks):
        """
        Reprojection loss: project pred to 3D then back to 2D and compare with GT 2D.
        This enforces that predicted keypoints+depths, when projected back, should match ground truth.
        """
        total_coord = 0.0
        total_depth = 0.0
        count = 0
        kpts_3d_pred = unproject_points(keypoints_2d_pred, depths_pred, self.projection.unsqueeze(0)) # [B, V, N, 3]
        for ref_view in range(self.num_views):
            mask_ref = masks[:, ref_view]
            num_occluded = mask_ref.sum()
            X_ref = kpts_3d_pred[:, ref_view]
            for target_view in range(self.num_views):
                if target_view == ref_view:
                    continue
                rprj_pred, rprj_depth = project_points(X_ref, self.projection[:, target_view]) # [B, N, 2] [B, N]

                gt_2d_t = keypoints_2d_gt[:, target_view]  # [B, N, 2]
                gt_z_t  = depths_gt[:, target_view]      # or depth_gt if you have it

                # Huber per-pixel/point
                loss_coord = self.huber(rprj_pred, gt_2d_t)           # [B, N, 2] or [B,N]
                loss_depth = self.huber(rprj_depth[...,None], gt_z_t)
                loss_coord = (loss_coord * mask_ref).sum()/num_occluded
                loss_depth = (loss_depth * mask_ref).sum()/num_occluded

                total_coord += loss_coord
                total_depth += loss_depth
                count += 1

        loss_coord = total_coord / max(count, 1)
        loss_depth = total_depth / max(count, 1)

        return loss_coord, loss_depth
    
    def compute_relative_reprojection_loss(self, keypoints_2d_pred, depths_pred, keypoints_2d_gt, depths_gt, masks):
        """
        Differential reprojection (robustness)
        """
        total_coord = 0.0
        total_depth = 0.0
        count = 0
        kpts_3d_pred = unproject_points(keypoints_2d_pred, depths_pred, self.projection.unsqueeze(0)) # [B, V, N, 3]
        kpts_3d_gt = unproject_points(keypoints_2d_gt, depths_gt, self.projection.unsqueeze(0)) # [B, V, N, 3]
        for ref_view in range(self.num_views):
            # [B, N, 1] mask for joints predicted in ref_view
            mask_ref = masks[:, ref_view]   # [B, N, 1]
            num_occluded = mask_ref.sum()
            X_pred_ref = kpts_3d_pred[:, ref_view]        # [B, N, 3]
            X_gt_ref   = kpts_3d_gt[:, ref_view]          # [B, N, 3]

            for target_view in range(self.num_views):
                if target_view == ref_view:
                    continue

                P_t = self.projection[:, target_view]          # [B, 3, 4]

                gt_2d_t = keypoints_2d_gt[:, target_view]  # [B, N, 2]
                gt_z_t  = depths_gt[:, target_view]        # [B, N]

                # Reproject pred and GT 3D from ref into target
                pred_2d_t, pred_z_t = project_points(X_pred_ref, P_t)  # [B,N,2],[B,N]
                gt_2d_proj_t, gt_z_proj_t = project_points(X_gt_ref, P_t)

                # Absolute reprojection residuals (per joint)
                coord_res_pred = self.huber(pred_2d_t, gt_2d_t)
                coord_res_gt   = self.huber(gt_2d_proj_t, gt_2d_t)

                depth_res_pred = self.huber(pred_z_t[...,None], gt_z_t)          # [B, N]
                depth_res_gt   = self.huber(gt_z_proj_t[...,None], gt_z_t)       # [B, N]

                # Relative residuals: difference of errors
                coord_rel = (coord_res_pred - coord_res_gt) * mask_ref       # [B,N,2]
                depth_rel = (depth_res_pred - depth_res_gt) * mask_ref  # [B,N]

                # Aggregate (mean over B,N and coord dims)
                loss_coord = coord_rel.abs().sum()/num_occluded
                loss_depth = depth_rel.abs().sum()/num_occluded

                total_coord += loss_coord
                total_depth += loss_depth
                count += 1

        loss_coord_rel = total_coord / max(count, 1)
        loss_depth_rel = total_depth / max(count, 1)

        return loss_coord_rel, loss_depth_rel

    def compute_cycle_consistency_loss(self, keypoints_2d_pred, depths_pred):
        """
        Vectorized cycle consistency: ref → target → back to ref
        Enforces round-trip consistency for all view pairs (excluding self-pairs)
        """
        B, V, N, _ = keypoints_2d_pred.shape
        # Precompute 3D from all views (world coordinates)
        kpts_3d_pred = unproject_points(keypoints_2d_pred, depths_pred, self.projection.unsqueeze(0))

        # Expand all quantities for pairwise comparisons
        X_ref = kpts_3d_pred.unsqueeze(2).expand(-1, -1, V, -1, -1)  # [B, V, V, N, 3]
        u_ref = keypoints_2d_pred.unsqueeze(2).expand(-1, -1, V, -1, -1)  # [B, V, V, N, 2]
        z_ref = depths_pred.unsqueeze(2).expand(-1, -1, V, -1, -1)  # [B, V, V, N, 1]
        
        P_ref = self.projection.unsqueeze(0).unsqueeze(2).expand(B, -1, V, -1, -1)
        P_target = self.projection.unsqueeze(0).unsqueeze(1).expand(B, V, -1, -1, -1)

        # Reshape to treat (V_ref, V_target) as batch dimension
        BVV = B * V * V
        X_ref_flat = X_ref.reshape(BVV, N, 3)
        P_target_flat = P_target.reshape(BVV, 3, 4)
        P_ref_flat = P_ref.reshape(BVV, 3, 4)
        
        # 1) Forward: ref 3D → target 2D+depth
        u_t, z_t = project_points(X_ref_flat, P_target_flat)
        # Reshape back for unproject
        u_t = u_t.reshape(B, V, V, N, 2)
        z_t = z_t.reshape(B, V, V, N).unsqueeze(-1)
        
        # 2) Backward: target 2D+depth → 3D (world)
        u_t_unproj = u_t.reshape(B * V, V, N, 2)
        z_t_unproj = z_t.reshape(B * V, V, N, 1)
        P_unproj = self.projection.unsqueeze(0).expand(B * V, -1, -1, -1)

        
        X_from_t = unproject_points(u_t_unproj, z_t_unproj, P_unproj)
        X_from_t = X_from_t.reshape(B, V, V, N, 3)
        
        # 3) Project back to ref view
        X_from_t_flat = X_from_t.reshape(BVV, N, 3)
        u_back, z_back = project_points(X_from_t_flat, P_ref_flat)
        
        u_back = u_back.reshape(B, V, V, N, 2)
        z_back = z_back.reshape(B, V, V, N).unsqueeze(-1)
        
        # 4) Create mask to exclude diagonal
        mask = ~torch.eye(V, dtype=torch.bool, device=keypoints_2d_pred.device)
        mask = mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, N)
        
        # Compute errors
        coord_err = self.huber(u_back, u_ref)
        depth_err = self.huber(z_back, z_ref)
        
        # Apply mask and average
        mask_2d = mask.unsqueeze(-1).expand_as(coord_err)
        mask_1d = mask.unsqueeze(-1)
        
        loss_coord = (coord_err * mask_2d).sum() / mask_2d.sum()
        loss_depth = (depth_err * mask_1d).sum() / mask_1d.sum()

        loss_coord = torch.where(torch.abs(loss_coord) < 1e-8, torch.tensor(0.0), loss_coord)
        loss_depth = torch.where(torch.abs(loss_depth) < 1e-13, torch.tensor(0.0), loss_depth)
        
        return loss_coord, loss_depth

    def compute_differential_triangulation_loss(self, keypoints_2d_pred, keypoints_2d_gt):
        """
        Differential triangulation: Triangulate BOTH pred and GT, compare the 3D points.
        This removes systematic biases from calibration errors.
        """
        P = self.projection.unsqueeze(0).expand(keypoints_2d_pred.shape[0], -1, 3, 4)
        
        # Triangulate both predictions and ground truth
        kpts_3d_pred = triangulate_points(keypoints_2d_pred, P)  # [B, N, 3]
        kpts_3d_gt = triangulate_points(keypoints_2d_gt, P)      # [B, N, 3]
        
        # Compare in 3D space - systematic errors cancel out
        loss_3d = self.huber(kpts_3d_pred, kpts_3d_gt).mean()
        
        return loss_3d

class CurriculumScheduler:
    """
    Curriculum learning scheduler for gradually enabling losses and increasing
    masking difficulty.
    """
    
    def __init__(self, config):
        # Extract loss components and curriculum settings
        self.loss_components = config['loss']['components']
        self.curriculum = config['curriculum']
        
        self.loss_schedule = self.curriculum['loss_schedule']
        self.masking_schedule = self.curriculum['masking_schedule']
        
        # Validate that all scheduled losses exist in components
        for loss_name in self.loss_schedule.keys():
            if loss_name not in self.loss_components:
                raise ValueError(
                    f"Loss '{loss_name}' in loss_schedule but not in loss.components"
                )
    
    def get_loss_weights(self, epoch):
        """
        Get loss weights for current epoch with curriculum scheduling.
        """
        loss_weights = {}
        
        for loss_name, loss_config in self.loss_components.items():
            # Get loss settings
            enabled = loss_config['enabled']
            target_weight = loss_config['weight']
            warmup_epochs = loss_config.get('warmup_epochs', 1)
            enable_epoch = self.loss_schedule[loss_name]
            
            if not enabled:
                loss_weights[loss_name] = 0.0
            elif epoch < enable_epoch:
                loss_weights[loss_name] = 0.0
            else:
                epochs_since_enable = epoch - enable_epoch
                if warmup_epochs <= 1:
                    ramp_progress = 1.0
                else:
                    ramp_progress = min(1.0, (epochs_since_enable + 1) / warmup_epochs)
                loss_weights[loss_name] = target_weight * ramp_progress
        return loss_weights
    
    def get_masking_ratio(self, epoch):
        """
        Get occlusion ratio for current epoch.
        """
        scheme = self.masking_schedule['scheme']
        start_ratio = self.masking_schedule['start_ratio']
        end_ratio = self.masking_schedule['end_ratio']
        total_epochs = self.masking_schedule['total_epochs']
        
        if scheme == 'constant':
            return start_ratio
        
        elif scheme == 'linear':
            ratio = start_ratio + (end_ratio - start_ratio) * (epoch / total_epochs)
            return min(end_ratio, ratio)
        
        else:
            raise ValueError(
                f"Unknown masking scheme '{scheme}'. "
                f"Supported: 'constant', 'linear'"
            )
    
    def get_curriculum_info(self, epoch):
        """
        Get human-readable curriculum information for logging.
        """
        loss_weights = self.get_loss_weights(epoch)
        masking_ratio = self.get_masking_ratio(epoch)
        active_losses = [name for name, weight in loss_weights.items() if weight > 0]
        
        # Find losses in warmup
        warmup_losses = []
        for loss_name, weight in loss_weights.items():
            if weight > 0 and weight < self.loss_components[loss_name]['weight']:
                warmup_losses.append(loss_name)
        
        return {
            'epoch': epoch,
            'masking_ratio': masking_ratio,
            'active_losses': active_losses,
            'num_active_losses': len(active_losses),
            'warmup_losses': warmup_losses,
            'loss_weights': loss_weights,
        }