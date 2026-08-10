import torch
import torch.nn as nn
import torch.nn.functional as F

class BiomechanicalPoseLoss(nn.Module):
    def __init__(self, joint_edges, enable_skeleton=True, enable_temporal=True, lambda_coord=1.0, lambda_bone=0.5, lambda_vel=0.1):
        super().__init__()
        self.joint_edges = joint_edges
        self.lambda_coord = lambda_coord
        self.lambda_bone = lambda_bone
        self.lambda_vel = lambda_vel

        self.skeleton = enable_skeleton
        self.temporal = enable_temporal
        
        self.coord_loss = nn.HuberLoss(delta=1.0)
        
    def forward(self, d_corrected, d_gt, d_history=None):
        """
        Args:
            d_corrected: (B, N, 3) - Model predictions at t+1
            d_gt: (B, N, 3) - Ground truth at t+1
            d_history: (B, N, T, 3) - History up to t (optional, for velocity loss)
        """
        loss_coord = self.coord_loss(d_corrected, d_gt)

        loss_bone = 0.0
        if self.skeleton and self.lambda_bone > 0:
            for (i, j) in self.joint_edges:
                bone_pred = torch.norm(d_corrected[:, i, :] - d_corrected[:, j, :], dim=-1)
                bone_gt = torch.norm(d_gt[:, i, :] - d_gt[:, j, :], dim=-1)
                loss_bone += F.mse_loss(bone_pred, bone_gt)
                
            loss_bone = loss_bone / len(self.joint_edges)
            
        loss_vel = 0.0
        if self.temporal and self.lambda_vel > 0:
            last_history = d_history[:, -1, :, :]
            
            v_pred = d_corrected - last_history
            v_gt = d_gt - last_history
            loss_vel = F.mse_loss(v_pred, v_gt)
            
        return {
            'total': (self.lambda_coord * loss_coord) + (self.lambda_bone * loss_bone) + (self.lambda_vel * loss_vel),
            'coord':(self.lambda_coord * loss_coord),
            'skeleton': (self.lambda_bone * loss_bone),
            'velocity': (self.lambda_vel * loss_vel)
        }
