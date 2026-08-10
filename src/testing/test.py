import torch
import pandas as pd
import numpy as np
import json
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from MVT.dataset import DataDrivenNormalization, exact_mask
from utils.geometry import get_predefined_cams, project_points
from MVT.triangulation_system import SkeletonAligner, MultiViewTriangulation
from MVT.models import MultiView3DKeypointModel
from MTT.SpatialDenoiser.model import DeformationCorrector


class TestDataset(Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.normalizer = DataDrivenNormalization()
            
        dataset = pd.read_csv(conf['file_path'])
        self.body_parts = {part:idx for idx,part in enumerate(dataset['part'].unique())}
        dataset['p_idx'] = dataset["part"].map(self.body_parts)
        self.dataset = dataset[["time"] + ["p_idx"] + dataset.columns.drop(["time", "p_idx"]).tolist()].drop(columns=['part'])
        self.part_count = len(self.body_parts.keys())

        # Extract coordinates
        self.rigid_coords_3d = torch.tensor(self.dataset.iloc[:, 2:5].to_numpy(), dtype=torch.float32).view(-1, self.part_count, 3)
        self.deformable_coords_3d = torch.tensor((self.dataset.drop(columns=['x_r', 'y_r', 'z_r']).iloc[:, 2:5]).to_numpy(), dtype=torch.float32).view(-1, self.part_count, 3)

        # Get projections
        Ps = []
        for cam in self.conf['cameras']:
            P = get_predefined_cams(cam)
            Ps.append(P)
        self.view_count = len(Ps)
        self.projections = torch.stack(Ps, dim=0)

        traj_list_d = []
        # Project to 2D (unnormalized)
        for i, view in enumerate(self.projections):
            proj_data_d, __ = project_points(self.deformable_coords_3d[:, :, :], view)
            traj_list_d.append(proj_data_d[:, None, ...])
        self.deformable_coords_2d = torch.concatenate(traj_list_d, dim=1)

        # Normalize 2D coordinates using isotropic minmax
        self.centers = self.deformable_coords_2d.mean(dim=2)
        coords_centered = self.deformable_coords_2d - self.centers.unsqueeze(2)

        if not self.normalizer.is_fitted:
            print("Fitting normalizer on 2D data...")
            self.normalizer.fit(coords_centered, None)
        self.deformable_coords_2d_norm = self.normalizer.normalize(coords_centered)  
  
    def set_occlusion(self, ratio):
        self.mask_ratio = float(ratio)

    def __getitem__(self, idx):
        if self.mask_ratio <= 0:
            mask = torch.ones((self.view_count, self.part_count, 1), dtype=torch.bool)
        else:
            n_hide = int(round(self.mask_ratio * self.part_count))
            n_hide = max(0, min(self.part_count, n_hide))
            mask = torch.ones((self.view_count, self.part_count, 1), dtype=torch.bool)
            for v in range(self.view_count):
                m = exact_mask(self.part_count, self.mask_ratio) 
                mask[v, :, 0] = m

        return {
            "2d_norm": self.deformable_coords_2d_norm[idx],
            "2d_original": self.deformable_coords_2d[idx],
            "mask": mask,
            "3d_r": self.rigid_coords_3d[idx],
            "3d_d": self.deformable_coords_3d[idx],
            "centers": self.centers[idx]
        }
        
    def __len__(self):
        return self.deformable_coords_2d_norm.shape[0]
    
    def denormalize_2d(self, keypoints_2d_norm):
        return self.normalizer.denormalize(keypoints_2d_norm)


class MetricsCalculator:
    """Calculate various metrics for keypoint prediction evaluation"""
    
    @staticmethod
    def mpjpe(pred, target, valid_mask=None):
        """Mean Per Joint Position Error (Euclidean distance)"""
        error = torch.norm(pred - target, dim=-1)
        if valid_mask is not None:
            error = error * valid_mask
            return error.sum() / (valid_mask.sum() + 1e-8)
        return error.mean()
    
    @staticmethod
    def pck(pred, target, threshold=0.05, valid_mask=None):
        """Percentage of Correct Keypoints within threshold"""
        distances = torch.norm(pred - target, dim=-1)
        correct = (distances < threshold).float()
        if valid_mask is not None:
            correct = correct * valid_mask
            return correct.sum() / (valid_mask.sum() + 1e-8)
        return correct.mean()
    
    @staticmethod
    def per_joint_error(pred, target):
        """Per-joint position error for detailed analysis"""
        return torch.norm(pred - target, dim=-1)
    
    @staticmethod
    def per_view_error(pred_2d, target_2d):
        """Per-view error for 2D predictions"""
        # pred_2d, target_2d: [batch, views, joints, 2]
        errors = torch.norm(pred_2d - target_2d, dim=-1)  # [batch, views, joints]
        return errors.mean(dim=(0, 2))  # Average over batch and joints for each view

class TestEvaluator:
    def __init__(self, dataset, dataset_config, model, corrector, triangulator, rigid_matcher, device, output_dir):
        self.model = model
        self.corrector = corrector
        self.triangulator = triangulator
        self.rigid_matcher = rigid_matcher
        self.device = device
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup test dataset and dataloader
        print(f"\n{'='*60}")
        print(f"Setting up test dataset")
        print(f"{'='*60}")
        
        # Apply occlusion
        dataset.set_occlusion(ratio=dataset_config['occlusion_ratio'])
        print(f"✓ Dataset size: {len(dataset)} samples")
        print(f"✓ Occlusion ratio: {dataset_config['occlusion_ratio']}")
        
        # Create dataloader
        self.test_loader = DataLoader(
            dataset,
            batch_size=dataset_config['batch_size'],
            shuffle=False,  # Important for temporal linear interpolation
            num_workers=dataset_config.get('num_workers', 0),
            pin_memory=True
        )
        
        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        # Storage for detailed results
        self.results = {
            'stage1_2d': [], 
            'stage2_3d_deformable': [],
            'stage3_3d_rigid': [],
            'stage4_corrector': [],  # New: Corrector results
            'stage4_lerp': [],       # New: Linear Interpolation baseline
        }
        
        # Storage for visualizations
        self.all_samples = []

    def _temporal_linear_interpolation(self, keypoints_3d, masks):
        """
        Baseline: Temporal linear interpolation for occluded joints across the batch.
        Assumes the batch is a continuous sequence (DataLoader shuffle=False).
        """
        B, J, C = keypoints_3d.shape
        # Identify which joints are visible in at least one view
        is_visible_3d = torch.clip((~masks).sum(dim=1).squeeze(-1), min=0, max=1).bool()
        is_occluded_3d = ~is_visible_3d
        
        interp_kpts = keypoints_3d.clone().cpu()
        
        for j in range(J):
            valid_mask = is_visible_3d[:, j].cpu()
            valid_idx = torch.where(valid_mask)[0].numpy()
            missing_idx = torch.where(~valid_mask)[0].numpy()
            
            # Interpolate only if we have both missing frames and valid anchor frames
            if len(valid_idx) > 1 and len(missing_idx) > 0:
                for c in range(C):
                    # 1D Linear Interpolation
                    interp_vals = np.interp(
                        missing_idx,
                        valid_idx,
                        interp_kpts[valid_idx, j, c].numpy()
                    )
                    interp_kpts[missing_idx, j, c] = torch.tensor(interp_vals, dtype=interp_kpts.dtype)
                    
        return interp_kpts.to(self.device)

    def evaluate(self):
        """Main evaluation loop"""
        self.model.eval()
        self.corrector.eval()
        
        stage1_mpjpe, stage1_pck, stage1_per_joint, stage1_per_view = [], [], [], []
        stage2_mpjpe, stage2_pck, stage2_per_joint = [], [], []
        stage3_mpjpe, stage3_pck, stage3_per_joint = [], [], []
        stage4_corr_mpjpe, stage4_corr_pck, stage4_corr_per_joint = [], [], []
        stage4_lerp_mpjpe, stage4_lerp_pck, stage4_lerp_per_joint = [], [], []
        
        # ---------------------------------------------------------------
        # Rolling corrector history: list of per-frame tensors (N, 3)
        # We track global frame index to correctly index into it
        # ---------------------------------------------------------------
        corrector_history = []   # grows as we process frames; index = global frame idx
        global_frame_idx = 0     # tracks absolute frame position across batches
        
        print(f"\n{'='*60}")
        print(f"Running evaluation")
        print(f"{'='*60}")
        
        with torch.no_grad():
            for batch_idx, datapoint in enumerate(tqdm(self.test_loader, desc="Testing")):
                keypoints_2d_norm       = datapoint["2d_norm"].to(self.device)
                keypoints_2d_original   = datapoint["2d_original"].to(self.device)
                keypoints_3d_rigid      = datapoint["3d_r"].to(self.device)
                keypoints_3d_deformable = datapoint["3d_d"].to(self.device)
                masks                   = datapoint["mask"].to(self.device).bool()
                centers                 = datapoint["centers"].to(self.device)

                # --- STAGE 1: 2D Prediction ---
                output = self.model(keypoints_2d=keypoints_2d_norm, occlusion_mask=masks)
                pred_2d_norm = output.get('coordinates', None)
                if pred_2d_norm is None:
                    continue

                pred_2d = self.dataset.denormalize_2d(pred_2d_norm.cpu()).to(self.device)
                pred_2d = pred_2d + centers.unsqueeze(2)
                gt_2d   = keypoints_2d_original

                valid_mask_2d = (~masks).reshape(-1)
                mpjpe_2d = self.metrics_calc.mpjpe(pred_2d.reshape(-1, 2), gt_2d.reshape(-1, 2), valid_mask=valid_mask_2d)
                pck_2d   = self.metrics_calc.pck(pred_2d.reshape(-1, 2), gt_2d.reshape(-1, 2), threshold=5.0)
                stage1_mpjpe.append(mpjpe_2d.item())
                stage1_pck.append(pck_2d.item())
                stage1_per_joint.append(self.metrics_calc.per_joint_error(pred_2d, gt_2d).cpu())
                stage1_per_view.append(self.metrics_calc.per_view_error(pred_2d, gt_2d).cpu())

                # --- STAGE 2: 3D Triangulation (Deformable) ---
                batch_size  = pred_2d.shape[0]
                confidences = (masks + 1).float()
                pred_3d_deformable_list = []
                for i in range(batch_size):
                    pred_3d_sample = self.triangulator.triangulate(
                        pred_2d[i].unsqueeze(0), confidences=confidences[i].unsqueeze(0)
                    )
                    pred_3d_deformable_list.append(pred_3d_sample)

                pred_3d_deformable = torch.cat(pred_3d_deformable_list, dim=0)
                gt_3d_deformable   = keypoints_3d_deformable[:, :, :3]
                valid_mask_3d      = torch.clip((~masks).sum(dim=1).squeeze(-1), min=0, max=1).bool()

                mpjpe_3d_def = self.metrics_calc.mpjpe(pred_3d_deformable, gt_3d_deformable, valid_mask=valid_mask_3d)
                pck_3d_def   = self.metrics_calc.pck(pred_3d_deformable, gt_3d_deformable, threshold=0.1)
                stage2_mpjpe.append(mpjpe_3d_def.item())
                stage2_pck.append(pck_3d_def.item())
                stage2_per_joint.append(self.metrics_calc.per_joint_error(pred_3d_deformable, gt_3d_deformable).cpu())

                # --- STAGE 3: 3D Rigid Alignment ---
                pred_3d_rigid_list = []
                for i in range(batch_size):
                    aligned, _, _, _ = self.rigid_matcher.align(
                        pred_3d_deformable[i], keypoints_3d_rigid[i, :, :3], method='weighted'
                    )
                    pred_3d_rigid_list.append(aligned.unsqueeze(0))

                pred_3d_rigid = torch.cat(pred_3d_rigid_list, dim=0)
                gt_3d_rigid   = keypoints_3d_rigid[:, :, :3]

                mpjpe_3d_rigid = self.metrics_calc.mpjpe(pred_3d_rigid, gt_3d_rigid, valid_mask=valid_mask_3d)
                pck_3d_rigid   = self.metrics_calc.pck(pred_3d_rigid, gt_3d_rigid, threshold=0.1)
                stage3_mpjpe.append(mpjpe_3d_rigid.item())
                stage3_pck.append(pck_3d_rigid.item())
                stage3_per_joint.append(self.metrics_calc.per_joint_error(pred_3d_rigid, gt_3d_rigid).cpu())

                # --- STAGE 4A: Corrector ---
                # Relative distances between deformable and rigid predictions: (B, N, 3)
                relative_dist = pred_3d_deformable - pred_3d_rigid

                # Build d_history (B, N, 5, 3) from the rolling corrector output buffer.
                # For the very first frames (< 5 available), pad with zeros.
                d_history = self._build_corrector_history(
                    corrector_history,
                    global_frame_idx,
                    batch_size,
                    n_joints=pred_3d_rigid.shape[1],
                ).transpose(1,2)  # (B, N, 5, 3)

                # Forward pass through corrector
                pred_delta      = self.corrector(relative_dist, d_history)   # (B, N, 3)
                pred_3d_corrector = pred_3d_rigid + pred_delta

                # ---------------------------------------------------------------
                # Update rolling history with the corrector's OWN outputs.
                # This is what gets fed back as d_history in future frames.
                # ---------------------------------------------------------------
                for i in range(batch_size):
                    corrector_history.append(pred_3d_corrector[i].detach().cpu())

                global_frame_idx += batch_size

                # --- STAGE 4B: LERP Baseline ---
                pred_3d_lerp = self._temporal_linear_interpolation(pred_3d_deformable, masks)

                # Metrics: Corrector
                mpjpe_corr = self.metrics_calc.mpjpe(pred_3d_corrector, gt_3d_deformable, valid_mask=valid_mask_3d)
                pck_corr   = self.metrics_calc.pck(pred_3d_corrector, gt_3d_deformable, threshold=0.1)
                stage4_corr_mpjpe.append(mpjpe_corr.item())
                stage4_corr_pck.append(pck_corr.item())
                stage4_corr_per_joint.append(self.metrics_calc.per_joint_error(pred_3d_corrector, gt_3d_deformable).cpu())

                # Metrics: LERP
                mpjpe_lerp = self.metrics_calc.mpjpe(pred_3d_lerp, gt_3d_deformable, valid_mask=valid_mask_3d)
                pck_lerp   = self.metrics_calc.pck(pred_3d_lerp, gt_3d_deformable, threshold=0.1)
                stage4_lerp_mpjpe.append(mpjpe_lerp.item())
                stage4_lerp_pck.append(pck_lerp.item())
                stage4_lerp_per_joint.append(self.metrics_calc.per_joint_error(pred_3d_lerp, gt_3d_deformable).cpu())

                # Store for visualization
                for i in range(batch_size):
                    self.all_samples.append({
                        'pred_2d': pred_2d[i].cpu(),
                        'gt_2d': gt_2d[i].cpu(),
                        'pred_3d_deformable': pred_3d_deformable[i].cpu(),
                        'gt_3d_deformable': gt_3d_deformable[i].cpu(),
                        'pred_3d_rigid': pred_3d_rigid[i].cpu(),
                        'gt_3d_rigid': gt_3d_rigid[i].cpu(),
                        'pred_3d_corrector': pred_3d_corrector[i].cpu(),
                        'pred_3d_lerp': pred_3d_lerp[i].cpu(),
                        'mask': masks[i].cpu(),
                        'stage1_error': mpjpe_2d.item(),
                        'stage2_error': mpjpe_3d_def.item(),
                        'stage3_error': mpjpe_3d_rigid.item(),
                        'corrector_error': mpjpe_corr.item(),
                        'lerp_error': mpjpe_lerp.item(),
                    })

            # Aggregate
            self.results = {
                'stage1_2d': {
                    'mpjpe': np.mean(stage1_mpjpe), 'mpjpe_std': np.std(stage1_mpjpe),
                    'pck': np.mean(stage1_pck), 'pck_std': np.std(stage1_pck),
                    'per_joint_errors': torch.cat(stage1_per_joint, dim=0),
                    'per_view_errors': torch.stack(stage1_per_view, dim=0),
                },
                'stage2_3d_deformable': {
                    'mpjpe': np.mean(stage2_mpjpe), 'mpjpe_std': np.std(stage2_mpjpe),
                    'pck': np.mean(stage2_pck), 'pck_std': np.std(stage2_pck),
                    'per_joint_errors': torch.cat(stage2_per_joint, dim=0),
                },
                'stage3_3d_rigid': {
                    'mpjpe': np.mean(stage3_mpjpe), 'mpjpe_std': np.std(stage3_mpjpe),
                    'pck': np.mean(stage3_pck), 'pck_std': np.std(stage3_pck),
                    'per_joint_errors': torch.cat(stage3_per_joint, dim=0),
                },
                'stage4_corrector': {
                    'mpjpe': np.mean(stage4_corr_mpjpe), 'mpjpe_std': np.std(stage4_corr_mpjpe),
                    'pck': np.mean(stage4_corr_pck), 'pck_std': np.std(stage4_corr_pck),
                    'per_joint_errors': torch.cat(stage4_corr_per_joint, dim=0),
                },
                'stage4_lerp': {
                    'mpjpe': np.mean(stage4_lerp_mpjpe), 'mpjpe_std': np.std(stage4_lerp_mpjpe),
                    'pck': np.mean(stage4_lerp_pck), 'pck_std': np.std(stage4_lerp_pck),
                    'per_joint_errors': torch.cat(stage4_lerp_per_joint, dim=0),
                }
            }
            return self.results


    def _build_corrector_history(self, corrector_history, global_frame_idx, batch_size, n_joints):
        """
        Assemble d_history of shape (B, N, 5, 3) for the current batch.

        For frame at absolute position `global_frame_idx + i`, history is the
        corrector outputs at positions [f-5, f-4, f-3, f-2, f-1].
        Positions before the start of the sequence are zero-padded.

        Args:
            corrector_history : list of (N, 3) CPU tensors, one per past frame
            global_frame_idx  : absolute index of the first frame in this batch
            batch_size        : B
            n_joints          : N
        Returns:
            d_history : (B, N, 5, 3) on self.device
        """
        HISTORY_LEN = 5
        d_history = torch.zeros(batch_size, n_joints, HISTORY_LEN, 3, device=self.device)

        for i in range(batch_size):
            frame_abs = global_frame_idx + i          # absolute frame index
            for h in range(HISTORY_LEN):
                src_idx = frame_abs - HISTORY_LEN + h  # e.g. h=4 → frame_abs - 1 (most recent)
                if 0 <= src_idx < len(corrector_history):
                    d_history[i, :, h, :] = corrector_history[src_idx].to(self.device)
                # else: leave as zero (padding for early frames)

        return d_history  # (B, N, 5, 3)
    
    def print_results(self):
        """Print evaluation results"""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        
        stages = [
            ('STAGE 1: 2D Prediction', 'stage1_2d', 'pixels', 5.0),
            ('STAGE 2: 3D Triangulation', 'stage2_3d_deformable', 'XYZ', 0.1),
            ('STAGE 3: 3D Rigid Alignment', 'stage3_3d_rigid', 'XYZ', 0.1),
            ('STAGE 4: LERP Baseline', 'stage4_lerp', 'XYZ', 0.1),
            ('STAGE 4: Corrector Final', 'stage4_corrector', 'XYZ', 0.1)
        ]
        
        for name, key, unit, pck_thresh in stages:
            print(f"\n[{name}]")
            print(f"  MPJPE ({unit}): {self.results[key]['mpjpe']:.4f} ± {self.results[key]['mpjpe_std']:.4f}")
            print(f"  PCK@{pck_thresh}: {self.results[key]['pck']*100:.2f}%")
            
        # Quick Corrector vs Lerp highlight
        imp = self.results['stage4_lerp']['mpjpe'] - self.results['stage4_corrector']['mpjpe']
        print(f"\n✓ Corrector vs LERP MPJPE Difference: {imp:+.4f} (Negative is better for Corrector)")

    def save_metrics_to_csv(self):
        """Save summary metrics to CSV"""
        csv_path = self.output_dir / 'metrics_summary.csv'
        
        stages = ['stage1_2d', 'stage2_3d_deformable', 'stage3_3d_rigid', 'stage4_lerp', 'stage4_corrector']
        stage_names = ['2D Prediction', '3D Triangulation', '3D Rigid Alignment', 'Linear Interp (Baseline)', 'Corrector']
        
        data = {
            'Stage': stage_names,
            'MPJPE_Mean': [self.results[k]['mpjpe'] for k in stages],
            'MPJPE_Std': [self.results[k]['mpjpe_std'] for k in stages],
            'PCK': [self.results[k]['pck'] for k in stages],
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved metrics summary to {csv_path}")
        
        # Save detailed per-joint errors
        per_joint_path = self.output_dir / 'per_joint_errors.csv'
        joint_data = {}
        
        for joint_idx in range(self.results['stage1_2d']['per_joint_errors'].shape[1]):
            for k, name in zip(stages, ['2D', '3D_Def', '3D_Rigid', '3D_Lerp', '3D_Corr']):
                joint_data[f'Joint_{joint_idx}_{name}'] = self.results[k]['per_joint_errors'][:, joint_idx].mean().item()
        
        pd.DataFrame([joint_data]).to_csv(per_joint_path, index=False)
        print(f"✓ Saved per-joint errors to {per_joint_path}")
    

    def plot_metrics(self):
        """Generate and save all metric plots"""
        print(f"\\n{'='*60}")
        print(f"Generating plots...")
        print(f"{'='*60}")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Overall MPJPE comparison across stages
        self._plot_mpjpe_comparison()
        
        # Per-joint error analysis
        self._plot_per_joint_errors()
        
        # Per-view error analysis (2D only)
        self._plot_per_view_errors()
        
        # Error distribution histograms
        self._plot_error_distributions()
        
        # PCK curves
        self._plot_pck_curves()
        
        print(f"✓ All plots saved to {self.output_dir}")
        
    def _plot_per_view_errors(self):
        """Plot per-view error analysis for 2D predictions"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        view_errors = self.results['stage1_2d']['per_view_errors'].mean(dim=0).numpy()
        view_names = [f'View {i}' for i in range(len(view_errors))]
        
        bars = ax.bar(view_names, view_errors, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Mean Error (pixels)', fontsize=14, fontweight='bold')
        ax.set_title('2D Prediction Error Per View', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, error in zip(bars, view_errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{error:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_view_errors.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_mpjpe_comparison(self):
        """Plot MPJPE comparison across stages"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        stages = ['2D\nPrediction', '3D\nTriangulation', '3D\nRigid\nAlignment', 'LERP\nBaseline', 'Corrector\nFinal']
        keys = ['stage1_2d', 'stage2_3d_deformable', 'stage3_3d_rigid', 'stage4_lerp', 'stage4_corrector']
        mpjpe_means = [self.results[k]['mpjpe'] for k in keys]
        mpjpe_stds = [self.results[k]['mpjpe_std'] for k in keys]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        bars = ax.bar(stages, mpjpe_means, yerr=mpjpe_stds, capsize=10, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('MPJPE', fontsize=14, fontweight='bold')
        ax.set_title('Mean Per Joint Position Error Across Stages', fontsize=16, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean, std in zip(bars, mpjpe_means, mpjpe_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}\n±{std:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mpjpe_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_joint_errors(self):
        """Plot per-joint error analysis"""
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        joint_names = [f'J{i}' for i in range(self.results['stage1_2d']['per_joint_errors'].shape[-1])]
        
        plots = [
            ('stage1_2d', 'Stage 1: 2D', '#3498db'),
            ('stage2_3d_deformable', 'Stage 2: 3D Triang', '#e74c3c'),
            ('stage3_3d_rigid', 'Stage 3: 3D Rigid', '#2ecc71'),
            ('stage4_lerp', 'Stage 4: LERP Baseline', '#f39c12'),
            ('stage4_corrector', 'Stage 4: Corrector', '#9b59b6')
        ]
        
        for i, (key, title, color) in enumerate(plots):
            errors = self.results[key]['per_joint_errors'].mean(dim=0).numpy()
            if i == 0:
                errors = self.results[key]['per_joint_errors'].mean(dim=(0, 1)).numpy() # 2D fix
                
            axes[i].bar(joint_names, errors, color=color, alpha=0.7, edgecolor='black')
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Mean Error', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(axis='y', alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_joint_errors.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_error_distributions(self):
        """Plot error distribution histograms"""
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        plots = [
            ('stage1_2d', 'Stage 1: 2D Error', '#3498db'),
            ('stage2_3d_deformable', 'Stage 2: 3D Triang', '#e74c3c'),
            ('stage3_3d_rigid', 'Stage 3: 3D Rigid', '#2ecc71'),
            ('stage4_lerp', 'Stage 4: LERP Error', '#f39c12'),
            ('stage4_corrector', 'Stage 4: Corrector Error', '#9b59b6')
        ]
        
        for i, (key, title, color) in enumerate(plots):
            errors = self.results[key]['per_joint_errors'].flatten().numpy()
            axes[i].hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
            axes[i].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')
            axes[i].set_xlabel('Error', fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pck_curves(self):
        """Plot PCK curves at different thresholds"""
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        thresholds_2d = np.linspace(0, 10, 50)
        thresholds_3d = np.linspace(0, 0.5, 50)
        
        plots = [
            ('stage1_2d', thresholds_2d, 'Stage 1: 2D PCK', '#3498db'),
            ('stage2_3d_deformable', thresholds_3d, 'Stage 2: 3D Triang PCK', '#e74c3c'),
            ('stage3_3d_rigid', thresholds_3d, 'Stage 3: 3D Rigid PCK', '#2ecc71'),
            ('stage4_lerp', thresholds_3d, 'Stage 4: LERP PCK', '#f39c12'),
            ('stage4_corrector', thresholds_3d, 'Stage 4: Corrector PCK', '#9b59b6')
        ]
        
        for i, (key, thresh_arr, title, color) in enumerate(plots):
            pck_list = []
            errors = self.results[key]['per_joint_errors'].flatten()
            for thresh in thresh_arr:
                pck_list.append((errors < thresh).float().mean().item() * 100)
                
            axes[i].plot(thresh_arr, pck_list, linewidth=2, color=color)
            axes[i].fill_between(thresh_arr, pck_list, alpha=0.3, color=color)
            axes[i].set_xlabel('Threshold', fontsize=12)
            axes[i].set_ylabel('PCK (%)', fontsize=12)
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].grid(alpha=0.3)
            axes[i].set_ylim([0, 105])
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pck_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_best_worst_predictions(self):
        """Visualize best and worst predictions for each stage"""
        print(f"\n{'='*60}")
        print(f"Generating best/worst prediction visualizations...")
        print(f"{'='*60}")
        
        # Sort samples by error for each stage
        samples_sorted_stage1 = sorted(self.all_samples, key=lambda x: x['stage1_error'])
        samples_sorted_stage2 = sorted(self.all_samples, key=lambda x: x['stage2_error'])
        samples_sorted_stage3 = sorted(self.all_samples, key=lambda x: x['stage3_error'])
        
        # Get top 3 best and worst 3 for each stage
        best_stage1 = samples_sorted_stage1[0]
        worst_stage1 = samples_sorted_stage1[-1]

        best_stage2 = samples_sorted_stage2[:3]
        worst_stage2 = samples_sorted_stage2[-3:]

        best_stage3 = samples_sorted_stage3[:3]
        worst_stage3 = samples_sorted_stage3[-3:]

        
        # Visualize each stage
        self._visualize_stage1_predictions(best_stage1, worst_stage1)
        self._visualize_stage2_predictions(best_stage2, worst_stage2)
        self._visualize_stage3_predictions(best_stage3, worst_stage3)
        
        print(f"✓ Best/worst visualizations saved to {self.output_dir}")
    
    def _visualize_stage1_predictions(self, best_sample, worst_sample):
        """Visualize Stage 1: 2D predictions for all 3 views of the best and worst sample"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Stage 1: 2D Prediction - Best vs Worst (Top, Side, Front Views)', fontsize=16, fontweight='bold')
        
        view_names = ['Top View', 'Side View', 'Front View']
        
        for v_idx in range(3):
            # Best prediction (Top Row, across 3 views)
            ax_best = axes[0, v_idx]
            self._plot_2d_comparison(
                ax_best, 
                best_sample, 
                f"Best Sample - {view_names[v_idx]}\nError: {best_sample['stage1_error']}px",
                view_idx=v_idx
            )
            
            # Worst prediction (Bottom Row, across 3 views)
            ax_worst = axes[1, v_idx]
            self._plot_2d_comparison(
                ax_worst, 
                worst_sample, 
                f"Worst Sample - {view_names[v_idx]}\nError: {worst_sample['stage1_error']}px",
                view_idx=v_idx
            )
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stage1_best_worst.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_2d_comparison(self, ax, sample, title, view_idx=0):
        """Plot 2D keypoint comparison for a single sample"""
        pred_2d = sample['pred_2d'].numpy()  # [views, joints, 2]
        gt_2d = sample['gt_2d'].numpy()
        mask = sample['mask'].squeeze(-1).numpy()  # [views, joints]
        
        # Plot specified view
        pred = pred_2d[view_idx]
        gt = gt_2d[view_idx]
        visible = mask[view_idx]
        # Highlight occluded keypoints
        visible_idx = np.where(visible)[0]
        occluded_idx = np.where(~visible)[0]

        # Plot ground truth
        ax.scatter(gt[visible_idx, 0], gt[visible_idx, 1], c='green', s=100, marker='o', 
                  label='Ground Truth', alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Plot predictions
        ax.scatter(pred[occluded_idx, 0], pred[occluded_idx, 1], c='red', s=100, marker='x', 
                  label='Prediction', alpha=0.7, linewidths=2)
        
        # Draw connections
        for i in range(len(gt)):
            ax.plot([gt[i, 0], pred[i, 0]], [gt[i, 1], pred[i, 1]], 
                   'gray', alpha=0.3, linestyle='--')
        
        if len(occluded_idx) > 0:
            ax.scatter(gt[occluded_idx, 0], gt[occluded_idx, 1], 
                      c='orange', s=150, marker='*', label='Occluded (predicted)',
                      edgecolors='black', linewidth=1)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X (pixels)', fontsize=10)
        ax.set_ylabel('Y (pixels)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
    
    def _visualize_stage2_predictions(self, best_samples, worst_samples):
        """Visualize Stage 2: 3D triangulation"""
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Stage 2: 3D Triangulation - Best vs Worst (XYZ)', fontsize=16, fontweight='bold')
        
        # Best predictions
        for idx, sample in enumerate(best_samples):
            # Added projection='3d' for 3D axis
            ax = fig.add_subplot(2, 3, idx+1, projection='3d')
            self._plot_3d_comparison(ax, sample, 'deformable', 
                                    f"Best #{idx+1}\nError: {sample['stage2_error']}")
        
        # Worst predictions
        for idx, sample in enumerate(worst_samples):
            ax = fig.add_subplot(2, 3, idx+4, projection='3d')
            self._plot_3d_comparison(ax, sample, 'deformable',
                                    f"Worst #{idx+1}\nError: {sample['stage2_error']}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stage2_best_worst.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_stage3_predictions(self, best_samples, worst_samples):
        """Visualize Stage 3: 3D rigid alignment"""
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Stage 3: 3D Rigid Alignment - Best vs Worst (XYZ)', fontsize=16, fontweight='bold')
        
        # Best predictions
        for idx, sample in enumerate(best_samples):
            ax = fig.add_subplot(2, 3, idx+1, projection='3d')
            self._plot_3d_comparison(ax, sample, 'rigid',
                                    f"Best #{idx+1}\nError: {sample['stage3_error']}")
        
        # Worst predictions
        for idx, sample in enumerate(worst_samples):
            ax = fig.add_subplot(2, 3, idx+4, projection='3d')
            self._plot_3d_comparison(ax, sample, 'rigid',
                                    f"Worst #{idx+1}\nError: {sample['stage3_error']}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stage3_best_worst.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_3d_comparison(self, ax, sample, stage_type, title):
        """Plot 3D keypoint comparison (XYZ)"""
        if stage_type == 'deformable':
            pred_3d = sample['pred_3d_deformable'].numpy()  # [joints, 3]
            gt_3d = sample['gt_3d_deformable'].numpy()
        else:  # rigid
            pred_3d = sample['pred_3d_rigid'].numpy()
            gt_3d = sample['gt_3d_rigid'].numpy()
        mask = sample['mask'].squeeze(-1)
        occluded = torch.clip((~mask).sum(dim=0).squeeze(-1), min=0, max=1).bool().numpy()
        visible_idx = np.where(~occluded)[0]
        occluded_idx = np.where(occluded)[0]

        # Plot XYZ fully
        ax.scatter(gt_3d[visible_idx, 0], gt_3d[visible_idx, 1], gt_3d[visible_idx, 2], c='green', s=100, marker='o',
                  label='GT (Visible)', alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.scatter(gt_3d[occluded_idx, 0], gt_3d[occluded_idx, 1], gt_3d[occluded_idx, 2], c='yellow', s=100, marker='*',
                  label='GT (Occluded)', alpha=0.7, edgecolors='black', linewidth=1.5)
        ax.scatter(pred_3d[occluded_idx, 0], pred_3d[occluded_idx, 1], pred_3d[occluded_idx, 2], c='red', s=100, marker='x',
                  label='Prediction', alpha=0.7, linewidths=2)
        
        # Draw 3D connections
        for i in range(len(gt_3d)):
            ax.plot([gt_3d[i, 0], pred_3d[i, 0]], 
                   [gt_3d[i, 1], pred_3d[i, 1]],
                   [gt_3d[i, 2], pred_3d[i, 2]],
                   'gray', alpha=0.3, linestyle='--')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

def load_checkpoint(checkpoint_path, model, device):
    """Load model weights from checkpoint"""
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        best_loss = checkpoint.get('best_loss', 'unknown')
        print(f"✓ Loaded model from epoch {epoch}")
        print(f"✓ Best training loss: {best_loss}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model weights")
    
    return model, checkpoint

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    output_dir = config['test_dataset']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"KEYPOINT PREDICTION MODEL TESTING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    # Prepare the test dataset
    dataset_config = config['test_dataset']
    test_dataset = TestDataset(dataset_config)
    
    # Prepare the model
    model = MultiView3DKeypointModel(
        config['model'], 
        test_dataset.part_count, 
        test_dataset.view_count
    ).to(device)

    corrector = DeformationCorrector(
        history_window = 5,
        **config['corrector']
    ).to(device)
    
    checkpoint_path = config['training']['model_dir'] + "best_model.pt"
    model, __ = load_checkpoint(checkpoint_path, model, device)
    checkpoint_path = config['training']['corrector_dir'] + "best.pt"
    corrector, __ = load_checkpoint(checkpoint_path, corrector, device)
    # Prepare the triangulator and rigid matcher
    triangulator = MultiViewTriangulation(projection_matrices=test_dataset.projections, device=device)
    rigid_matcher = SkeletonAligner(device=device)
    
    # Create evaluator
    evaluator = TestEvaluator(
        dataset=test_dataset,
        dataset_config=dataset_config,
        model=model,
        corrector=corrector,
        triangulator=triangulator,
        rigid_matcher=rigid_matcher,
        device=device,
        output_dir=output_dir
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_results()
    
    # Save metrics to CSV
    evaluator.save_metrics_to_csv()
    
    # Generate plots
    evaluator.plot_metrics()
    
    # Visualize best/worst predictions
    evaluator.visualize_best_worst_predictions()
    
    print(f"\n{'='*60}")
    print(f"TESTING COMPLETE!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()