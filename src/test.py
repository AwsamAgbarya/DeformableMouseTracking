import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import hydra
from omegaconf import DictConfig, OmegaConf

from MVT.models import MultiView3DKeypointModel
from MVT.dataset import MV_Dataset
from MVT.loss import KeypointLoss
from MVT.metrics import MetricsTracker
from utils.geometry import get_predefined_cams


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


def setup_dataset(args, projections):
    """Setup test dataset and dataloader"""
    print(f"\n{'='*60}")
    print(f"Setting up test dataset")
    print(f"{'='*60}")
    
    # Dataset configuration
    dataset_conf = args['test_dataset']
    # Create dataset
    test_dataset = MV_Dataset(conf=dataset_conf)
    test_dataset.reset_views(projections=projections, normalize=True)
    
    # Apply occlusion
    test_dataset.occlude(ratio=dataset_conf['occlusion_ratio'])
    
    print(f"✓ Dataset size: {len(test_dataset)} samples")
    print(f"✓ Occlusion ratio: {dataset_conf['occlusion_ratio']}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_conf['batch_size'],
        shuffle=False,
        num_workers=dataset_conf['num_workers'],
        pin_memory=True
    )
    
    return test_dataset, test_loader


def compute_metrics(model, test_loader, criterion, device):
    """Compute comprehensive metrics on test set"""
    print(f"\n{'='*60}")
    print(f"Computing test metrics")
    print(f"{'='*60}")
    
    model.eval()
    
    # Initialize metric trackers
    metrics = {
        'total_loss': [],
        'reprojection_loss': [],
        'depth_loss': [],
        'confidence_loss': [],
        'per_keypoint_error': [[] for _ in range(test_loader.dataset.part_count)],
        'per_view_error': [[] for _ in range(test_loader.dataset.view_count)],
        'predictions': [],
        'ground_truth': [],
        'occlusion_masks': []
    }
    
    with torch.no_grad():
        for batch_idx, (keypoints_2d, depths, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move to device
            keypoints_2d = keypoints_2d.to(device)
            depths = depths.to(device)
            masks = masks.to(device).bool()  # (B, C, N)
            
            # Forward pass
            output = model(
                keypoints_2d=keypoints_2d,
                occlusion_mask=masks,
                depth_map=depths
            )
            pred_2d = output.get('coordinates', None)
            pred_depths = output.get('depth', None)
            confidence = output.get('confidence', None)

            # Compute loss
            loss_dict = criterion(
                keypoints_2d_pred=pred_2d,
                keypoints_2d_gt=keypoints_2d,
                masks=masks,
                depths_pred=pred_depths,
                depths_gt=depths,
                confidence_pred=confidence,
            )
            
            # Store overall metrics
            metrics.update(loss_dict)
            
            # Compute per-keypoint and per-view errors
            pred_coords = output['coordinates']
            
            # Only compute error on occluded keypoints (the ones we predicted)
            occluded_mask = ~masks
            
            if occluded_mask.any():
                # Per-keypoint error (average across batch and views)
                for k in range(keypoints_2d.shape[2]):  # Iterate over keypoints
                    kp_mask = occluded_mask[:, :, k]
                    if kp_mask.any():
                        error = torch.norm(
                            pred_coords[:, :, k, :] - keypoints_2d[:, :, k, :],
                            dim=-1
                        )
                        masked_error = error[kp_mask.squeeze(-1)]
                        metrics['per_keypoint_error'][k].extend(masked_error.cpu().numpy())
                
                # Per-view error (average across batch and keypoints)
                for v in range(keypoints_2d.shape[1]):  # Iterate over views
                    view_mask = occluded_mask[:, v, :]
                    if view_mask.any():
                        error = torch.norm(
                            pred_coords[:, v, :, :] - keypoints_2d[:, v, :, :],
                            dim=-1
                        )
                        masked_error = error[view_mask.squeeze(-1)]
                        metrics['per_view_error'][v].extend(masked_error.cpu().numpy())
            
            # Store predictions for visualization
            if batch_idx < 3:  # Store first few batches
                metrics['predictions'].append(pred_coords.cpu())
                metrics['ground_truth'].append(keypoints_2d.cpu())
                metrics['occlusion_masks'].append(masks.cpu())
    
    # Aggregate metrics
    results = {
        'mean_reprojection': np.mean(metrics['reprojection_loss']) if metrics['reprojection_loss'] else None,
        'mean_depth_loss': np.mean(metrics['depth_loss']) if metrics['depth_loss'] else None,
        'mean_confidence_loss': np.mean(metrics['confidence_loss']) if metrics['confidence_loss'] else None,
        'per_keypoint_mean_error': [np.mean(errors) if errors else 0 for errors in metrics['per_keypoint_error']],
        'per_keypoint_std_error': [np.std(errors) if errors else 0 for errors in metrics['per_keypoint_error']],
        'per_view_mean_error': [np.mean(errors) if errors else 0 for errors in metrics['per_view_error']],
        'per_view_std_error': [np.std(errors) if errors else 0 for errors in metrics['per_view_error']],
        'predictions': torch.cat(metrics['predictions'], dim=0) if metrics['predictions'] else None,
        'ground_truth': torch.cat(metrics['ground_truth'], dim=0) if metrics['ground_truth'] else None,
        'occlusion_masks': torch.cat(metrics['occlusion_masks'], dim=0) if metrics['occlusion_masks'] else None,
    }
    
    return results


def plot_metrics(results, output_dir, dataset):
    """Generate comprehensive performance plots"""
    print(f"\n{'='*60}")
    print(f"Generating performance plots")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Overall metrics summary
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    # Loss summary
    ax = axes[0, 0]
    losses = []
    labels = []
    if results['mean_reprojection'] is not None:
        losses.append(results['mean_reprojection'])
        labels.append('Reprojection')
    if results['mean_depth_loss'] is not None:
        losses.append(results['mean_depth_loss'])
        labels.append('Depth')
    if results['mean_confidence_loss'] is not None:
        losses.append(results['mean_confidence_loss'])
        labels.append('Confidence')
    
    if losses:
        bars = ax.bar(labels, losses, color=['#3498db', '#e74c3c', '#2ecc71'][:len(losses)])
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    # Per-keypoint error
    ax = axes[0, 1]
    keypoint_names = list(dataset.body_parts.keys())
    mean_errors = results['per_keypoint_mean_error']
    std_errors = results['per_keypoint_std_error']
    
    x_pos = np.arange(len(keypoint_names))
    ax.bar(x_pos, mean_errors, yerr=std_errors, capsize=5, alpha=0.7, color='#9b59b6')
    ax.set_xlabel('Keypoint', fontsize=12)
    ax.set_ylabel('Mean Error (normalized)', fontsize=12)
    ax.set_title('Per-Keypoint Reprojection Error', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(keypoint_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Per-view error
    ax = axes[1, 0]
    view_names = [f'View {i+1}' for i in range(len(results['per_view_mean_error']))]
    mean_errors = results['per_view_mean_error']
    std_errors = results['per_view_std_error']
    
    x_pos = np.arange(len(view_names))
    ax.bar(x_pos, mean_errors, yerr=std_errors, capsize=5, alpha=0.7, color='#f39c12')
    ax.set_xlabel('Camera View', fontsize=12)
    ax.set_ylabel('Mean Error (normalized)', fontsize=12)
    ax.set_title('Per-View Reprojection Error', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(view_names)
    ax.grid(True, alpha=0.3)
    
    # Error distribution histogram
    ax = axes[1, 1]
    all_errors = []
    for errors in results['per_keypoint_mean_error']:
        if errors > 0:
            all_errors.append(errors)
    
    if all_errors:
        ax.hist(all_errors, bins=20, alpha=0.7, color='#1abc9c', edgecolor='black')
        ax.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_errors):.4f}')
        ax.set_xlabel('Error (normalized)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved metrics_summary.png")
    
    # 2. Detailed per-keypoint error plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    keypoint_names = list(dataset.body_parts.keys())
    mean_errors = results['per_keypoint_mean_error']
    std_errors = results['per_keypoint_std_error']
    
    x_pos = np.arange(len(keypoint_names))
    bars = ax.bar(x_pos, mean_errors, yerr=std_errors, capsize=5, alpha=0.7, color='#3498db')
    
    # Color code bars by error magnitude
    colors = plt.cm.RdYlGn_r(np.array(mean_errors) / max(mean_errors))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Keypoint', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Reprojection Error (normalized)', fontsize=14, fontweight='bold')
    ax.set_title('Detailed Per-Keypoint Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(keypoint_names, rotation=45, ha='right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, error) in enumerate(zip(bars, mean_errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{error:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_keypoint_error.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved per_keypoint_error.png")


def visualize_predictions(results, output_dir, dataset, num_samples=8):
    """Visualize sample predictions with ground truth"""
    print(f"\n{'='*60}")
    print(f"Generating prediction visualizations")
    print(f"{'='*60}")
    
    if results['predictions'] is None:
        print("⚠ No predictions to visualize")
        return
    
    predictions = results['predictions'][:num_samples]
    ground_truth = results['ground_truth'][:num_samples]
    occlusion_masks = results['occlusion_masks'][:num_samples]
    
    # Denormalize coordinates for better visualization
    predictions_denorm = dataset.denormalize_2d(predictions)
    ground_truth_denorm = dataset.denormalize_2d(ground_truth)
    
    num_views = predictions.shape[1]
    keypoint_names = list(dataset.body_parts.keys())
    
    # Create a visualization for each sample
    for sample_idx in range(min(num_samples, predictions.shape[0])):
        fig, axes = plt.subplots(1, num_views, figsize=(6*num_views, 6))
        if num_views == 1:
            axes = [axes]
        
        fig.suptitle(f'Sample {sample_idx + 1} - Predictions vs Ground Truth', 
                    fontsize=16, fontweight='bold')
        
        for view_idx in range(num_views):
            ax = axes[view_idx]
            
            pred = predictions_denorm[sample_idx, view_idx].numpy()
            gt = ground_truth_denorm[sample_idx, view_idx].numpy()
            mask = occlusion_masks[sample_idx, view_idx].numpy()
            
            # Plot ground truth (all keypoints)
            ax.scatter(gt[:, 0], gt[:, 1], c='blue', s=100, alpha=0.5, 
                      label='Ground Truth', marker='o', edgecolors='black', linewidths=1.5)
            
            # Plot predictions (only for occluded keypoints)
            occluded_indices = np.where(~mask)[0]
            if len(occluded_indices) > 0:
                ax.scatter(pred[occluded_indices, 0], pred[occluded_indices, 1], 
                          c='red', s=100, alpha=0.7, label='Predicted (occluded)', 
                          marker='x', linewidths=2)
                
                # Draw error vectors
                for idx in occluded_indices:
                    ax.arrow(gt[idx, 0], gt[idx, 1],
                            pred[idx, 0] - gt[idx, 0],
                            pred[idx, 1] - gt[idx, 1],
                            head_width=5, head_length=5, fc='orange', ec='orange', 
                            alpha=0.5, linewidth=1.5)
            
            # Annotate keypoints
            for i, name in enumerate(keypoint_names):
                if not mask[i]:  # Occluded
                    ax.annotate(name, (gt[i, 0], gt[i, 1]), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, color='red', fontweight='bold')
            
            ax.set_xlabel('X (pixels)', fontsize=12)
            ax.set_ylabel('Y (pixels)', fontsize=12)
            ax.set_title(f'View {view_idx + 1}', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_sample_{sample_idx+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {min(num_samples, predictions.shape[0])} prediction visualizations")


def save_results_summary(results, output_dir, args):
    """Save numerical results to JSON file"""
    summary = {
        'test_config': {
            'csv_path': args['file_path'],
            'num_views': 3,
            'num_keypoints': 10,
            'occlusion_ratio': args['occlusion_ratio'],
            'batch_size': args['batch_size'],
        },
        'overall_metrics': {
            'mean_loss': float(results['mean_loss']),
            'std_loss': float(results['std_loss']),
            'mean_reprojection': float(results['mean_reprojection']) if results['mean_reprojection'] else None,
            'mean_depth_loss': float(results['mean_depth_loss']) if results['mean_depth_loss'] else None,
            'mean_confidence_loss': float(results['mean_confidence_loss']) if results['mean_confidence_loss'] else None,
        },
        'per_keypoint_errors': {
            'mean': [float(x) for x in results['per_keypoint_mean_error']],
            'std': [float(x) for x in results['per_keypoint_std_error']],
        },
        'per_view_errors': {
            'mean': [float(x) for x in results['per_view_mean_error']],
            'std': [float(x) for x in results['per_view_std_error']],
        }
    }
    
    output_path = os.path.join(output_dir, 'test_results.json')
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✓ Saved test results to: {output_path}")


def print_results_summary(results):
    """Print formatted results summary to console"""
    print(f"\n{'='*60}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n📊 Overall Metrics:")
    print(f"  • Mean Total Loss: {results['mean_loss']:.6f} ± {results['std_loss']:.6f}")
    if results['mean_reprojection']:
        print(f"  • Mean Reprojection Loss: {results['mean_reprojection']:.6f}")
    if results['mean_depth_loss']:
        print(f"  • Mean Depth Loss: {results['mean_depth_loss']:.6f}")
    if results['mean_confidence_loss']:
        print(f"  • Mean Confidence Loss: {results['mean_confidence_loss']:.6f}")
    
    print(f"\n🎯 Per-Keypoint Performance:")
    mean_errors = results['per_keypoint_mean_error']
    best_kp = np.argmin(mean_errors)
    worst_kp = np.argmax(mean_errors)
    print(f"  • Best keypoint: #{best_kp} (error: {mean_errors[best_kp]:.6f})")
    print(f"  • Worst keypoint: #{worst_kp} (error: {mean_errors[worst_kp]:.6f})")
    print(f"  • Average across keypoints: {np.mean(mean_errors):.6f}")
    
    print(f"\n📷 Per-View Performance:")
    view_errors = results['per_view_mean_error']
    for i, error in enumerate(view_errors):
        print(f"  • View {i+1}: {error:.6f} ± {results['per_view_std_error'][i]:.6f}")
    
    print(f"\n{'='*60}\n")

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Create output directory
    os.makedirs(config['test_dataset']['output_dir'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Setup dataset
    Ps = []
    for cam in ['top', 'side', 'front']:
        P = get_predefined_cams(cam)
        Ps.append(P)
    projections = torch.stack(Ps, dim=0)
    test_dataset, test_loader = setup_dataset(config,projections)
    
    # Load model configuration
    model_config = config['model']
    
    # Initialize model
    model = MultiView3DKeypointModel(
        conf=model_config,
        num_keypoints=10,
        num_views=3
    ).to(device)
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(config['training']['checkpoint_dir']+'/best_model.pt', model, device)
    
    # Initialize loss function
    loss_config = config['loss']
    criterion = KeypointLoss(projections, test_dataset.normalizer,loss_config=loss_config)
    
    # Compute metrics
    results = compute_metrics(model, test_loader, criterion, device)
    
    # Print results
    print_results_summary(results)
    
    # Save results
    save_results_summary(results, config['test_dataset']['output_dir'], config['test_dataset'])
    
    # Generate plots
    plot_metrics(results, config['test_dataset']['output_dir'], test_dataset)
    
    # Visualize predictions
    visualize_predictions(results, config['test_dataset']['output_dir'], test_dataset, num_samples=config['test_dataset']['num_vis_samples'])
    
    print(f"\n✅ Testing complete! Results saved to: {config['test_dataset']['output_dir']}")


if __name__ == '__main__':
    main()
