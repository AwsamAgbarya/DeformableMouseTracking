"""
Mouse Movement Dataset Analysis Script
======================================
Analyzes multiple mouse movement datasets to determine:
- Data diversity and distribution
- Relative movement patterns (center-of-mass normalized)
- Velocities and accelerations
- Angular spans and velocities
- Dataset similarity for optimal train/val/test splits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class MouseDatasetAnalyzer:
    """Comprehensive analyzer for mouse movement datasets"""

    def __init__(self, data_folder='data'):
        self.output_dir = data_folder+'/analysis'
        self.data_folder = Path(data_folder)
        self.datasets = {}
        self.stats = {}
        self.centered_datasets = {}

    def load_datasets(self):
        """Load all CSV files from the data folder"""
        csv_files = sorted(self.data_folder.glob('*.csv'))
        print(f"Found {len(csv_files)} CSV files:")

        for csv_file in csv_files:
            name = csv_file.stem
            print(f"  - Loading {name}...")
            df = pd.read_csv(csv_file)
            self.datasets[name] = df
            print(f"    Shape: {df.shape}, Timeframes: {df['time'].nunique()}, Parts: {df['part'].nunique()}")

        return csv_files

    def center_by_com(self, df):
        """Center coordinates by center of mass for each timeframe"""
        df_centered = df.copy()

        # Process rigid body coordinates
        for time in df['time'].unique():
            mask = df['time'] == time
            # Compute COM for rigid body
            com_x_r = df.loc[mask, 'x_r'].mean()
            com_y_r = df.loc[mask, 'y_r'].mean()
            com_z_r = df.loc[mask, 'z_r'].mean()

            df_centered.loc[mask, 'x_r'] -= com_x_r
            df_centered.loc[mask, 'y_r'] -= com_y_r
            df_centered.loc[mask, 'z_r'] -= com_z_r

            # Compute COM for deformable body
            com_x_d = df.loc[mask, 'x_d'].mean()
            com_y_d = df.loc[mask, 'y_d'].mean()
            com_z_d = df.loc[mask, 'z_d'].mean()

            df_centered.loc[mask, 'x_d'] -= com_x_d
            df_centered.loc[mask, 'y_d'] -= com_y_d
            df_centered.loc[mask, 'z_d'] -= com_z_d

        return df_centered

    def compute_velocities(self, df, coord_suffix='_d'):
        """Compute velocities for each body part"""
        velocities = []
        parts = df['part'].unique()
        for part in parts:
            part_data = df[df['part'] == part].sort_values('time').reset_index(drop=True)

            if len(part_data) < 2:
                continue

            # Compute position differences
            dx = part_data[f'x{coord_suffix}'].diff()
            dy = part_data[f'y{coord_suffix}'].diff()
            dz = part_data[f'z{coord_suffix}'].diff()
            dt = part_data['time'].diff()

            # Velocity magnitude
            vel = np.sqrt(dx**2 + dy**2 + dz**2) / dt

            velocities.extend(vel.dropna().values)

        return np.array(velocities)

    def compute_angular_metrics(self, df, coord_suffix='_d'):
        """Compute angular spans and velocities between body parts"""
        angular_data = {'spans': [], 'velocities': []}

        times = sorted(df['time'].unique())

        for i, time in enumerate(times):
            frame = df[df['time'] == time]
            positions = frame[[f'x{coord_suffix}', f'y{coord_suffix}', f'z{coord_suffix}']].values
            if len(positions) < 2:
                continue

            # Compute pairwise distances (spans)
            distances = cdist(positions, positions)
            angular_data['spans'].extend(distances[np.triu_indices_from(distances, k=1)])

            # Compute angular velocities (change in relative positions)
            if i > 0:
                prev_frame = df[df['time'] == times[i-1]]
                if len(prev_frame) == len(frame):
                    prev_positions = prev_frame[[f'x{coord_suffix}', f'y{coord_suffix}', f'z{coord_suffix}']].values

                    # Compute vectors between consecutive frames
                    for p1 in range(len(positions)):
                        for p2 in range(p1+1, len(positions)):
                            vec_curr = positions[p2] - positions[p1]
                            vec_prev = prev_positions[p2] - prev_positions[p1]

                            # Angle between vectors
                            cos_angle = np.dot(vec_curr, vec_prev) / (np.linalg.norm(vec_curr) * np.linalg.norm(vec_prev) + 1e-8)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)

                            dt = (times[i] - times[i-1])
                            angular_vel = angle / dt if dt > 0 else 0

                            angular_data['velocities'].append(angular_vel)

        return angular_data

    def compute_pose_diversity(self, df, coord_suffix='_d'):
        """Compute diversity metrics for pose configurations"""
        poses = []

        for time in df['time'].unique():
            frame = df[df['time'] == time].sort_values('part')
            pose_vector = frame[[f'x{coord_suffix}', f'y{coord_suffix}', f'z{coord_suffix}']].values.flatten()
            poses.append(pose_vector)

        poses = np.array(poses)

        # Compute pairwise distances between poses
        if len(poses) > 1:
            pose_distances = cdist(poses, poses)
            diversity_score = np.mean(pose_distances)
            std_diversity = np.std(pose_distances)
        else:
            diversity_score = 0
            std_diversity = 0

        return {
            'diversity_score': diversity_score,
            'std_diversity': std_diversity,
            'n_poses': len(poses)
        }
    
    def detect_trajectory_outliers(self, std_threshold=3.0):
        """
        Detect unrealistic sharp turns in mouse trajectories.

        Analyzes the angular change in movement direction between consecutive frames
        and identifies outliers (e.g., boundary collision events).

        Parameters:
        -----------
        std_threshold : float
            Number of standard deviations from mean to consider as outlier

        Returns:
        --------
        dict : Dictionary with outlier analysis for each dataset
        """
        print("\n" + "="*60)
        print("TRAJECTORY OUTLIER DETECTION")
        print("="*60)

        outlier_results = {}

        for name, df in self.datasets.items():
            print(f"\nAnalyzing {name}...")

            # Compute COM trajectory over time
            times = sorted(df['time'].unique())
            com_trajectory = []

            for time in times:
                frame = df[df['time'] == time]
                # Use deformable body center of mass
                com = np.array([
                    frame['x_d'].mean(),
                    frame['y_d'].mean(),
                    frame['z_d'].mean()
                ])
                com_trajectory.append(com)

            com_trajectory = np.array(com_trajectory)

            if len(com_trajectory) < 3:
                print(f"  Insufficient frames for analysis")
                continue

            # Compute velocity vectors (direction of movement)
            velocity_vectors = np.diff(com_trajectory, axis=0)

            # Compute angular changes between consecutive velocity vectors
            angular_changes = []
            outlier_frames = []

            for i in range(len(velocity_vectors) - 1):
                v1 = velocity_vectors[i]
                v2 = velocity_vectors[i + 1]

                # Normalize vectors
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)

                # Skip if either vector is too small (mouse not moving)
                if v1_norm < 1e-6 or v2_norm < 1e-6:
                    continue

                v1_unit = v1 / v1_norm
                v2_unit = v2 / v2_norm

                # Compute angle between vectors using dot product
                cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                angular_changes.append(angle_deg)
                # Frame index (i+1 because we're looking at change from i to i+1)
                outlier_frames.append(i + 1)

            angular_changes = np.array(angular_changes)
            outlier_frames = np.array(outlier_frames)

            if len(angular_changes) == 0:
                print(f"  No valid angular changes computed")
                continue

            # Compute statistics
            mean_angle = np.mean(angular_changes)
            std_angle = np.std(angular_changes)
            median_angle = np.median(angular_changes)

            # Identify outliers (sharp turns)
            outlier_threshold = mean_angle + std_threshold * std_angle
            outlier_mask = angular_changes > outlier_threshold
            outlier_indices = outlier_frames[outlier_mask]
            outlier_angles = angular_changes[outlier_mask]

            # Count outliers
            n_outliers = np.sum(outlier_mask)
            outlier_percentage = (n_outliers / len(angular_changes)) * 100

            # Store results
            outlier_results[name] = {
                'angular_changes': angular_changes,
                'outlier_frames': outlier_indices,
                'outlier_angles': outlier_angles,
                'n_outliers': n_outliers,
                'outlier_percentage': outlier_percentage,
                'mean_angle': mean_angle,
                'std_angle': std_angle,
                'median_angle': median_angle,
                'threshold': outlier_threshold,
                'max_angle': np.max(angular_changes),
                'min_angle': np.min(angular_changes)
            }

            print(f"  Total frames analyzed: {len(angular_changes)}")
            print(f"  Mean angular change: {mean_angle:.2f}° ± {std_angle:.2f}°")
            print(f"  Median angular change: {median_angle:.2f}°")
            print(f"  Outlier threshold ({std_threshold}σ): {outlier_threshold:.2f}°")
            print(f"  Sharp turns detected: {n_outliers} ({outlier_percentage:.2f}%)")
            print(f"  Max angular change: {np.max(angular_changes):.2f}°")

            if n_outliers > 0:
                print(f"  Outlier angles: min={np.min(outlier_angles):.2f}°, max={np.max(outlier_angles):.2f}°")

        return outlier_results

    def plot_outlier_analysis(self, outlier_results):
        """Plot angular change distributions and outlier detection"""
        n_datasets = len(outlier_results)
        if n_datasets == 0:
            print("No outlier results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Angular change distributions
        ax = axes[0, 0]
        for name, results in outlier_results.items():
            ax.hist(results['angular_changes'], bins=50, alpha=0.5, label=name, density=True)
        ax.set_xlabel('Angular Change (degrees)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Angular Changes Between Frames')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=90, color='red', linestyle='--', alpha=0.5, label='90° (right angle)')
        ax.axvline(x=180, color='darkred', linestyle='--', alpha=0.5, label='180° (reversal)')

        # Plot 2: Cumulative distribution
        ax = axes[0, 1]
        for name, results in outlier_results.items():
            sorted_angles = np.sort(results['angular_changes'])
            cumulative = np.arange(1, len(sorted_angles) + 1) / len(sorted_angles)
            ax.plot(sorted_angles, cumulative, label=name, linewidth=2)
        ax.set_xlabel('Angular Change (degrees)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution of Angular Changes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=90, color='red', linestyle='--', alpha=0.5)

        # Plot 3: Outlier percentages
        ax = axes[1, 0]
        names = list(outlier_results.keys())
        percentages = [outlier_results[name]['outlier_percentage'] for name in names]
        colors = plt.cm.RdYlGn_r(np.array(percentages) / max(percentages) if max(percentages) > 0 else [0.5]*len(percentages))
        bars = ax.bar(range(len(names)), percentages, color=colors, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Sharp Turn Outliers (Boundary Collisions)')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        # Plot 4: Angular change statistics comparison
        ax = axes[1, 1]
        x_pos = np.arange(len(names))
        width = 0.25

        means = [outlier_results[name]['mean_angle'] for name in names]
        medians = [outlier_results[name]['median_angle'] for name in names]
        maxs = [outlier_results[name]['max_angle'] for name in names]

        ax.bar(x_pos - width, means, width, label='Mean', alpha=0.7, color='steelblue')
        ax.bar(x_pos, medians, width, label='Median', alpha=0.7, color='orange')
        ax.bar(x_pos + width, maxs, width, label='Max', alpha=0.7, color='crimson')

        ax.set_xlabel('Dataset')
        ax.set_ylabel('Angular Change (degrees)')
        ax.set_title('Angular Change Statistics')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir+'/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Additional plot: Time series of angular changes for each dataset
        fig, axes = plt.subplots(len(outlier_results), 1, figsize=(16, 4*len(outlier_results)))
        if len(outlier_results) == 1:
            axes = [axes]

        for idx, (name, results) in enumerate(outlier_results.items()):
            ax = axes[idx]

            # Plot angular changes over frames
            frames = np.arange(len(results['angular_changes']))
            ax.plot(frames, results['angular_changes'], linewidth=1, alpha=0.7, color='steelblue')

            # Mark outliers
            outlier_mask = results['angular_changes'] > results['threshold']
            outlier_x = frames[outlier_mask]
            outlier_y = results['angular_changes'][outlier_mask]
            ax.scatter(outlier_x, outlier_y, color='red', s=50, zorder=5, 
                      label=f'Outliers (n={results["n_outliers"]})', alpha=0.8)

            # Add threshold line
            ax.axhline(y=results['threshold'], color='red', linestyle='--', 
                      alpha=0.5, label=f'Threshold ({results["threshold"]:.1f}°)')
            ax.axhline(y=results['mean_angle'], color='green', linestyle='--', 
                      alpha=0.5, label=f'Mean ({results["mean_angle"]:.1f}°)')

            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Angular Change (degrees)')
            ax.set_title(f'{name} - Angular Changes Over Time')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, min(200, results['max_angle'] * 1.1))

        plt.tight_layout()
        plt.savefig(self.output_dir+'/angular_changes_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_all_datasets(self):
        """Run comprehensive analysis on all datasets"""
        print("\n" + "="*60)
        print("ANALYZING ALL DATASETS")
        print("="*60)

        for name, df in self.datasets.items():
            print(f"\nAnalyzing {name}...")

            # Center by COM
            df_centered = self.center_by_com(df)
            self.centered_datasets[name] = df_centered

            # Compute statistics
            stats = {}

            # Basic stats
            stats['n_frames'] = df['time'].nunique()
            stats['n_parts'] = df['part'].nunique()
            stats['duration_s'] = df['time'].max() - df['time'].min()

            # Coordinate ranges (centered)
            for coord in ['x_r', 'y_r', 'z_r', 'x_d', 'y_d', 'z_d']:
                stats[f'{coord}_mean'] = df_centered[coord].mean()
                stats[f'{coord}_std'] = df_centered[coord].std()
                stats[f'{coord}_range'] = df_centered[coord].max() - df_centered[coord].min()

            # Rigid body displacement (COM trajectory)
            com_trajectory = []
            for time in sorted(df['time'].unique()):
                frame = df[df['time'] == time]
                com = [frame['x_r'].mean(), frame['y_r'].mean(), frame['z_r'].mean()]
                com_trajectory.append(com)
            com_trajectory = np.array(com_trajectory)

            if len(com_trajectory) > 1:
                com_distances = np.sqrt(np.sum(np.diff(com_trajectory, axis=0)**2, axis=1))
                stats['total_distance_traveled'] = np.sum(com_distances)
                stats['avg_speed'] = stats['total_distance_traveled'] / stats['duration_s']
            else:
                stats['total_distance_traveled'] = 0
                stats['avg_speed'] = 0

            # Velocities
            velocities = self.compute_velocities(df_centered, '_d')
            stats['velocity_mean'] = np.mean(velocities)
            stats['velocity_std'] = np.std(velocities)
            stats['velocity_max'] = np.max(velocities) if len(velocities) > 0 else 0

            # Angular metrics
            angular_data = self.compute_angular_metrics(df_centered, '_d')
            stats['angular_span_mean'] = np.mean(angular_data['spans']) if len(angular_data['spans']) > 0 else 0
            stats['angular_span_std'] = np.std(angular_data['spans']) if len(angular_data['spans']) > 0 else 0
            stats['angular_velocity_mean'] = np.mean(angular_data['velocities']) if len(angular_data['velocities']) > 0 else 0
            stats['angular_velocity_std'] = np.std(angular_data['velocities']) if len(angular_data['velocities']) > 0 else 0

            # Pose diversity
            diversity = self.compute_pose_diversity(df_centered, '_d')
            stats.update(diversity)

            self.stats[name] = stats

            print(f"  Duration: {stats['duration_s']:.1f} s")
            print(f"  Avg velocity: {stats['velocity_mean']:.4f} ± {stats['velocity_std']:.4f}")
            print(f"  Pose diversity: {stats['diversity_score']:.4f}")

    def plot_coordinate_distributions(self):
        """Plot coordinate distributions for all datasets"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        coords = ['x_d', 'y_d', 'x_r', 'y_r']
        titles = ['X Deformable', 'Y Deformable', 
                  'X Rigid', 'Y Rigid']

        for idx, (coord, title) in enumerate(zip(coords, titles)):
            ax = axes[idx // 2, idx % 2]

            for name, df in self.centered_datasets.items():
                ax.hist(df[coord], bins=50, alpha=0.5, label=name, density=True)

            ax.set_xlabel(f'{title} (centered)')
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {title} Coordinates')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir+'/coordinate_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_velocity_distributions(self):
        """Plot velocity distributions"""
        plt.figure(figsize=(15, 5))

        # Velocity distributions
        for name, df in self.centered_datasets.items():
            velocities = self.compute_velocities(df, '_d')
            plt.hist(velocities, bins=50, alpha=0.5, label=name, density=True)

        plt.xlabel('Velocity (units/s)')
        plt.ylabel('Density')
        plt.title('Velocity Distributions (Deformable Body)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir+'/velocity_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()



    def plot_angular_metrics(self):
        """Plot angular span and velocity distributions"""
        plt.figure(figsize=(15, 5))

        # Angular velocities
        for name, df in self.centered_datasets.items():
            angular_data = self.compute_angular_metrics(df, '_d')
            if len(angular_data['velocities']) > 0:
                plt.hist(angular_data['velocities'], bins=50, alpha=0.5, label=name, density=True)

        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Density')
        plt.title('Distribution of Angular Velocities')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir+'/angular_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_trajectory_comparison(self):
        """Plot COM trajectories for all datasets"""
        fig = plt.figure(figsize=(15, 5))

        # 2D projection (XY)
        ax1 = fig.add_subplot(131)
        for name, df in self.datasets.items():
            com_x = df.groupby('time')['x_r'].mean()
            com_y = df.groupby('time')['y_r'].mean()
            ax1.plot(com_x, com_y, alpha=0.7, label=name, linewidth=2)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Mouse Trajectory (Top View - XY)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 2D projection (XZ)
        ax2 = fig.add_subplot(132)
        for name, df in self.datasets.items():
            com_x = df.groupby('time')['x_r'].mean()
            com_z = df.groupby('time')['z_r'].mean()
            ax2.plot(com_x, com_z, alpha=0.7, label=name, linewidth=2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('Mouse Trajectory (Side View - XZ)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 2D projection (YZ)
        ax3 = fig.add_subplot(133)
        for name, df in self.datasets.items():
            com_y = df.groupby('time')['y_r'].mean()
            com_z = df.groupby('time')['z_r'].mean()
            ax3.plot(com_y, com_z, alpha=0.7, label=name, linewidth=2)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('Mouse Trajectory (Front View - YZ)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir+'/trajectory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_summary_statistics(self):
        """Create summary comparison plots"""
        stats_df = pd.DataFrame(self.stats).T

        # Select key metrics
        metrics = ['avg_speed', 'angular_velocity_mean', 'total_distance_traveled']

        fig, axes = plt.subplots(1, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            if metric in stats_df.columns:
                stats_df[metric].plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir+'/summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def compute_dataset_similarity(self):
        """Compute similarity between datasets using Wasserstein distance"""
        dataset_names = list(self.centered_datasets.keys())
        n_datasets = len(dataset_names)

        # Compute velocity distributions for each dataset
        velocity_dists = {}
        for name, df in self.centered_datasets.items():
            velocities = self.compute_velocities(df, '_d')
            velocity_dists[name] = velocities

        # Compute pairwise Wasserstein distances
        similarity_matrix = np.zeros((n_datasets, n_datasets))

        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i == j:
                    similarity_matrix[i, j] = 0
                else:
                    dist = wasserstein_distance(velocity_dists[name1], velocity_dists[name2])
                    similarity_matrix[i, j] = dist

        # Plot similarity matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_matrix, cmap='viridis_r', aspect='auto')

        ax.set_xticks(np.arange(n_datasets))
        ax.set_yticks(np.arange(n_datasets))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.set_yticklabels(dataset_names)

        # Add values to cells
        for i in range(n_datasets):
            for j in range(n_datasets):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white", fontsize=10)

        ax.set_title('Dataset Similarity Matrix\n(Wasserstein Distance - Lower is More Similar)')
        plt.colorbar(im, ax=ax, label='Distance')
        plt.tight_layout()
        plt.savefig(self.output_dir+'/dataset_similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        return similarity_matrix, dataset_names

    def generate_report(self):
        """Generate a comprehensive text report"""
        report = []
        report.append("="*70)
        report.append("MOUSE MOVEMENT DATASET ANALYSIS REPORT")
        report.append("="*70)
        report.append("")

        stats_df = pd.DataFrame(self.stats).T

        report.append("DATASET SUMMARY")
        report.append("-"*70)
        for name in self.stats.keys():
            report.append(f"\n{name}:")
            report.append(f"  Number of frames: {self.stats[name]['n_frames']}")
            report.append(f"  Number of body parts: {self.stats[name]['n_parts']}")
            report.append(f"  Duration: {self.stats[name]['duration_s']:.1f} s")
            report.append(f"  Total distance traveled: {self.stats[name]['total_distance_traveled']:.2f}")
            report.append(f"  Average speed: {self.stats[name]['avg_speed']:.4f}cm/s")
            report.append(f"  Velocity (mean ± std): {self.stats[name]['velocity_mean']:.4f} ± {self.stats[name]['velocity_std']:.4f}cm/s")
            report.append(f"  Max velocity: {self.stats[name]['velocity_max']:.4f}cm/s")
            report.append(f"  Angular velocity: {self.stats[name]['angular_velocity_mean']:.4f} ± {self.stats[name]['angular_velocity_std']:.4f}rad/s")
            report.append(f"  Pose diversity score: {self.stats[name]['diversity_score']:.4f}")

        report_text = "\n".join(report)

        with open(self.output_dir+'/analysis_report.txt', 'w') as f:
            f.write(report_text)

        stats_df.to_csv(self.output_dir+'/report.csv')

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*70)
        print("MOUSE MOVEMENT DATASET ANALYZER")
        print("="*70)

        # Load data
        self.load_datasets()

        # Analyze
        self.analyze_all_datasets()

        # Generate plots
        self.plot_coordinate_distributions()
        self.plot_velocity_distributions()
        self.plot_angular_metrics()
        self.plot_trajectory_comparison()
        self.plot_summary_statistics()
        outliers = self.detect_trajectory_outliers()
        self.plot_outlier_analysis(outliers)
        # Similarity analysis
        self.compute_dataset_similarity()

        # Generate report
        self.generate_report()


if __name__ == "__main__":
    analyzer = MouseDatasetAnalyzer(data_folder='data')
    analyzer.run_full_analysis()