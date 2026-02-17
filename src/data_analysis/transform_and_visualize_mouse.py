"""
Mouse Data Transformation and Visualization
===========================================
Applies translation and rotation invariance transformations to mouse data:
- Translation invariance: Center by COM (deformable)
- Rotation invariance: Align heading direction to X-axis
- Visualization: 3D projection and animation with speed control
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import seaborn as sns
from sklearn.decomposition import PCA

sns.set_style("whitegrid")


class MouseDataTransformer:
    """Transform mouse data for translation and rotation invariance"""

    def __init__(self, csv_path):
        self.output_dir = './data/analysis'
        self.csv_path = Path(csv_path)
        self.df = None
        self.transformed_df = None
        self.part_names = None

    def load_data(self):
        """Load CSV data"""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        self.part_names = sorted(self.df['part'].unique())

        print(f"  Frames: {self.df['time'].nunique()}")
        print(f"  Body parts: {len(self.part_names)}")
        print(f"  Parts: {self.part_names}")

        return self.df
    
    def compute_heading_from_rigid_vector(self, df, reference_part=('head center','tail base')):
        """
        Parameters:
        -----------
        df : DataFrame
            Original dataframe
        reference_part : tuple of str, optional
            (part1, part2) to define the reference vector.
            If None, uses first and last parts alphabetically.
            Example: ('tail_base', 'nose')

        Returns:
        --------
        list of np.ndarray : Rotation matrices for each frame (3x3)
        """
        times = sorted(df['time'].unique())
        part_names = sorted(df['part'].unique())

        # Define reference vector parts
        if reference_part is None:
            # Use first and last parts (e.g., tail to head direction)
            part1 = part_names[0]
            part2 = part_names[-1]
        else:
            part1, part2 = reference_part

        print(f"  Using reference vector: {part1} → {part2}")

        rotation_matrices = []

        for time in times:
            frame = df[df['time'] == time]

            # Get positions of reference parts
            pos1 = frame[frame['part'] == part1][['x_r', 'y_r', 'z_r']].values
            pos2 = frame[frame['part'] == part2][['x_r', 'y_r', 'z_r']].values

            if len(pos1) == 0 or len(pos2) == 0:
                # Missing data, use identity
                rotation_matrices.append(np.eye(3))
                continue

            pos1 = pos1[0]
            pos2 = pos2[0]

            # Compute reference vector
            ref_vector = pos2 - pos1
            ref_vector_2d = ref_vector[:2]  # XY plane only

            # Handle zero vector
            if np.linalg.norm(ref_vector_2d) < 1e-6:
                rotation_matrices.append(np.eye(3))
                continue

            # Normalize
            ref_vector_2d = ref_vector_2d / np.linalg.norm(ref_vector_2d)

            # Target direction (X-axis)
            target = np.array([1.0, 0.0])

            # Compute rotation angle
            angle = np.arctan2(ref_vector_2d[1], ref_vector_2d[0])

            # Create 3D rotation matrix around Z-axis
            cos_a = np.cos(-angle)  # Negative to rotate TO target
            sin_a = np.sin(-angle)

            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])

            rotation_matrices.append(R)

        return rotation_matrices

    def transform_data(self, method='velocity'):
        """
        Apply translation and rotation invariance transformations.

        Parameters:
        -----------
        method : str
            Method for computing heading direction:
            - 'velocity': Use COM velocity (displacement) direction
            - 'pca': Use direction of largest variance per frame
            - 'rigid': Use rigid body COM displacement
        """
        print(f"\nTransforming data (method: {method})...")

        transformed_data = []
        times = sorted(self.df['time'].unique())

        # Store COM trajectory for velocity-based heading
        com_trajectory = []
        rigid_com_trajectory = []

        # First pass: compute COM trajectories
        for time in times:
            frame = self.df[self.df['time'] == time]

            # Deformable COM
            com = np.array([
                frame['x_d'].mean(),
                frame['y_d'].mean(),
                frame['z_d'].mean()
            ])
            com_trajectory.append(com)

            # Rigid COM
            rigid_com = np.array([
                frame['x_r'].mean(),
                frame['y_r'].mean(),
                frame['z_r'].mean()
            ])
            rigid_com_trajectory.append(rigid_com)

        com_trajectory = np.array(com_trajectory)
        rigid_com_trajectory = np.array(rigid_com_trajectory)

        rotation_matrices = self.compute_heading_from_rigid_vector(self.df)

        # Second pass: transform each frame
        for idx, time in enumerate(times):
            frame = self.df[self.df['time'] == time].copy()

            # 1. Translation invariance: Center by deformable COM
            com = com_trajectory[idx]

            frame['x_d_centered'] = frame['x_d'] - com[0]
            frame['y_d_centered'] = frame['y_d'] - com[1]
            frame['z_d_centered'] = frame['z_d'] - com[2]

            # 2. Apply rotation
            R = rotation_matrices[idx]
            centered_coords = frame[['x_d_centered', 'y_d_centered', 'z_d_centered']].values
            rotated_coords = (R @ centered_coords.T).T

            frame['x_aligned'] = rotated_coords[:, 0]
            frame['y_aligned'] = rotated_coords[:, 1]
            frame['z_aligned'] = rotated_coords[:, 2]

            transformed_data.append(frame)

        self.transformed_df = pd.concat(transformed_data, ignore_index=True)
        print(f"  Transformation complete!")
        print(f"  New columns: x_aligned, y_aligned, z_aligned")

        return self.transformed_df

    def plot_coordinate_spans_2d(self, n_frames=None):
        """
        Plot 2D (XY) coordinate spans for each body part.
        Z-dimension is collapsed/ignored.
        """
        if self.transformed_df is None:
            raise ValueError("Data not transformed yet. Call transform_data() first.")

        print("\nPlotting 2D coordinate spans...")

        # Use all frames or subset
        if n_frames:
            times = sorted(self.transformed_df['time'].unique())[:n_frames]
            plot_df = self.transformed_df[self.transformed_df['time'].isin(times)]
        else:
            plot_df = self.transformed_df

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot 1: All parts overlaid
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.part_names)))

        for idx, part in enumerate(self.part_names):
            part_data = plot_df[plot_df['part'] == part]
            ax.scatter(part_data['x_aligned'], part_data['y_aligned'], 
                      alpha=0.3, s=2, color=colors[idx], label=part)

        ax.set_xlabel('X (aligned, cm)')
        ax.set_ylabel('Y (aligned, cm)')
        ax.set_title('2D Coordinate Spans (All Parts Overlaid)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        plt.savefig(self.output_dir+'/coordinate_spans_2d.png', dpi=300, bbox_inches='tight')
        print("Saved: coordinate_spans_2d.png")
        plt.close()

        # Per-part coordinate ranges
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for idx, part in enumerate(self.part_names):
            if idx >= len(axes):
                break

            ax = axes[idx]
            part_data = plot_df[plot_df['part'] == part]

            ax.scatter(part_data['x_aligned'], part_data['y_aligned'], 
                      alpha=0.3, s=5, color=colors[idx])
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_title(f'{part}')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

        plt.tight_layout()
        plt.savefig(self.output_dir+'/coordinate_spans_per_part.png', dpi=300, bbox_inches='tight')
        print("Saved: coordinate_spans_per_part.png")
        plt.close()

    def create_animation(self, n_frames=100, output_file='mouse_animation.gif', 
                        speed_factor=1.0, view_3d=True):
        """
        Create an animation of the aligned mouse movement.

        Parameters:
        -----------
        n_frames : int
            Number of frames to animate
        output_file : str
            Output filename for animation
        speed_factor : float
            Speed multiplier for animation (default 1.0 = real-time)
            < 1.0 = slower (e.g., 0.5 = half speed, 0.3 = 30% speed)
            > 1.0 = faster (e.g., 2.0 = double speed)
        view_3d : bool
            If True, create 3D visualization. If False, create 2D (XY plane)
        """
        if self.transformed_df is None:
            raise ValueError("Data not transformed yet. Call transform_data() first.")

        print(f"\nCreating {'3D' if view_3d else '2D'} animation ({n_frames} frames, speed={speed_factor}x)...")

        times = sorted(self.transformed_df['time'].unique())[:n_frames]

        # Setup figure
        if view_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 10))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.part_names)))

        # Get coordinate ranges for consistent axis limits
        all_data = self.transformed_df[self.transformed_df['time'].isin(times)]
        x_min, x_max = all_data['x_aligned'].min(), all_data['x_aligned'].max()
        y_min, y_max = all_data['y_aligned'].min(), all_data['y_aligned'].max()
        z_min, z_max = all_data['z_aligned'].min(), all_data['z_aligned'].max()

        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min if view_3d else 1
        padding = 0.2

        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

        if view_3d:
            ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
            ax.set_zlabel('Z (aligned, cm)', fontsize=10)

        # Initialize scatter plots for each part
        scatters = []
        for idx, part in enumerate(self.part_names):
            if view_3d:
                scatter = ax.scatter([], [], [], s=100, color=colors[idx], 
                                   label=part, alpha=0.8, edgecolors='black', linewidths=1)
            else:
                scatter = ax.scatter([], [], s=100, color=colors[idx], 
                                   label=part, alpha=0.8, edgecolors='black', linewidths=1)
            scatters.append(scatter)

        # Add lines connecting parts (skeleton)
        lines = []
        for i in range(1, len(self.part_names)):
            if view_3d:
                line, = ax.plot([], [], [], 'k-', alpha=0.3, linewidth=2)
            else:
                line, = ax.plot([], [], 'k-', alpha=0.3, linewidth=1)
            lines.append(line)

        ax.set_xlabel('X (aligned, cm)', fontsize=10)
        ax.set_ylabel('Y (aligned, cm)', fontsize=10)
        ax.set_title('Mouse Movement (Aligned - 3D View)' if view_3d else 'Mouse Movement (Aligned)', 
                     fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        if not view_3d:
            ax.axis('equal')

        # Add title text
        if view_3d:
            title = fig.suptitle('', fontsize=12, fontweight='bold')
        else:
            title = ax.text(0.5, 1.05, '', transform=ax.transAxes, 
                           ha='center', fontsize=12, fontweight='bold')

        # For 3D, add rotation
        rotation_angle = 0

        def init():
            for scatter in scatters:
                if view_3d:
                    scatter._offsets3d = ([], [], [])
                else:
                    scatter.set_offsets(np.empty((0, 2)))
            title.set_text('')
            return scatters + lines + [title]

        def update(frame_idx):
            nonlocal rotation_angle

            time = times[frame_idx]
            frame_data = self.transformed_df[self.transformed_df['time'] == time]

            # Update scatter plots
            positions = []
            for idx, part in enumerate(self.part_names):
                part_data = frame_data[frame_data['part'] == part]
                if not part_data.empty:
                    x = part_data['x_aligned'].values[0]
                    y = part_data['y_aligned'].values[0]
                    z = part_data['z_aligned'].values[0]

                    if view_3d:
                        scatters[idx]._offsets3d = ([x], [y], [z])
                    else:
                        scatters[idx].set_offsets([[x, y]])

                    positions.append([x, y, z])

            title.set_text(f'Frame {frame_idx+1}/{len(times)} | Time: {time:.2f}s | Speed: {speed_factor}x')

            return scatters + lines + [title]

        # Compute interval based on speed_factor
        # Assuming original data is sampled at some rate, we want to display it slower/faster
        if len(times) > 1:
            avg_dt = np.mean(np.diff(times))  # Average time between frames in seconds
            # Convert to milliseconds and adjust by speed factor
            interval_ms = (avg_dt * 1000) / speed_factor
        else:
            interval_ms = 50  # Default 50ms

        # Compute FPS for saving
        fps = int(1000 / interval_ms) if interval_ms > 0 else 20
        fps = max(1, min(fps, 30))  # Clamp between 1-30 fps

        anim = FuncAnimation(fig, update, init_func=init, 
                           frames=len(times), interval=interval_ms, blit=False)

        # Save animation
        writer = PillowWriter(fps=fps)
        anim.save(self.output_dir+'/'+output_file, writer=writer)
        print(f"Saved: {self.output_dir+'/'+output_file} (FPS: {fps}, interval: {interval_ms:.1f}ms)")
        plt.close()

def main():
    """Main execution"""
    import sys

    # Get CSV file path from command line or use default
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'data/your_file.csv'  # Change this to your file

    print("="*70)
    print("MOUSE DATA TRANSFORMATION AND VISUALIZATION")
    print("="*70)

    # Initialize transformer
    transformer = MouseDataTransformer(csv_path)

    # Load data
    transformer.load_data()

    # Transform data (try different methods)
    method = 'rigid'  # Options: 'velocity', 'pca', 'rigid'
    transformer.transform_data(method=method)

    # Plot 2D coordinate spans
    transformer.plot_coordinate_spans_2d()

    # Create animation (3D view, slower speed)
    transformer.create_animation(n_frames=200, 
                                output_file='mouse_animation_3d.gif',
                                speed_factor=0.5,  # 30% speed (slower than real-time)
                                view_3d=True)


if __name__ == "__main__":
    main()