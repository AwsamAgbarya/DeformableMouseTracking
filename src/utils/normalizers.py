import torch
import numpy as np

class Normalizer3D:
    """
    Normalize 3D coordinates to [-1, 1].
    X and Y share a single isotropic scale (so post-normalization Z-axis 
    rotations remain valid, angle-preserving transforms).
    Z uses its own independent scale, since Z varies far less than X/Y 
    and would otherwise be squashed by a shared XY range.
    """

    def __init__(self):
        self.global_centering = None
        self.center_x = None
        self.center_y = None
        self.center_z = None
        self.scale_xy = None
        self.scale_z = None
        self.is_fitted = False

    def fit(self, keypoints_3d):
        """
        Compute normalization bounds:
        - XY: isotropic scale using the max range across X and Y combined.
        - Z: independent scale using only Z's own range.
        """
        if isinstance(keypoints_3d, torch.Tensor):
            keypoints_3d = keypoints_3d.detach().cpu().numpy()

        self.global_centering = keypoints_3d.mean(axis=1, keepdims=True)

        x_min, x_max = keypoints_3d[..., 0].min(), keypoints_3d[..., 0].max()
        y_min, y_max = keypoints_3d[..., 1].min(), keypoints_3d[..., 1].max()
        z_min, z_max = keypoints_3d[..., 2].min(), keypoints_3d[..., 2].max()

        self.center_x = (x_max + x_min) / 2.0
        self.center_y = (y_max + y_min) / 2.0
        self.center_z = (z_max + z_min) / 2.0

        range_x = x_max - x_min
        range_y = y_max - y_min
        range_z = z_max - z_min

        margin = 0.05

        # Isotropic scale for X and Y jointly (preserves shape/angles under Z-rotation)
        print(f'Range X: {range_x}, Range Y: {range_y}, Range Z: {range_z}')
        max_range_xy = max(range_x, range_y)
        self.scale_xy = (max_range_xy * (1 + margin)) / 2.0

        # Independent scale for Z (prevents Z from being squashed by XY range)
        self.scale_z = (range_z * (1 + margin)) / 2.0

        self.is_fitted = True
        print(f"✓ XY Isotropic Scale: {self.scale_xy:.4f} | Z Scale: {self.scale_z:.4f}\n\n")
        return self

    def normalize(self, keypoints_3d, center: bool):
        """Normalize 3D keypoints to [-1, 1]"""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted! Call fit() first.")
        kpts_3d = keypoints_3d.clone() if isinstance(keypoints_3d, torch.Tensor) else keypoints_3d.copy()

        if center:
            kpts_3d = kpts_3d - self.global_centering

        kpts_3d[..., 0] = (kpts_3d[..., 0] - self.center_x) / self.scale_xy
        kpts_3d[..., 1] = (kpts_3d[..., 1] - self.center_y) / self.scale_xy
        kpts_3d[..., 2] = (kpts_3d[..., 2] - self.center_z) / self.scale_z

        return kpts_3d

    def denormalize(self, keypoints_3d_norm, center: bool):
        """Denormalize from [-1, 1] back to original scale"""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted! Call fit() first.")
        kpts_3d = keypoints_3d_norm.clone() if isinstance(keypoints_3d_norm, torch.Tensor) else keypoints_3d_norm.copy()

        kpts_3d[..., 0] = keypoints_3d_norm[..., 0] * self.scale_xy + self.center_x
        kpts_3d[..., 1] = keypoints_3d_norm[..., 1] * self.scale_xy + self.center_y
        kpts_3d[..., 2] = keypoints_3d_norm[..., 2] * self.scale_z + self.center_z

        if center:
            kpts_3d = kpts_3d + self.global_centering

        return kpts_3d

    def save(self, path):
        np.save(path, {
            "center_x": self.center_x, "center_y": self.center_y, "center_z": self.center_z,
            "scale_xy": self.scale_xy, "scale_z": self.scale_z
        })

    def load(self, path):
        p = np.load(path, allow_pickle=True).item()
        self.center_x, self.center_y, self.center_z = p["center_x"], p["center_y"], p["center_z"]
        self.scale_xy, self.scale_z = p["scale_xy"], p["scale_z"]
        self.is_fitted = True
        return self



class NormalizerMV:
    """
    Normalize 2D coordinates to [-1, 1] while preserving aspect ratio.
    Uses the largest dimension's range to scale both X and Y equally.
    """
    
    def __init__(self):
        # 2D normalization
        self.center_x = None
        self.center_y = None
        self.scale = None
        
        # Depth normalization
        self.depth_min = None
        self.depth_max = None
        self.is_fitted = False
    
    def fit(self, keypoints_2d, depths):
        """
        Compute normalization bounds using the maximum range across X and Y.
        """
        if isinstance(keypoints_2d, torch.Tensor):
            keypoints_2d = keypoints_2d.detach().cpu().numpy()
        
        # Find the global center
        assert keypoints_2d.dim == 100, "Must fix the centering effect first"
        self.global_centering = keypoints_2d.mean(dim=1, keepdim=True)
        
        # Find Center
        x_min, x_max = keypoints_2d[..., 0].min(), keypoints_2d[..., 0].max()
        y_min, y_max = keypoints_2d[..., 1].min(), keypoints_2d[..., 1].max()
        
        self.center_x = (x_max + x_min) / 2.0
        self.center_y = (y_max + y_min) / 2.0
        
        # Range
        range_x = x_max - x_min
        range_y = y_max - y_min
        
        # Isotropic scaling
        max_range = max(range_x, range_y)
        margin = 0.05
        self.scale = (max_range * (1 + margin)) / 2.0 

        if depths is not None:
            if isinstance(depths, torch.Tensor):
                depths = depths.detach().cpu().numpy()
            self.depth_min = depths.min()
            self.depth_max = depths.max()
            
            # Add margin to depth
            d_range = self.depth_max - self.depth_min
            self.depth_min -= d_range * margin
            self.depth_max += d_range * margin
        
        self.is_fitted = True
        print(f"✓ Isotropic Scale: {self.scale:.4f}")
        return self
    
    def normalize(self, keypoints_2d):
        """Normalize 2D keypoints to [-1, 1]"""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted! Call fit() first.")
        kpts_2d = keypoints_2d.clone() if isinstance(keypoints_2d, torch.Tensor) else keypoints_2d.copy()
        
        # Center the data globally
        kpts_2d = kpts_2d - self.global_centering
        
        # Normalize x and y
        kpts_2d[..., 0] = (kpts_2d[..., 0] - self.center_x) / self.scale
        kpts_2d[..., 1] = (kpts_2d[..., 1] - self.center_y) / self.scale
        
        return kpts_2d
    
    def denormalize(self, keypoints_2d_norm):
        """Denormalize from [-1, 1] back to original scale"""
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted! Call fit() first.")
        kpts_2d = keypoints_2d_norm.clone() if isinstance(keypoints_2d_norm, torch.Tensor) else keypoints_2d_norm.copy()
        
        # Denormalize x and y
        kpts_2d[..., 0] = keypoints_2d_norm[..., 0] * self.scale + self.center_x
        kpts_2d[..., 1] = keypoints_2d_norm[..., 1] * self.scale + self.center_y
        
        # Denormalize the global centering
        kpts_2d = kpts_2d + self.global_centering
        
        return kpts_2d
    
    def normalize_depth(self, depths):
        """Normalize depths to [-1, 1]"""
        if self.depth_min is None or self.depth_max is None:
            raise ValueError("Depth normalization not fitted!")
        
        is_torch = isinstance(depths, torch.Tensor)
        depths_norm = depths.clone() if is_torch else depths.copy()
        
        depths_norm = 2 * (depths - self.depth_min) / (self.depth_max - self.depth_min) - 1
        return depths_norm
    
    def denormalize_depth(self, depths_norm):
        """Denormalize depths from [-1, 1] to original scale"""
        if self.depth_min is None or self.depth_max is None:
            raise ValueError("Depth normalization not fitted!")
        
        is_torch = isinstance(depths_norm, torch.Tensor)
        depths = depths_norm.clone() if is_torch else depths_norm.copy()
        
        depths = (depths_norm + 1) / 2 * (self.depth_max - self.depth_min) + self.depth_min
        return depths

    def save(self, path):
        np.save(path, {"center_x": self.center_x, "center_y": self.center_y, "scale": self.scale, "depth_min": self.depth_min, 'depth_max':self.depth_max})

    def load(self, path):
        p = np.load(path, allow_pickle=True).item()
        self.center_x, self.center_y, self.scale = p["center_x"], p["center_y"], p["scale"]
        self.depth_max, self.depth_min = p['depth_max'], p['depth_min']
        self.is_fitted = True
        return self