import torch
import warnings

class MultiViewTriangulation:
    def __init__(self, projection_matrices, device='cpu'):
        self.P = projection_matrices.to(device)
        self.V = projection_matrices.shape[0]
        self.device = device
        
    def triangulate(self, points_2d, confidences= None):
        """
        Weighted Direct Linear Transform (DLT) triangulation using SVD
        
        Args:
            points_2d: [N, V, S, 2] - 2D points (N instances, V views, S keypoints)
            confidences: [N, V, S] - Optional weights/confidences for each point (e.g., from 0.0 to 1.0).
            
        Returns:
            points_3d: [N, S, 3] - triangulated 3D points
        """
        N, V, S, _ = points_2d.shape
        points_3d = torch.zeros(N, S, 3, device=self.device, dtype=points_2d.dtype)
        
        for n in range(N):
            for s in range(S):
                # Build linear system A @ X = 0 for this keypoint
                A_list = []
                for v in range(V):
                    x, y = points_2d[n, v, s]
                    P = self.P[v]  # [3, 4]

                    if confidences is not None:
                        w = confidences[n, v, s]
                    else:
                        w = 1.0
                        
                    # Multiply the equations by the weight
                    A_list.append(w * (x * P[2, :] - P[0, :]))
                    A_list.append(w * (y * P[2, :] - P[1, :]))
                
                # Each valid view gives 2 equations. We still need at least 4 equations total.
                if len(A_list) < 4:  
                    points_3d[n, s] = torch.tensor([float('nan')] * 3, device=self.device)
                    continue
                    
                A = torch.stack(A_list) 
                
                # SVD solution: minimize ||A @ X|| subject to ||X|| = 1
                try:
                    U, S_vals, Vt = torch.linalg.svd(A, full_matrices=False)
                    X = Vt[-1, :]  # Last right singular vector (smallest singular value)
                    
                    # Dehomogenize: [x, y, z, w] -> [x/w, y/w, z/w]
                    if abs(X[3].item()) > 1e-10:
                        points_3d[n, s] = X[:3] / X[3]
                    else:
                        points_3d[n, s] = torch.tensor([float('nan')] * 3, device=self.device)
                except Exception as e:
                    warnings.warn(f"DLT failed for instance {n}, keypoint {s}: {e}")
                    points_3d[n, s] = torch.tensor([float('nan')] * 3, device=self.device)
                    
        return points_3d

class SkeletonAligner:
    def __init__(self, device = 'cuda'):
        self.device = device

    def align(self, noisy_points, template, confidence_weights = None, method = 'ransac', ransac_iterations = 100, ransac_threshold = 0.05, min_sample_size = 3, min_inlier_ratio = 0.6):
        """
        Main interface to align template to noisy points.
        
        Args:
            noisy_points: (N, 3) Observed/deformed keypoints
            template: (N, 3) Canonical skeleton template
            confidence_weights: (N,) Optional confidence scores (e.g., higher for ground truth)
            method: 'weighted' for standard Procrustes, 'ransac' for robust alignment
            ransac_iterations: Number of RANSAC trials (if method='ransac')
            ransac_threshold: Distance threshold for inliers (if method='ransac')
            min_sample_size: Min points for rigid transform (if method='ransac')
            min_inlier_ratio: Min fraction of inliers required (if method='ransac')
            
        Returns:
            aligned_points: (N, 3) Rigidly aligned template
            rotation: (3, 3) Optimal rotation matrix
            translation: (3,) Optimal translation vector
            inlier_mask: (N,) Boolean mask of inliers (all True if method='weighted')
        """
        noisy_points = noisy_points.to(self.device)
        template = template.to(self.device)
        
        if confidence_weights is None:
            confidence_weights = torch.ones(template.shape[0], device=self.device)
        else:
            confidence_weights = confidence_weights.to(self.device)

        if method == 'weighted':
            aligned, R, t = self._weighted_procrustes(template, noisy_points, confidence_weights)
            inlier_mask = torch.ones(template.shape[0], dtype=torch.bool, device=self.device)
            return aligned, R, t, inlier_mask
            
        elif method == 'ransac':
            return self._extract_deformable_coordinates_ransac(
                noisy_points, template, confidence_weights, 
                ransac_iterations, ransac_threshold, min_sample_size, min_inlier_ratio
            )
            
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'weighted' or 'ransac'.")

    def _weighted_procrustes(self, template, noisy_points, weights):
        """Weighted Kabsch/Procrustes alignment."""
        # Normalize weights
        W = weights / (weights.sum() + 1e-8)
        
        # Weighted centroids
        centroid_template = (template * W[:, None]).sum(dim=0)
        centroid_noisy = (noisy_points * W[:, None]).sum(dim=0)
        
        # Center points
        template_centered = template - centroid_template
        noisy_centered = noisy_points - centroid_noisy
        
        # Weighted covariance
        H = (template_centered.T * W[None, :]) @ noisy_centered
        
        # SVD
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation
        t = centroid_noisy - R @ centroid_template
        
        # Apply transformation
        aligned = (R @ template.T).T + t
        return aligned, R, t

    def _fit_rigid_transform(self, source, target):
        """Closed-form rigid transformation from minimal point set."""
        centroid_source = source.mean(dim=0)
        centroid_target = target.mean(dim=0)
        
        source_centered = source - centroid_source
        target_centered = target - centroid_target
        
        H = source_centered.T @ target_centered
        
        U, S, Vt = torch.linalg.svd(H)
        R = Vt.T @ U.T
        
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        t = centroid_target - R @ centroid_source
        return R, t

    def _extract_deformable_coordinates_ransac(self, noisy_points, template, confidence_weights, ransac_iterations, ransac_threshold, min_sample_size, min_inlier_ratio):
        """RANSAC-enhanced alignment."""
        num_points = template.shape[0]
        best_inliers = None
        best_num_inliers = 0
        best_R = None
        best_t = None
        
        for _ in range(ransac_iterations):
            # Weighted sampling
            sample_probs = confidence_weights / (confidence_weights.sum() + 1e-8)
            sample_indices = torch.multinomial(sample_probs, min_sample_size, replacement=False)
            
            try:
                R_sample, t_sample = self._fit_rigid_transform(
                    template[sample_indices],
                    noisy_points[sample_indices]
                )
            except RuntimeError:
                # Skip degenerate subsets (e.g. SVD failure on collinear points)
                continue
                
            # Evaluate consensus
            transformed = (R_sample @ template.T).T + t_sample
            distances = torch.norm(transformed - noisy_points, dim=1)
            inliers = distances < ransac_threshold
            num_inliers = inliers.sum().item()
            
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = inliers
                best_R = R_sample
                best_t = t_sample
                
        # Check validity of best model
        if best_inliers is None or (best_num_inliers / num_points) < min_inlier_ratio:
            print(f"WARNING: RANSAC failed to find sufficient inliers "
                  f"({best_num_inliers}/{num_points}). Falling back to weighted Procrustes on all points.")
            return self._weighted_procrustes(template, noisy_points, confidence_weights) + (torch.ones(num_points, dtype=torch.bool, device=self.device),)
            
        # Refine using all inliers and their confidence weights
        inlier_weights = confidence_weights.clone()
        inlier_weights[~best_inliers] = 0.0
        
        if inlier_weights.sum() > 0:
            aligned, R_refined, t_refined = self._weighted_procrustes(template, noisy_points, inlier_weights)
        else:
            aligned = (best_R @ template.T).T + best_t
            R_refined, t_refined = best_R, best_t
            
        return aligned, R_refined, t_refined, best_inliers
