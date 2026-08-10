import torch
import torch.nn.functional as F
from collections import deque

class Masker:
    def __init__(self, dimensions, mask_strategy='constant',mask_min=0.3, mask_max=0.7,
                 warmup_epochs=50, seed=42, n_control=3,
                 parent=None, bone_lengths=None, corr_tau=1.5, random_frac=0.3):

        self.batch, self.window, self.parts = dimensions
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.mask_strategy = mask_strategy
        self.warmup_epochs = warmup_epochs
        self.n_control = n_control
        self.corr_tau = corr_tau
        self.generator = torch.Generator().manual_seed(seed)
        self.random_frac = random_frac

        self.corr = build_group_correlation(self.parts, parent, bone_lengths, corr_tau)
        self._chol = torch.linalg.cholesky(self.corr + 1e-5 * torch.eye(self.parts))

    def get_mask(self, epoch=0, batch_size=None):
        if batch_size is not None:
            self.batch = batch_size
        if self.mask_strategy == 'constant':
            return self._exact(self.mask_min)
        elif self.mask_strategy == 'linear':
            return self._linear(epoch)
        elif self.mask_strategy == 'random':
            return self._random()
        elif self.mask_strategy == 'temporal':
            return self._temporal(epoch)
        else:
            raise ValueError(f'Invalid mask strategy: {self.mask_strategy}')

    def _exact(self, ratio):
        """Each batch entry gets a random mask with exactly `ratio` proportion hidden."""
        masks = torch.stack([
            exact_mask(self.parts, ratio, generator=self.generator).squeeze(-1)
            for _ in range(self.batch)
        ])
        return masks  # (batch, parts), True = visible

    def _linear(self, epoch):
        """Linearly ramp ratio from mask_min to mask_max over warmup_epochs."""
        t = min(epoch / max(self.warmup_epochs, 1), 1.0)
        ratio = self.mask_min + t * (self.mask_max - self.mask_min)
        return self._exact(ratio)

    def _random(self):
        """Each batch entry samples its own ratio from U[mask_min, mask_max]."""
        masks = []
        for _ in range(self.batch):
            ratio = torch.empty(1).uniform_(self.mask_min, self.mask_max, generator=self.generator).item()
            masks.append(exact_mask(self.parts, ratio, generator=self.generator).squeeze(-1))
        return torch.stack(masks)  # (batch, parts)

    def _enforce_bounds(self, hidden: torch.Tensor, scores: torch.Tensor = None):
        """hidden: (K,) bool. Nudges the hidden count into [min_hide, max_hide]"""
        K = self.parts
        min_hide = int(round(self.mask_min * K))
        max_hide = int(round(self.mask_max * K))
        n_hidden = int(hidden.sum().item())

        if n_hidden < min_hide:
            visible_idx = torch.nonzero(~hidden, as_tuple=True)[0]
            need = min_hide - n_hidden
            if scores is not None:
                order = torch.argsort(scores[visible_idx], descending=True)
                pick = visible_idx[order[:need]]
            else:
                pick = visible_idx[torch.randperm(len(visible_idx), generator=self.generator)[:need]]
            hidden[pick] = True

        elif n_hidden > max_hide:
            hidden_idx = torch.nonzero(hidden, as_tuple=True)[0]
            excess = n_hidden - max_hide
            if scores is not None:
                order = torch.argsort(scores[hidden_idx], descending=False)
                pick = hidden_idx[order[:excess]]
            else:
                pick = hidden_idx[torch.randperm(len(hidden_idx), generator=self.generator)[:excess]]
            hidden[pick] = False

        return hidden

    def _sample_group_curve(self, ratio):
        """Skeleton-correlated smooth curve (window, parts) in [0,1], higher = more
        likely hidden at that frame. Correlated across parts via self._chol,
        smoothed across time via linear interpolation of a few control points."""
        z = torch.randn(self.parts, self.n_control, generator=self.generator)
        correlated = self._chol @ z                      # (parts, n_control), spatial correlation
        correlated = correlated.unsqueeze(0)              # (1, parts, n_control)
        curve = F.interpolate(correlated, size=self.window, mode='linear', align_corners=True)
        curve = curve.squeeze(0).T                         # (window, parts)
        # rank-normalise per frame into [0,1] so the ratio/bounds logic is scale-free
        curve = curve.argsort(dim=1).argsort(dim=1).float() / max(self.parts - 1, 1)
        return curve

    def _clip_ratio(self, ratio):
        return min(max(ratio, self.mask_min), self.mask_max)

    def _temporal(self, epoch=0):
        if self.mask_strategy == 'temporal' and self.warmup_epochs:
            t = min(epoch / max(self.warmup_epochs, 1), 1.0)
            base_ratio = self.mask_min + t * (self.mask_max - self.mask_min)
        else:
            base_ratio = self.mask_min

        masks = []
        for b in range(self.batch):
            use_random = torch.rand(1, generator=self.generator).item() < self.random_frac

            if use_random:
                # plain independent-per-frame random masking for this sample
                frame_masks = []
                for _ in range(self.window):
                    ratio = self._clip_ratio(
                        base_ratio + torch.empty(1).uniform_(-0.1, 0.1, generator=self.generator).item()
                    )
                    probs = torch.full((self.parts,), ratio)
                    hidden = torch.bernoulli(probs, generator=self.generator).bool()
                    hidden = self._enforce_bounds(hidden)
                    frame_masks.append(~hidden)
                masks.append(torch.stack(frame_masks))
                
            else:
                curve = self._sample_group_curve(base_ratio)     # (window, parts)
                window_masks = []
                for t in range(self.window):
                    ratio_t = self._clip_ratio(
                        base_ratio + torch.empty(1).uniform_(-0.05, 0.05, generator=self.generator).item()
                    )
                    threshold = 1.0 - ratio_t
                    hidden = curve[t] >= threshold
                    hidden = self._enforce_bounds(hidden, scores=curve[t])
                    window_masks.append(~hidden)
                masks.append(torch.stack(window_masks))

        return torch.stack(masks)  # (batch, window, parts), True = visible

def exact_mask(K: int, ratio: float, generator=None, device=None):
    n_hide = int(round(ratio * K))
    n_hide = max(0, min(K, n_hide))
    mask = torch.ones((K, 1), dtype=torch.bool)

    if ratio < 0 or ratio > 1 or n_hide == 0:
        return mask

    m = torch.ones(K, dtype=torch.bool, device=device)
    idx = torch.randperm(K, generator=generator, device=device)[:n_hide]
    m[idx] = False  # False = hidden
    mask[:, 0] = m
    return mask


def build_group_correlation(K: int, parent=None, bone_lengths=None, tau: float = 1.5) -> torch.Tensor:
    """
    Build a (K,K) correlation matrix used to make nearby body parts share
    similar occlusion behaviour ("move together").

    parent (list[int], optional): skeleton tree, parent[i] = parent index, -1 = root.
        Uses graph hop-distance between joints. Good default: treats every edge
        equally, robust, no extra data needed beyond the skeleton topology.

    bone_lengths (dict[(i,j)] -> float, optional): if you want distance to
        reflect actual/canonical bone length instead of hop-count (e.g. the
        nose-to-head bone is short but head-to-tailtip spans many long bones),
        pass edge lengths here and they are used as edge weights in the same
        graph-shortest-path computation instead of a flat hop cost.

    tau (float): correlation decay rate. Larger tau -> correlation extends
        further along the skeleton (whole tail moves as one blob). Smaller
        tau -> only immediate neighbours correlate.

    Returns a positive semi-definite (K,K) correlation matrix, diag=1.
    """
    if parent is None:
        return torch.eye(K)

    if bone_lengths is None:
        adj = [[] for _ in range(K)]
        for i, p in enumerate(parent):
            if p >= 0:
                adj[i].append(p)
                adj[p].append(i)

        dist = torch.full((K, K), float('inf'))
        for s in range(K):
            dist[s, s] = 0.0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[s, v] == float('inf'):
                        dist[s, v] = dist[s, u] + 1
                        q.append(v)
    else:
        adj = {i: [] for i in range(K)}
        for (i, j), length in bone_lengths.items():
            adj[i].append((j, length))
            adj[j].append((i, length))
        dist = torch.full((K, K), float('inf'))
        for s in range(K):
            dist[s, s] = 0.0
            pq = [(0.0, s)]
            visited = set()
            import heapq
            heapq.heapify(pq)
            while pq:
                d, u = heapq.heappop(pq)
                if u in visited:
                    continue
                visited.add(u)
                dist[s, u] = d
                for v, w in adj[u]:
                    if v not in visited:
                        heapq.heappush(pq, (d + w, v))

    corr = torch.exp(-dist / tau)
    return corr