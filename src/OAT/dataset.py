import torch
from torch.utils.data import Dataset

class SnapshotWrapper(Dataset):
    def __init__(self, deformable_coords):
        self.deformable_coords = deformable_coords

    def __getitem__(self, idx):
        return self.deformable_coords[idx]
    
    def __len__(self):
        return self.deformable_coords.shape[0]

class TemporalWrapper(Dataset):
    def __init__(self, data_d, relative_dist, outlier_flags,
                 T_half=8, max_radius=32,
                 dropout_min=0.10, dropout_max=0.20,
                 clean_frac=0.15,
                 pattern_weights=None,
                 seed=42):
        """
        data_d         : (T, N, 3) deformable coords (global + local)
        relative_dist  : (T, N, 3) auxiliary target (deform - rigid)
        outlier_flags  : (T,) bool, True = unusable frame (discontinuity)
        T_half         : number of PAST and FUTURE context frames each
        max_radius K   : hard cap on how far a search can look for a valid neighbor
        clean_frac     : fraction of __getitem__ calls that get NO extra dropout
                         (uniform spacing, only outliers excluded)
        """
        assert data_d.shape[0] == relative_dist.shape[0] == outlier_flags.shape[0]
        self.data_d = data_d
        self.relative_dist = relative_dist
        self.outlier_flags = outlier_flags.bool()
        self.T = data_d.shape[0]
        self.T_half = T_half
        self.K = max_radius
        self.dropout_min = dropout_min
        self.dropout_max = dropout_max
        self.clean_frac = clean_frac
        self.pattern_weights = pattern_weights or {"isolated": 0.35, "burst": 0.30, "center": 0.35}
        self.generator = torch.Generator().manual_seed(seed)

        self.new_epoch()  # builds first epoch's mask + valid_centers

    def __len__(self):
        return len(self.valid_centers)

    def new_epoch(self):
        """Call once at the start of every training epoch."""
        # outliers + synthetic drops
        self.epoch_dropout = self._generate_global_dropout()
        self.epoch_unusable_context = self.epoch_dropout
        self.valid_centers = self._build_valid_centers(
            context_mask=self.epoch_unusable_context,
            center_eligibility_mask=self.outlier_flags
        )

    def _build_valid_centers(self, context_mask, center_eligibility_mask):
        centers = []
        for i in range(self.T):
            if center_eligibility_mask[i]:
                continue  # true outlier: never usable as ground-truth target
            lo, hi = max(0, i - self.K), min(self.T, i + self.K + 1)
            past_ok = (~context_mask[lo:i]).sum() >= self.T_half
            fut_ok = (~context_mask[i+1:hi]).sum() >= self.T_half
            if past_ok and fut_ok:
                centers.append(i)
        return torch.tensor(centers, dtype=torch.long)
            
    def _generate_global_dropout(self):
        """
        Runs once per epoch. Returns a (T,) bool tensor, True = unusable
        (real outlier OR synthetic dropout), built using the SAME pattern
        logic (isolated / burst / center) as generate_dropout_mask, just
        applied across the whole sequence instead of a local window.
        """
        combined = self.outlier_flags.clone()

        if torch.rand(1, generator=self.generator).item() < self.clean_frac:
            return combined  # clean epoch: only real outliers are unusable

        ratio = torch.empty(1).uniform_(
            self.dropout_min, self.dropout_max, generator=self.generator).item()
        n_drop = max(1, int(round(ratio * self.T)))

        pattern_idx = torch.multinomial(
            torch.tensor(list(self.pattern_weights.values())), 1, generator=self.generator
        ).item()
        pattern = list(self.pattern_weights.keys())[pattern_idx]

        drop_mask = torch.zeros(self.T, dtype=torch.bool)

        if pattern == "isolated":
            drop_mask = self._drop_isolated(combined, n_drop)

        elif pattern == "burst":
            drop_mask = self._drop_burst(combined, n_drop)

        elif pattern == "center":
            drop_mask = self._drop_center(combined, n_drop)

        return combined | drop_mask


    def _drop_isolated(self, base, n_drop):
        """Scatter n_drop single-frame drops uniformly at random over non-outlier frames."""
        drop_mask = torch.zeros(self.T, dtype=torch.bool)
        candidates = torch.nonzero(~base, as_tuple=True)[0]
        if len(candidates) > 0:
            n_drop = min(n_drop, len(candidates))
            perm = torch.randperm(len(candidates), generator=self.generator)[:n_drop]
            drop_mask[candidates[perm]] = True
        return drop_mask


    def _drop_burst(self, base, n_drop):
        """Scatter multiple bursts of length 2-6 across the sequence until n_drop frames used."""
        drop_mask = torch.zeros(self.T, dtype=torch.bool)
        remaining = n_drop
        max_tries = max(20, n_drop)  # scale tries with target size so large ratios still converge
        tries = 0

        while remaining > 0 and tries < max_tries:
            burst_len = min(remaining, torch.randint(2, 6, (1,), generator=self.generator).item())
            if self.T - burst_len <= 0:
                break
            start = torch.randint(0, self.T - burst_len, (1,), generator=self.generator).item()
            span = slice(start, start + burst_len)

            if not base[span].any() and not drop_mask[span].any():
                drop_mask[span] = True
                remaining -= burst_len
            tries += 1

        return drop_mask


    def _drop_center(self, base, n_drop, n_anchors=None):
        """
        Global analogue of the local 'center' pattern: since there's no single
        center in a full-sequence mask, scatter several random anchor points
        across the sequence and concentrate drops around each anchor with the
        same exponential decay used locally. This mimics 'many windows will
        have drops concentrated near their center' without knowing window
        boundaries in advance.
        """
        drop_mask = torch.zeros(self.T, dtype=torch.bool)
        if n_anchors is None:
            n_anchors = max(1, n_drop // (2 * self.K + 1))  # roughly one anchor per window-sized chunk

        anchor_candidates = torch.nonzero(~base, as_tuple=True)[0]
        if len(anchor_candidates) == 0:
            return drop_mask

        n_anchors = min(n_anchors, len(anchor_candidates))
        anchors = anchor_candidates[
            torch.randperm(len(anchor_candidates), generator=self.generator)[:n_anchors]
        ]

        weights = torch.zeros(self.T)
        idx_range = torch.arange(self.T).float()
        for a in anchors:
            dist = torch.abs(idx_range - a.item())
            weights += torch.exp(-dist / (self.K / 3 + 1e-6))

        weights[base] = 0.0
        weights[anchors] = 0.0  # don't drop the anchors themselves, only frames around them

        if weights.sum() > 0:
            n_drop = min(n_drop, (weights > 0).sum().item())
            idx = torch.multinomial(weights, n_drop, replacement=False, generator=self.generator)
            drop_mask[idx] = True

        return drop_mask
    
    def _gather_side(self, center, lo, direction):
        """direction: -1 for past, +1 for future. Returns (indices, offsets, pad_mask)."""
        idxs, offs = [], []
        step = 1
        while len(idxs) < self.T_half and step <= self.K:
            cand = center + direction * step
            if 0 <= cand < self.T and not self.epoch_unusable_context[cand]:
                idxs.append(cand)
                offs.append(direction * step)
            step += 1

        pad_mask = torch.zeros(self.T_half, dtype=torch.bool)
        if len(idxs) < self.T_half:
            n_missing = self.T_half - len(idxs)
            pad_mask[len(idxs):] = True
            fill_idx = idxs[-1] if idxs else center
            fill_off = offs[-1] if offs else 0
            idxs += [fill_idx] * n_missing
            offs += [fill_off] * n_missing

        if direction == -1:
            idxs, offs, pad_mask = idxs[::-1], offs[::-1], pad_mask.flip(0)
        return torch.tensor(idxs), torch.tensor(offs, dtype=torch.float32), pad_mask
    
    def __getitem__(self, idx):
        center = self.valid_centers[idx].item()
        lo = max(0, center - self.K)
        past_idx, past_off, past_pad = self._gather_side(center, lo, -1)
        fut_idx, fut_off, fut_pad = self._gather_side(center, lo, +1)

        window_idx = torch.cat([past_idx, torch.tensor([center]), fut_idx])
        window_offsets = torch.cat([past_off, torch.tensor([0.0]), fut_off])
        window_pad_mask = torch.cat([past_pad, torch.tensor([False]), fut_pad])

        window_data = self.data_d[window_idx]          # (T+1, N, 3)
        target_deform = self.data_d[center].unsqueeze(0)     # (1, N, 3)
        target_rel = self.relative_dist[center].unsqueeze(0)  # (1, N, 3)

        theta = torch.empty(1).uniform_(0, 2 * torch.pi, generator=self.generator).item()
        c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

        window_data = window_data @ R.T
        target_deform = target_deform @ R.T
        target_rel = target_rel @ R.T

        return {
            "window": window_data,              # (T+1, N, 3)
            "offsets": window_offsets,          # (T+1,)
            # "pad_mask": window_pad_mask,         # (T+1,) True = padded/fake slot
            "target_deform": target_deform,      # (1, N, 3)
            "target_rel": target_rel,            # (1, N, 3)
        }