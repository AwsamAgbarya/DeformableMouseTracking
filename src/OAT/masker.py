import math
import torch


def mean_joint_distance(poses, max_frames=5000, seed=0):
    """(K, K) mean pairwise Euclidean joint distance over up to `max_frames` frames."""
    P = torch.as_tensor(poses, dtype=torch.float32)
    if P.shape[0] > max_frames:
        g = torch.Generator().manual_seed(int(seed))
        P = P[torch.randperm(P.shape[0], generator=g)[:max_frames]]
    return torch.cdist(P, P).mean(0)


class RecordingOcclusion:
    """
    Occlusion sampled on the RECORDING CLOCK.
    One occlusion field over the whole recording, resampled once per epoch,

    Modes (config["mode"])
    ----------------------
    "world" (default) world-fixed blobs + bursty frame drops.
    "iid"             Baseline for comparison.
    "world" mode (default)
    ----------------------
    world occluders   Blobs (balls) that sit still in CAGE coordinates.
    frame drops       Bursty whole-frame losses 
    Target ratio (curriculum off) sets the AVERAGE hidden fraction on usable frames
    Curriculum bisects one intensity multiplier `alpha`, which scalesthe occluder rate by alpha ** alpha_rate
    """

    def __init__(self, n_frames, n_joints, fps=30, config=None, joint_distance=None, poses=None, seed=42):
        cfg = config or {}
        self.config = cfg
        self.T = int(n_frames)
        self.K = int(n_joints)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.seed = seed

        if poses is None:
            raise ValueError(
                "RecordingOcclusion needs the WORLD joint positions (T, K, 3) in mm: the "
                "occluders are blobs in cage coordinates. Pass poses=ds.deform_global."
            )
        X = torch.as_tensor(poses, dtype=torch.float32).detach().cpu()
        assert X.shape == (self.T, self.K, 3), f"poses must be ({self.T}, {self.K}, 3), got {tuple(X.shape)}"
        self.poses = X.contiguous()

        # body speed in mm/s: centroid of all joints, smoothed over ~0.3 s
        c = X[..., :2].mean(1)
        v = torch.zeros(self.T)
        if self.T > 1:
            v[1:] = (c[1:] - c[:-1]).norm(dim=-1) * self.fps
            v[0] = v[1]
        k = 2 * int(round(0.15 * self.fps)) + 1
        if self.T >= k:
            v = torch.nn.functional.avg_pool1d(v.view(1, 1, -1), k, stride=1, padding=k // 2,
                                               count_include_pad=False).view(-1)
        self.speed = v
        # world occluders
        self.occluder_rate_hz = float(cfg.get("occluder_rate_hz", 2.5))
        self.occluder_life_s = float(cfg.get("occluder_life_s", 0.8))        # median
        self.occluder_life_sigma = float(cfg.get("occluder_life_sigma", 0.7))
        self.occluder_radius_sigma = float(cfg.get("occluder_radius_sigma", 0.3))
        # centre offset from the anchor joint, as a fraction of the blob radius
        self.occluder_jitter = float(cfg.get("occluder_jitter", 0.5))
        # blobs are placed along the path in proportion to (speed + this floor), mm/s.
        self.occluder_speed_floor = float(cfg.get("occluder_speed_floor_mm_s", 20.0))

        # whole-frame drops
        self.frame_drop_rate_hz = float(cfg.get("frame_drop_rate_hz", 0.02))
        self.frame_drop_len_s = float(cfg.get("frame_drop_len_s", 0.20))  # median
        self.frame_drop_len_sigma = float(cfg.get("frame_drop_len_sigma", 0.8))

        # A frame with too few visible joints cannot support a body-frame fit. Promote it to a frame drop instead of pretending it is usable context.
        self.min_visible_joints = int(cfg.get("min_visible_joints", 3))

        # curriculum
        cur = cfg.get("curriculum", {})
        self.curriculum_enabled = bool(cur.get("enabled", False))
        self.ratio_min = float(cur.get("ratio_min", 0.25))
        self.ratio_max = float(cur.get("ratio_max", 0.60))
        self.warmup_epochs = int(cur.get("warmup_epochs", 50))
        self.alpha = float(cfg.get("alpha", 1.0))
        # fixed hidden ratio without a curriculum: alpha is calibrated to it once
        tr = cfg.get("target_ratio", None)
        self.target_ratio = None if tr is None else float(tr)
        # occluder rate x alpha ** alpha_rate, blob radius x alpha ** alpha_radius
        self.alpha_rate = float(cfg.get("alpha_rate", 1.0))
        self.alpha_radius = float(cfg.get("alpha_radius", 0.0))
        assert self.alpha_rate >= 0 and self.alpha_radius >= 0 and self.alpha_rate + self.alpha_radius > 0, \
            "alpha_rate and alpha_radius must be >= 0 and not both 0"
        self._alpha_cache = {}
        self._calibration_frames = int(cfg.get("calibration_frames", 20000))

        # occlusion mode: "world" (structured blobs) or "iid" (independent per joint)
        self.mode = str(cfg.get("mode", "world")).lower()
        assert self.mode in ("world", "iid"), f"mode must be 'world' or 'iid', got {self.mode!r}"
        # iid: per-(frame, joint) hide probability, used as given when there is no target
        self.iid_p = float(cfg.get("iid_p", 0.5))
        assert 0.0 <= self.iid_p < 1.0, f"iid_p must be in [0, 1), got {self.iid_p}"
        self._iid_cache = {}

        # geometry: only sets the default blob size and the structure report
        if joint_distance is None:
            joint_distance = mean_joint_distance(X, max_frames=int(cfg.get("max_corr_frames", 5000)), seed=self.seed)
        D = torch.as_tensor(joint_distance, dtype=torch.float32)
        assert D.shape == (self.K, self.K), f"joint_distance must be ({self.K}, {self.K}), got {tuple(D.shape)}"
        self.joint_distance = D
        med = float(D[~torch.eye(self.K, dtype=torch.bool)].median())
        _r = cfg.get("occluder_radius", None)       # median blob radius in mm at alpha = 1
        if _r is None:
            self.occluder_radius = float(cfg.get("occluder_radius_scale", 0.5)) * med
            print(
                f"[RecordingOcclusion] occluder_radius auto-set to {self.occluder_radius:.1f} mm "
                f"({float(cfg.get('occluder_radius_scale', 0.5)):g} x median pairwise joint "
                f"distance {med:.1f} mm)."
            )
        else:
            self.occluder_radius = float(_r)

        self.epoch = -1
        self.visible = None
        self.frame_dropped = None
        self.set_epoch(0)

    # sampling

    @staticmethod
    def _lognormal_frames(median_s, sigma, fps, shape, g):
        """Log-normal durations in frames, >= 1. `median_s` is the median."""
        z = torch.randn(shape, generator=g)
        secs = float(median_s) * torch.exp(float(sigma) * z)
        return (secs * fps).round().long().clamp_min(1)

    @staticmethod
    def _poisson_quantile(u, lam):
        """
        Inverse Poisson CDF at u. For a fixed u it never decreases as lam grows,
        so with a fixed seed a larger alpha only ever ADDS occluders.
        """
        if lam <= 0:
            return 0
        u = min(max(float(u), 1e-12), 1.0 - 1e-12)
        if lam > 400:   # normal approximation; monotone in lam
            z = float(torch.special.ndtri(torch.tensor(u, dtype=torch.float64)))
            return max(0, int(math.floor(lam + math.sqrt(lam) * z + 0.5)))
        k, p = 0, math.exp(-lam)
        c = p
        while c < u:
            k += 1
            p *= lam / k
            c += p
        return k

    def _anchor_cdf(self, T):
        """CDF over frames [0, T) of where blobs are placed: proportional to
        speed + occluder_speed_floor_mm_s (float64, last entry 1)."""
        w = self.speed[:T].double() + self.occluder_speed_floor
        if float(w.sum()) <= 0:
            w = torch.ones(T, dtype=torch.float64)
        cdf = torch.cumsum(w, 0)
        return cdf / cdf[-1]

    def sample_occluders(self, T, g, alpha):
        """
        The world blobs of one field, as tensors of length n:
            birth, end (frames, end exclusive), joint, anchor (frame),
            centre (n, 3) in cage coordinates (mm), radius (n,) in mm.
        """
        assert T <= self.T, f"cannot sample {T} frames from a {self.T}-frame recording"
        lam = self.occluder_rate_hz * T * self.dt * (max(alpha, 1e-9) ** self.alpha_rate)
        n = self._poisson_quantile(torch.rand(1, generator=g).item(), lam)

        chunk = 256
        parts = {k: [] for k in ("u", "life", "joint", "phase", "zr", "jit")}
        for _ in range(-(-n // chunk)):
            parts["u"].append(torch.rand(chunk, generator=g))
            parts["life"].append(self._lognormal_frames(
                self.occluder_life_s, self.occluder_life_sigma, self.fps, (chunk,), g))
            parts["joint"].append(torch.randint(0, self.K, (chunk,), generator=g))
            parts["phase"].append(torch.rand(chunk, generator=g))
            parts["zr"].append(torch.randn(chunk, generator=g))
            parts["jit"].append(torch.randn(chunk, 3, generator=g))
        if n == 0:
            e = torch.zeros(0, dtype=torch.long)
            return {"birth": e, "end": e, "joint": e, "anchor": e,
                    "centre": torch.zeros(0, 3), "radius": torch.zeros(0)}
        a = {k: torch.cat(v)[:n] for k, v in parts.items()}

        cdf = self._anchor_cdf(T)
        anchor = torch.searchsorted(cdf, a["u"].double()).clamp_max(T - 1)
        birth = (anchor - (a["phase"] * a["life"].float()).long()).clamp_min(0)
        end = (birth + a["life"]).clamp_max(T)
        r_base = self.occluder_radius * torch.exp(self.occluder_radius_sigma * a["zr"])
        centre = self.poses[anchor, a["joint"]] + self.occluder_jitter * r_base.unsqueeze(-1) * a["jit"]
        radius = r_base * (max(alpha, 1e-9) ** self.alpha_radius)
        return {"birth": birth, "end": end, "joint": a["joint"], "anchor": anchor,
                "centre": centre, "radius": radius}

    def _world_occluders(self, T, g, alpha):
        """(T, K) bool: joint inside at least one live blob."""
        ev = self.sample_occluders(T, g, alpha)
        hits = torch.zeros(T, self.K, dtype=torch.int32)
        for s in range(0, int(ev["birth"].numel()), 2048):           # chunks bound the memory
            b, e = ev["birth"][s:s + 2048], ev["end"][s:s + 2048]
            life = e - b
            blob = torch.repeat_interleave(torch.arange(life.numel()), life)          # (M,)
            frame = b[blob] + torch.arange(blob.numel()) - (life.cumsum(0) - life)[blob]
            d2 = (self.poses[frame] - ev["centre"][s:s + 2048][blob, None]).pow(2).sum(-1)
            inside = d2 < ev["radius"][s:s + 2048][blob, None] ** 2               # (M, K)
            hits.index_add_(0, frame, inside.int())
        return hits > 0

    def _frame_drops(self, T, g):
        drop = torch.zeros(T, dtype=torch.bool)
        duration_s = T * self.dt
        lam = self.frame_drop_rate_hz * duration_s
        if lam <= 0:
            return drop
        n = int(torch.poisson(torch.tensor([lam]), generator=g).item())
        if n == 0:
            return drop
        births = torch.randint(0, T, (n,), generator=g)
        lens = self._lognormal_frames(
            self.frame_drop_len_s, self.frame_drop_len_sigma, self.fps, (n,), g
        )
        for i in range(n):
            b = int(births[i])
            drop[b:min(b + int(lens[i]), T)] = True
        return drop

    def _iid_hidden(self, T, g, p):
        """(T, K) bool: each (frame, joint) hidden independently with probability p.
        One uniform per entry, so with a fixed seed a larger p only adds hidden joints."""
        return torch.rand(T, self.K, generator=g) < float(p)

    def generate(self, T=None, seed=None, alpha=None, with_frame_drops=True, p=None):
        T = int(self.T if T is None else T)
        seed = int(self.seed if seed is None else seed)

        g = torch.Generator().manual_seed(seed)
        if self.mode == "iid":
            hidden = self._iid_hidden(T, g, self.iid_p if p is None else p)
        else:
            hidden = self._world_occluders(T, g, float(self.alpha if alpha is None else alpha))
        dropped = (self._frame_drops(T, torch.Generator().manual_seed(seed + 15485863))
                   if with_frame_drops else torch.zeros(T, dtype=torch.bool))

        visible = ~hidden
        visible[dropped] = False

        too_sparse = visible.sum(1) < self.min_visible_joints
        dropped = dropped | too_sparse
        visible[dropped] = False
        return visible, dropped

    # curriculum

    def ratio_at(self, epoch):
        """Target hidden ratio of this epoch: the curriculum if enabled, else
        `target_ratio` if set, else None (alpha used as given)."""
        if not self.curriculum_enabled:
            return self.target_ratio
        f = min(epoch / max(self.warmup_epochs, 1), 1.0)
        return self.ratio_min + f * (self.ratio_max - self.ratio_min)

    def _hidden_fraction(self, alpha, seed):
        """
        Hidden fraction of joint observations on USABLE frames.
        """
        T = min(self.T, self._calibration_frames)
        vis, dropped = self.generate(T=T, seed=seed, alpha=alpha, with_frame_drops=False)
        keep = ~dropped
        if float(keep.float().mean()) < 0.5:
            return 1.0
        return float((~vis[keep]).float().mean())

    def _iid_usable_frac(self, p):
        """P(a frame survives min_visible_joints) under iid masking with hide prob p."""
        K, m = self.K, max(self.min_visible_joints, 0)
        q = 1.0 - float(p)
        return sum(math.comb(K, v) * q ** v * float(p) ** (K - v) for v in range(m, K + 1))

    def _iid_hidden_fraction(self, p):
        """
        EXPECTED hidden fraction on USABLE frames under iid masking, exactly.
        """
        K, m = self.K, max(self.min_visible_joints, 0)
        p = float(p)
        q = 1.0 - p
        pmf = [math.comb(K, v) * q ** v * p ** (K - v) for v in range(K + 1)]
        usable = sum(pmf[m:])
        if usable <= 1e-12:
            return max(K - m, 0) / K
        ev = sum(v * pmf[v] for v in range(m, K + 1)) / usable
        return 1.0 - ev / K

    def _calibrate_iid(self, target_ratio, tol=0.005, verbose=False):
        """
        The hide probability p whose expected hidden fraction on usable frames
        """
        key = round(float(target_ratio), 3)
        if key in self._iid_cache:
            self.iid_p = self._iid_cache[key]
            return self.iid_p

        lo, hi = 0.0, 1.0 - 1e-9
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self._iid_hidden_fraction(mid) < target_ratio:
                lo = mid
            else:
                hi = mid
        p = 0.5 * (lo + hi)
        r = self._iid_hidden_fraction(p)

        if abs(r - target_ratio) >= tol:
            ceiling = max(self.K - self.min_visible_joints, 0) / self.K
            print(
                f"[RecordingOcclusion] WARNING: iid masking cannot reach hidden ratio "
                f"{target_ratio:.3f}; the most it gives on usable frames is {ceiling:.3f} "
                f"(K={self.K}, min_visible_joints={self.min_visible_joints}). Using p={p:.4f} "
                f"-> {r:.3f}."
            )
        self._iid_cache[key] = p
        self.iid_p = p
        if verbose:
            print(
                f"[RecordingOcclusion] iid: calibrated p={p:.4f} for target hidden ratio "
                f"{target_ratio:.3f} (expected {r:.3f}; "
                f"{1.0 - self._iid_usable_frac(p):.2%} of frames promoted to drops)"
            )
        return p

    def calibrate(self, target_ratio, tol=0.005, max_iter=30, seed=None, verbose=False):
        if self.mode == "iid":
            return self._calibrate_iid(target_ratio, tol=tol, verbose=verbose)

        key = round(float(target_ratio), 3)
        if key in self._alpha_cache:
            self.alpha = self._alpha_cache[key]
            return self.alpha

        probe = int(self.seed + 104729 if seed is None else seed)
        lo, hi = 1e-3, 1e3
        best, best_err = 1.0, float("inf")
        for _ in range(int(max_iter)):
            mid = math.sqrt(lo * hi)
            r = self._hidden_fraction(mid, probe)
            if abs(r - target_ratio) < best_err:
                best, best_err = mid, abs(r - target_ratio)
            if best_err < tol:
                break
            if r < target_ratio:
                lo = mid
            else:
                hi = mid

        if best_err >= tol:
            print(
                f"[RecordingOcclusion] WARNING: calibration reached hidden ratio within "
                f"{best_err:.3f} of {target_ratio:.3f}, not {tol}. Frames promoted to drops cap "
                "the usable-frame ratio; lower alpha_radius or the curriculum ceiling."
            )
        self._alpha_cache[key] = best
        self.alpha = best
        if verbose:
            print(
                f"[RecordingOcclusion] calibrated alpha={best:.4f} for "
                f"target hidden ratio {target_ratio:.3f} "
                f"(realized {self._hidden_fraction(best, probe):.3f})"
            )
        return best

    def set_epoch(self, epoch):
        """Resample the whole recording. Call once per epoch, before the loader."""
        self.epoch = int(epoch)
        target = self.ratio_at(self.epoch)
        if target is not None:
            self.calibrate(target)
        self.visible, self.frame_dropped = self.generate(
            seed=self.seed + 7919 * self.epoch
        )
        return self

    # report

    def _hidden_runs_s(self, hidden_1d):
        """Hidden-run durations in seconds for one joint."""
        h = hidden_1d.long()
        pad = torch.zeros(1, dtype=torch.long)
        d = torch.diff(torch.cat([pad, h, pad]))
        starts = (d == 1).nonzero().flatten()
        ends = (d == -1).nonzero().flatten()
        return (ends - starts).float() / self.fps
