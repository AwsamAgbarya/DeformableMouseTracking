"""
Heading estimation, canonicalization, pose de-duplication, yaw augmentation,
and SE(2) body-frame estimation for (partially occluded) keypoint sequences.
"""

import math

import torch
import torch.nn.functional as F


# --- primitives ---------------------------------------------------------------
def wrap(a):
    # a' = atan2(sin a, cos a) ∈ (-π, π]
    return torch.atan2(torch.sin(a), torch.cos(a))


def rot2(theta):
    # R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]      (...,) -> (..., 2, 2)
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _rotate_points(theta, xy):
    # p_n' = R(θ) p_n                                (..., N, 2)
    return torch.einsum("...ij,...nj->...ni", rot2(theta), xy)


def _rotate_vector(theta, v):
    # v' = R(θ) v                                    (..., 2)
    return torch.einsum("...ij,...j->...i", rot2(theta), v)


def _with_xy(x, xy):
    # x' = [xy, z]                                   (z kept if present)
    return torch.cat([xy, x[..., 2:]], dim=-1)


def se2_apply(x, theta, t):
    # xy' = R(θ) xy + t,  z' = z                     (body -> world)
    return _with_xy(x, _rotate_points(theta, x[..., :2]) + t.unsqueeze(-2))


def se2_inverse_apply(x, theta, t):
    # xy' = R(-θ) (xy - t),  z' = z                  (world -> body)
    return _with_xy(x, _rotate_points(-theta, x[..., :2] - t.unsqueeze(-2)))


def se2_rotate_vectors(v, theta):
    # v' = R(θ) v,  z' = z                           (no translation)
    return _with_xy(v, _rotate_points(theta, v[..., :2]))


def se2_compose(theta_a, t_a, theta_b, t_b):
    # (θ, t) = (θ_a + θ_b, t_a + R(θ_a) t_b)         (a ∘ b: b first, then a)
    return wrap(theta_a + theta_b), t_a + _rotate_vector(theta_a, t_b)


def se2_inverse(theta, t):
    # (θ, t)⁻¹ = (-θ, -R(-θ) t)
    return wrap(-theta), -_rotate_vector(-theta, t)


def circular_offset(a, b):
    # δ = atan2(mean sin(a - b), mean cos(a - b))
    d = a - b
    return torch.atan2(torch.sin(d).mean(), torch.cos(d).mean())


def rotate_about(x, theta, pivot_xy):
    # xy' = R(θ) (xy - c) + c,  z' = z               (c = pivot_xy, vertical axis)
    p = pivot_xy.unsqueeze(-2)
    return _with_xy(x, _rotate_points(theta, x[..., :2] - p) + p)


# --- heading from a left/right joint pair -------------------------------------

def smooth_angle(theta, window=5):
    """
    Wrap-safe moving average of a (T,) heading: filters the unit vector, not the angle.

    Hip/shoulder headings oscillate at stride frequency. That is harmless for a
    snapshot model, but a temporal model would see a fake body-frame oscillation
    and a corrupted yaw-rate target, so smooth before canonicalizing a sequence.
    """
    if window <= 1:
        return theta
    lo = window // 2
    v = torch.stack([torch.cos(theta), torch.sin(theta)]).unsqueeze(1)  # (2, 1, T)
    v = F.pad(v, (lo, window - 1 - lo), mode="replicate")               # keeps length T
    v = F.avg_pool1d(v, window, stride=1).squeeze(1)
    return torch.atan2(v[1], v[0])


def forward_facing(data, LR_joint_indices, forward_sign=1):
    """
    Returns the unit forward vector (T, 2) and its norm (T, 1), with 1 where degenerate.

    Forward is perpendicular to (right - left). Whether that points at the nose
    or the tail depends on the coordinate system's handedness, so get
    `forward_sign` from `resolve_forward_sign` rather than guessing.
    """
    li, ri = LR_joint_indices
    data = torch.as_tensor(data, dtype=torch.float32)
    lr = data[:, ri, :2] - data[:, li, :2]
    fwd = torch.stack([-lr[:, 1], lr[:, 0]], dim=1) * forward_sign
    norms = torch.linalg.norm(fwd, dim=1, keepdim=True)
    norms = torch.where(norms < 1e-9, torch.ones_like(norms), norms)
    return fwd / norms, norms


def resolve_forward_sign(data, lr_indices, nose_idx, tail_idx):
    """+1 or -1, whichever makes forward agree on average with (nose - tail)."""
    data = torch.as_tensor(data, dtype=torch.float32)
    fwd, _ = forward_facing(data, lr_indices)
    axis = data[:, nose_idx, :2] - data[:, tail_idx, :2]
    return 1 if (fwd * axis).sum(1).mean() >= 0 else -1, fwd


def estimate_heading(data, nose_idx, tail_idx, lr_indices, smooth_window=1):
    """Heading (T,) from a left/right pair, sign-checked against nose - tail."""
    sign, fwd = resolve_forward_sign(data, lr_indices, nose_idx, tail_idx)
    return smooth_angle(torch.atan2(fwd[:, 1], fwd[:, 0]), smooth_window)


# --- canonicalization ---------------------------------------------------------

def canonicalize(data, theta, origin=None):
    """
    Rotate (T, N, 3) data by -theta (T,) about z so the animal faces +X.
    origin (T, 3), if given, is subtracted first (on all three axes).
    """
    if origin is not None:
        data = data - origin[:, None, :]
    return se2_rotate_vectors(data, -theta)


def find_unique_poses(poses, threshold=0.01, chunk=2048, max_keep=None, verbose=True):
    """
    Greedy de-duplication of (N, J, 3) poses: keep a frame if its per-coordinate
    RMS distance to every already-kept frame exceeds `threshold`.
    Returns the kept frame indices.
    """
    N, J, _ = poses.shape
    log = print if verbose else (lambda *_: None)

    if threshold is None or threshold <= 0:
        log(f"[find_unique_poses] dedup disabled, keeping all {N} frames")
        return torch.arange(N)

    flat = poses.reshape(N, -1)
    thr = float(threshold) * math.sqrt(J * 3)
    kept_idx, kept = [0], flat[:1]

    for start in range(1, N, chunk):
        block = flat[start:start + chunk]
        # Screen the whole block against everything kept before it, so each
        # candidate only needs checking against rows added within this block.
        far = torch.cdist(block, kept).min(dim=1).values > thr
        added = block[:0]
        for j in far.nonzero().flatten().tolist():
            row = block[j:j + 1]
            if len(added) and torch.cdist(row, added).min() <= thr:
                continue
            added = torch.cat([added, row])
            kept_idx.append(start + j)
            if max_keep is not None and len(kept_idx) >= max_keep:
                log(f"[find_unique_poses] hit max_keep={max_keep}")
                return torch.tensor(kept_idx, dtype=torch.long)
        kept = torch.cat([kept, added])

    log(f"[find_unique_poses] kept {len(kept_idx)} / {N} frames "
        f"(per-joint threshold {threshold}, flat threshold {thr:.4f})")
    return torch.tensor(kept_idx, dtype=torch.long)


def procrustes_se2(template_xy, obs_xy, weights, total_weight=None, eps=1e-8):
    """
    Weighted rigid 2D Procrustes: find (theta, t) minimising

    template_xy  : (N, 2) canonical template, any gauge
    obs_xy       : (..., N, 2) observed world xy
    weights      : (..., N) >= 0; zero for occluded joints, else per-joint reliability
    total_weight : sum of the full (unoccluded) weights, used only to normalise
                   conf_t. Defaults to weights.sum(-1).max().

    Returns
        theta  (...,)    body yaw in world frame
        t      (..., 2)  body origin in world frame
        conf_R (...,)    rotation reliability in [0, 1], the optimal similarity
                         scale |M| / sum(w |p|^2). It is 1 for an exact rigid copy
                         and ~0 when the visible set is degenerate (one joint,
                         near-collinear), i.e. when yaw is not identifiable.
        conf_t (...,)    visible fraction of the total weight. Translation stays
                         identifiable from one joint, hence the gentler score.

    A frame with no visible joints gets theta = t = 0 and zero confidence (no NaNs).
    """
    w = weights
    W = w.sum(-1)
    Wc = W.clamp_min(eps).unsqueeze(-1)
    # Centroid of template and observation
    p_bar = (w.unsqueeze(-1) * template_xy).sum(-2) / Wc
    q_bar = (w.unsqueeze(-1) * obs_xy).sum(-2) / Wc
    # Centered template and observation
    p = template_xy - p_bar.unsqueeze(-2)
    q = obs_xy - q_bar.unsqueeze(-2)

    # S parallel
    dot = (w * (p * q).sum(-1)).sum(-1)
    # S cross
    cross = (w * (p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0])).sum(-1)
    spread = (w * (p * p).sum(-1)).sum(-1)

    theta = torch.atan2(cross, dot)
    t = q_bar - _rotate_vector(theta, p_bar)
    conf_R = (torch.hypot(dot, cross) / spread.clamp_min(eps)).clamp(0.0, 1.0)

    if total_weight is None:
        total_weight = W.max()
    conf_t = (W / max(float(total_weight), eps)).clamp(0.0, 1.0)
    return theta, t, conf_R, conf_t

def gauge_fix_template(template_xy, weights, axis_pair=None):
    """
    Pin the template gauge so fitted (theta, t) are comparable across runs:
    weighted centroid at the origin and axis_pair (from_idx -> to_idx, e.g.
    tail_base -> nose) along +X. Then theta = 0 means "nose along world +X"
    and t is the weighted body centroid.
    """
    c = (weights.unsqueeze(-1) * template_xy).sum(0) / weights.sum().clamp_min(1e-8)
    x = template_xy - c
    if axis_pair is not None:
        v = x[int(axis_pair[1])] - x[int(axis_pair[0])]
        x = _rotate_points(-torch.atan2(v[1], v[0]), x)
    return x


def fit_body_template(poses_xy, init_theta=None, init_t=None, axis_pair=None, n_iter=10,
                      weight_iters=3, tol=1e-5, weight_clip=25.0, eps=1e-8, verbose=True, tag=""):
    """
    Alternating fit of (template, per-joint weights, per-frame transforms).

    poses_xy     : (T, N, 2) world xy
    init_theta   : (T,) bootstrap yaw, e.g. from estimate_heading(). Without a
                   bootstrap the first template is a rotational blur of a turning
                   animal and the fit has nothing to lock onto. If None, the
                   axis_pair direction is used.
    init_t       : (T, 2) bootstrap translation, default per-frame centroid.
    axis_pair    : (from_idx, to_idx) defining the template's +X.
    weight_iters : re-estimate per-joint weights for this many iterations, then
                   freeze them. Re-weighting forever (IRLS) has no single
                   objective and jitters at ~0.1 deg; frozen weights leave plain
                   coordinate descent, which converges.
    tol          : stop once mean |d theta| between iterations (radians) < tol.

    Returns dict: template (N, 2), weights (N,) with mean 1, theta (T,), t (T, 2), report.
    """
    x = torch.as_tensor(poses_xy, dtype=torch.float32)
    assert x.dim() == 3 and x.shape[-1] == 2, f"expected (T, N, 2), got {tuple(x.shape)}"
    T, N, _ = x.shape

    if init_theta is not None:
        theta = torch.as_tensor(init_theta, dtype=torch.float32).reshape(-1)
        assert theta.shape[0] == T, f"init_theta has {theta.shape[0]} frames, poses have {T}"
    else:
        assert axis_pair is not None, "fit_body_template needs init_theta or axis_pair to bootstrap"
        v = x[:, int(axis_pair[1])] - x[:, int(axis_pair[0])]
        theta = torch.atan2(v[:, 1], v[:, 0])
    t = x.mean(1) if init_t is None else torch.as_tensor(init_t, dtype=torch.float32)

    w = torch.ones(N)
    history = []
    for it in range(n_iter):
        template = gauge_fix_template(se2_inverse_apply(x, theta, t).mean(0), w, axis_pair)
        prev_theta = theta
        theta, t, _, _ = procrustes_se2(template, x, w.expand(T, N), total_weight=float(w.sum()))

        if it < weight_iters:
            var = ((se2_inverse_apply(x, theta, t) - template) ** 2).sum(-1).mean(0)
            # Stable body-frame joints (spine) should outweigh flailing ones (tail
            # tip), but clip so the fit can't collapse onto a two-joint estimator.
            w = 1.0 / (var + eps)
            w = (w / w.mean()).clamp(1.0 / weight_clip, weight_clip)
            w = w / w.mean()

        history.append(float(wrap(theta - prev_theta).abs().mean()))
        if it >= weight_iters and history[-1] < tol:
            break

    resid = (se2_inverse_apply(x, theta, t) - template).norm(dim=-1)  # (T, N)
    report = {
        "iterations": len(history),
        "delta_theta_deg": [math.degrees(d) for d in history],
        "rms_residual": float(resid.pow(2).mean().sqrt()),
        "per_joint_rms": resid.pow(2).mean(0).sqrt().tolist(),
        "weights": w.tolist(),
    }

    if verbose:
        wl, order = w.tolist(), torch.argsort(w, descending=True).tolist()
        fmt = lambda idx: ", ".join(f"{i}:{wl[i]:.2f}" for i in idx)
        p = f"[fit_body_template{tag}]"
        print(f"{p} {len(history)} iters, last |dtheta| = {report['delta_theta_deg'][-1]:.2e} deg\n"
              f"{p} RMS body-frame residual = {report['rms_residual']:.3f} (input units)\n"
              f"{p} heaviest joints {fmt(order[:5])} | lightest {fmt(order[-5:])}")

    return {"template": template, "weights": w, "theta": theta, "t": t, "report": report}

def combine_window_frames(theta, t, offsets, tau_yaw, tau_xy=None,
                          conf_R=None, conf_t=None, valid=None, eps=1e-8):
    """
    Collapse a window's per-frame fits into a single SE(2) frame.

    theta          : (..., W)
    t              : (..., W, 2)
    offsets        : (..., W) signed frame distance from the window centre
    tau_yaw/tau_xy : decay lengths in frames (tau_xy defaults to tau_yaw)
    conf_R, conf_t : (..., W) optional confidences from procrustes_se2
    valid          : (..., W) bool, False for padded slots

    Returns theta_w (...,), t_w (..., 2), ok (...,) bool
    """
    tau_xy = tau_yaw if tau_xy is None else tau_xy
    a = offsets.abs().float()
    v = torch.ones_like(a) if valid is None else valid.to(a.dtype)
    a0 = a.masked_fill(v <= 0, float("inf")).min(-1, keepdim=True).values
    a0 = torch.where(torch.isfinite(a0), a0, torch.zeros_like(a0))  # no valid slots -> 0
    d = (a - a0).clamp_min(0.0)

    wR = torch.exp(-d / max(float(tau_yaw), eps)) * v
    wT = torch.exp(-d / max(float(tau_xy), eps)) * v
    if conf_R is not None:
        wR = wR * conf_R
    if conf_t is not None:
        wT = wT * conf_t

    theta_w = torch.atan2((wR * torch.sin(theta)).sum(-1), (wR * torch.cos(theta)).sum(-1))
    sT = wT.sum(-1)
    t_w = (wT.unsqueeze(-1) * t).sum(-2) / sT.clamp_min(eps).unsqueeze(-1)
    return theta_w, t_w, (wR.sum(-1) > eps) & (sT > eps)