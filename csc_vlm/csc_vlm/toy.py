"""
csc_vlm.toy  --  X1 toy-model arm (the kill-switch, pure numpy/CPU)
===================================================================

Why this file exists
--------------------
The p2 grid runs on real VLMs produced numbers we could not trust, because on a
real model there is NO ground truth: we cannot check whether the measured escape
fraction Gamma_r or the obfuscation cost OC equal their true values. The proof
plan says the FIRST thing to build (Phase 5.1 / "if you do only five things" #1)
is an Elhage-style toy with KNOWN geometry, "~before any VLM GPU-hour", where the
coherence, F_eff and the analytic minimal perturbation are all computable in
closed form. That is the only place an estimator can be validated.

This module builds that toy and uses it to do two things at once:

  (A) REPRODUCE the p2 failure in a controlled setting and prove it is a
      measurement artifact (not a property of the model):
        - the Gamma_r "noise floor" at small n  (exp_noise_floor)
        - escape ~ flat under merge / Gamma mirrors ||a||  (exp_frontier, old metric)
        - prune S blow-up from leverage collapse        (exp_frontier, old metric)

  (B) VALIDATE the replacement metric the plan prescribes (Phase 2.2 / 2.3):
        - per-example Jacobian Gamma_r with monitor != behaviour
        - per-example OC with the decomposition  log OC = log m - log||g_b|| - log Gamma_r
        - OC matches the closed-form minimal perturbation (attack is non-masked)
        - positive control: no cheap evasion in the non-superposed regime (Stevinson)
        - a mini gradient-masking check through the discrete prune (the Phase-3.3 / G2 risk)

Everything is linear + closed-form, so it runs on a laptop in seconds and is
fully deterministic. No torch, no GPU, no cluster. v0.4 is untouched; this is
new p2-only code.

Notation follows the plan: F latent features, D read-space dimensions (carriers),
keep-fraction kappa, superposition pressure rho = F / d_eff, evasion gain
Gamma_r, obfuscation cost OC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

__all__ = [
    "ToyConfig", "make_geometry", "sample_reads", "reads_packed",
    "behaviour_direction", "behaviour_labels", "compress_prune",
    "coherence", "welch_floor", "d_eff", "f_eff_decodable",
    "fit_monitor_p2", "measured_geometry_p2",
    "fit_monitor_detect", "jacobian_metric", "analytic_min_perturbation",
    "verify_oc_against_attack", "masking_index_through_prune",
    "spearman", "fit_loglog_slope", "DEFAULT_KEEPS",
]

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 1. Geometry + data (known ground truth)
# --------------------------------------------------------------------------- #
@dataclass
class ToyConfig:
    F: int = 120            # number of latent features
    D: int = 64             # read-space dimension at full budget (carriers)
    sparsity: float = 0.08  # P(a feature is active) per example
    n: int = 256            # examples per measurement
    monitor_rank: int = 8   # r in the rank-r monitor subspace U_r
    beta: float = 1.0       # OC scale constant
    behaviour_feat: int = 0 # which feature defines the behaviour (cat vs dog)
    n_sinks: int = 3        # high-norm, low-info "attention-sink" coordinates
    sink_scale: float = 4.0 # how much the sinks dominate the read magnitude
    seed: int = 0


def _sink_axes(cfg: "ToyConfig") -> np.ndarray:
    """Deterministic set of read coordinates that act as attention sinks (same for
    every call at a given config, so the frozen monitor and paired design hold)."""
    if cfg.n_sinks <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(cfg.seed + 9999)
    return rng.choice(cfg.D, size=min(cfg.n_sinks, cfg.D), replace=False)


def make_geometry(F: int, D: int, seed: int = 0) -> np.ndarray:
    """Carrier matrix U in R^{D x F}, unit-norm columns (each feature's direction
    in read space). F > D  =>  features share dimensions  =>  superposition.
    This IS the known true geometry: coherence and the Welch floor below read
    off it exactly."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((D, F))
    U /= (np.linalg.norm(U, axis=0, keepdims=True) + EPS)
    return U


def sample_reads(U: np.ndarray, cfg: ToyConfig, seed: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw n sparse feature vectors z and form the full read h0 = U z.
    Returns H0 [n, D], Z [n, F], and the behaviour logit b = h0 . w_b (no
    threshold yet). Behaviour feature contributes z_bf PLUS interference from
    every co-active feature it is non-orthogonal to -- exactly the superposition
    vulnerability Stevinson attacks."""
    rng = np.random.default_rng(seed)
    F = U.shape[1]
    active = rng.random((cfg.n, F)) < cfg.sparsity
    mag = rng.random((cfg.n, F))
    Z = active * mag
    # guarantee a usable class balance on the behaviour feature
    flip = rng.random(cfg.n) < 0.5
    Z[flip, cfg.behaviour_feat] = 0.3 + 0.7 * rng.random(flip.sum())
    H0 = Z @ U.T
    # add attention-sink carriers: a few coordinates with large, NEARLY CONSTANT,
    # class-irrelevant magnitude (sinks look the same on every input). They
    # dominate the L2 norm, so magnitude-pruning keeps them on every example and
    # the retained read collapses onto one near-constant sink vector -> the read
    # cloud loses rank, leverage ||a|| and escape go to ~0, and the p2 S = 1/escape
    # blows up. This is the VLM prune pathology (d_eff/leverage collapse, rival R4)
    # reproduced with known ground truth. Merge (reads_packed) never touches this.
    axes = _sink_axes(cfg)
    if axes.size:
        scale = cfg.sink_scale * (H0.std() + EPS)
        base = np.random.default_rng(cfg.seed + 55555).standard_normal(axes.size)
        H0[:, axes] = scale * base[None, :] + 0.08 * scale * rng.standard_normal((cfg.n, axes.size))
    return H0, Z, Z[:, cfg.behaviour_feat]


def reads_packed(U: np.ndarray, cfg: ToyConfig, keep: float, seed: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """The MERGE / information-preserving superposition sweep.

    Token compression keeps the hidden dimension D fixed but forces more latent
    features to share the read as fewer tokens carry the same F features. We model
    that by raising the number of CO-ACTIVE features per read as keep falls
    (effective sparsity s ~ sparsity/keep), with D and the geometry U fixed. So
    d_eff stays high (>> monitor rank), the class information survives (margin ~
    flat), but interference rises -> the behaviour direction leaks further outside
    a fixed-rank monitor -> Gamma_r rises and OC falls. This is the plan's
    mechanism (N down => rho up => Gamma up => OC down) at matched margin.

    No sinks here: merge is information-preserving, so it must NOT collapse."""
    s = float(min(0.9, cfg.sparsity / max(keep, 1e-3)))
    rng = np.random.default_rng(seed)
    F = U.shape[1]
    active = rng.random((cfg.n, F)) < s
    Z = active * rng.random((cfg.n, F))
    flip = rng.random(cfg.n) < 0.5
    Z[flip, cfg.behaviour_feat] = 0.3 + 0.7 * rng.random(flip.sum())
    return Z @ U.T, Z


def behaviour_direction(U: np.ndarray, cfg: ToyConfig) -> np.ndarray:
    """w_b: the causal read-space direction of the behaviour feature (unit norm).
    Moving along w_b changes the behaviour logit. This is DISTINCT from the
    monitor's detection direction once there is interference."""
    w = U[:, cfg.behaviour_feat].copy()
    return w / (np.linalg.norm(w) + EPS)


def behaviour_labels(H: np.ndarray, w_b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Binary behaviour y = 1[b > tau], tau = median logit (balanced boundary)."""
    b = H @ w_b
    tau = float(np.median(b))
    return (b > tau).astype(int), tau


# --------------------------------------------------------------------------- #
# 2. Compression on the read (merge = keep info, prune = drop info + discrete)
# --------------------------------------------------------------------------- #
# NOTE on MERGE: information-preserving merge is modelled generatively by
# `reads_packed` above (fixed D, rising feature-packing), NOT by compressing a
# fixed read. That is the faithful analogue of ToMe token-merging for a read-space
# toy: averaging across the hidden dimension would destroy the behaviour direction
# and is NOT what ToMe does (ToMe averages across tokens, preserving each hidden
# dimension's content). PRUNE below is the information-losing, discrete arm.


def compress_prune(H0: np.ndarray, keep: float) -> np.ndarray:
    """Information-LOSING compression with a DISCRETE selection: per example keep
    the m coordinates of largest magnitude, zero the rest. m = round(keep*D).
    Mirrors activation-magnitude token pruning: the surviving coordinates are the
    same high-variance ones on most examples, so the read cloud collapses to a
    low-rank blob (d_eff falls) -- and the top-k selection is non-differentiable,
    which is where gradient masking enters (plan R3 / gate G2)."""
    n, D = H0.shape
    m = max(1, int(round(keep * D)))
    if m >= D:
        return H0.copy()
    out = np.zeros_like(H0)
    # index of the m largest |value| coords per row
    idx = np.argpartition(np.abs(H0), D - m, axis=1)[:, D - m:]
    rows = np.arange(n)[:, None]
    out[rows, idx] = H0[rows, idx]
    return out


# --------------------------------------------------------------------------- #
# 3. Ground-truth geometry quantities (closed form on U)
# --------------------------------------------------------------------------- #
def coherence(U: np.ndarray) -> float:
    """Max |cos| between distinct feature directions (empirical mutual coherence)."""
    G = U.T @ U
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))


def welch_floor(F: int, m: float) -> float:
    """Welch lower bound on coherence for F unit vectors in m dimensions."""
    m = max(1.0, float(m))
    if F <= m:
        return 0.0
    return float(np.sqrt((F - m) / (m * (F - 1))))


def d_eff(H: np.ndarray) -> float:
    """Participation ratio of the centred read spectrum (same def as geometry.py)."""
    Hc = H - H.mean(0, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    ev = s ** 2
    tot = ev.sum()
    return float((tot ** 2) / (ev ** 2).sum()) if tot > 0 else 1.0


def f_eff_decodable(H: np.ndarray, Z: np.ndarray, thresh: float = 0.9,
                    max_feats: int = 60, seed: int = 0) -> int:
    """F_eff proxy: number of latent features still linearly decodable at
    >= `thresh` balanced accuracy from the (compressed) read, via a cheap
    ridge probe with a train/test split. This is what the plan's H1a / H2b use
    to tell superposition (F_eff flat as carriers fall, alpha<=0.6) from
    information loss (F_eff falls with carriers, alpha->1)."""
    rng = np.random.default_rng(seed)
    n = H.shape[0]
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    Htr, Hte = H[tr], H[te]
    Xtr = np.hstack([Htr, np.ones((len(tr), 1))])
    Xte = np.hstack([Hte, np.ones((len(te), 1))])
    lam = 1e-2 * Htr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    count = 0
    F = min(max_feats, Z.shape[1])
    for f in range(F):
        yf = (Z[:, f] > 0).astype(float)
        if yf[tr].std() < 1e-6 or yf[te].std() < 1e-6:
            continue                              # degenerate class -> skip
        w = np.linalg.solve(A, Xtr.T @ yf[tr])
        pred = (Xte @ w > 0.5).astype(int)
        gt = yf[te].astype(int)
        # balanced accuracy
        p, nmask = gt == 1, gt == 0
        if p.sum() == 0 or nmask.sum() == 0:
            continue
        bal = 0.5 * ((pred[p] == 1).mean() + (pred[nmask] == 0).mean())
        if bal >= thresh:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# 4. The p2 metric, ported verbatim from geometry.py (the one under audit)
# --------------------------------------------------------------------------- #
def _cov_half(H: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    Hc = H - H.mean(0, keepdims=True)
    Sig = (Hc.T @ Hc) / max(1, H.shape[0] - 1)
    d = Sig.shape[0]
    mu = np.trace(Sig) / d
    Sig = (1.0 - shrink) * Sig + shrink * mu * np.eye(d)
    w, V = np.linalg.eigh(Sig)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def fit_monitor_p2(H: np.ndarray, y: np.ndarray, rank: int) -> np.ndarray:
    """EXACTLY the p2 monitor: U = [w_b, top PCs]. The behaviour direction w_b is
    FORCED to be the first column -> true evasion is zero by construction; any
    nonzero Gamma it reports comes only from applying P_perp to Sigma^{1/2} w_b
    (mixed coordinate systems) plus small-sample noise. This is the bug."""
    mu1 = H[y == 1].mean(0) if (y == 1).any() else H.mean(0)
    mu0 = H[y == 0].mean(0) if (y == 0).any() else np.zeros(H.shape[1])
    w_b = mu1 - mu0
    w_b = w_b / (np.linalg.norm(w_b) + EPS)
    Hc = H - H.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    basis = [w_b]
    for k in range(Vt.shape[0]):
        v = Vt[k]
        for b in basis:
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            basis.append(v / nv)
        if len(basis) >= rank:
            break
    return np.stack(basis[:rank], axis=1)


def measured_geometry_p2(H: np.ndarray, U: np.ndarray, beta: float = 1.0,
                         shrink: float = 0.2) -> Dict[str, float]:
    """The p2 S = beta/(Gamma*||a||) on a frozen monitor U. w_b recomputed as the
    class-mean diff inside (as runner does); here U already carries it as col 0."""
    w_b = U[:, 0]
    Sh = _cov_half(H, shrink=shrink)
    a = Sh @ w_b
    lev = float(np.linalg.norm(a))
    esc = a - U @ (U.T @ a)
    escn = float(np.linalg.norm(esc))
    gamma = float(escn / (lev + EPS))
    S = float(beta / max(escn, 1e-9))
    return {"leverage": lev, "gamma_r": gamma, "escape": escn,
            "inside": float(np.sqrt(max(lev**2 - escn**2, 0.0))),
            "S": S, "d_eff": d_eff(H)}


# --------------------------------------------------------------------------- #
# 5. The replacement metric (plan Phase 2.2 / 2.3), done on the toy with truth
# --------------------------------------------------------------------------- #
def fit_monitor_detect(H: np.ndarray, y: np.ndarray, rank: int) -> np.ndarray:
    """Honest monitor: a rank-r DETECTION subspace fit from data -- first axis is
    the class-mean-difference (detection) direction, then top PCs orthogonal to
    it. The behaviour's causal direction w_b is NOT inserted, so monitor and
    behaviour are different objects and evasion is actually possible (the fix)."""
    mu1 = H[y == 1].mean(0) if (y == 1).any() else H.mean(0)
    mu0 = H[y == 0].mean(0) if (y == 0).any() else np.zeros(H.shape[1])
    p = mu1 - mu0
    p = p / (np.linalg.norm(p) + EPS)
    Hc = H - H.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    basis = [p]
    for k in range(Vt.shape[0]):
        v = Vt[k]
        for b in basis:
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            basis.append(v / nv)
        if len(basis) >= rank:
            break
    return np.stack(basis[:rank], axis=1)


def jacobian_metric(H: np.ndarray, w_b: np.ndarray, U: np.ndarray, tau: float,
                    beta: float = 1.0) -> Dict[str, np.ndarray]:
    """Per-example evasion metric, carrier space.

      g_b    = d(behaviour)/d(read) = w_b            (sensitivity direction)
      G_U    = d(monitor)/d(read)   = U              (what the monitor watches)
      Gamma_r= ||P_{U_perp} g_b|| / ||g_b||          (escape fraction, scale-free)
      margin = |b - tau|                             (how far from flipping)
      OC     = margin / (||g_b|| * Gamma_r)          (min-||delta|| to flip unseen)

    with the plan's decomposition  log OC = log margin - log||g_b|| - log Gamma_r.
    Linear here, so g_b is constant, but we return per-example arrays to mirror
    the real (per-image) pipeline and the paired design."""
    n = H.shape[0]
    g = w_b                                   # [D]
    gnorm = float(np.linalg.norm(g))
    p_perp = g - U @ (U.T @ g)
    esc_norm = float(np.linalg.norm(p_perp))
    gamma = esc_norm / (gnorm + EPS)
    b = H @ w_b
    margin = np.abs(b - tau)
    oc = margin / (gnorm * gamma + EPS)
    return {
        "gamma_r": np.full(n, gamma),
        "g_b_norm": np.full(n, gnorm),
        "margin": margin,
        "oc": beta * oc,
        "log_oc": np.log(beta * oc + EPS),
        "escape_dir": p_perp,                 # analytic attack direction
    }


def analytic_min_perturbation(h: np.ndarray, w_b: np.ndarray, U: np.ndarray,
                              tau: float) -> np.ndarray:
    """Closed-form minimal delta that flips the behaviour (b -> tau) while leaving
    every monitor coordinate unchanged (U^T delta = 0). Stevinson / Proposition-1
    minimal perturbation, exact in the linear toy. ||delta|| equals OC."""
    p_perp = w_b - U @ (U.T @ w_b)
    denom = float(p_perp @ p_perp) + EPS
    target = tau - float(h @ w_b)             # required change in logit
    return target * p_perp / denom


def verify_oc_against_attack(H: np.ndarray, w_b: np.ndarray, U: np.ndarray,
                             tau: float, n_check: int = 32, seed: int = 0
                             ) -> Dict[str, float]:
    """Confirm the OC formula by (a) the closed-form delta and (b) an independent
    projected-gradient attack, and check both actually flip the behaviour while
    the monitor stays put. Agreement => the attack is non-masked and OC is real."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(H.shape[0], size=min(n_check, H.shape[0]), replace=False)
    m = jacobian_metric(H, w_b, U, tau)
    oc_formula, oc_closed, oc_pgd = [], [], []
    mon_leak, flipped = [], []
    for i in idx:
        h = H[i]
        oc_formula.append(m["oc"][i])
        # (a) closed form
        d_closed = analytic_min_perturbation(h, w_b, U, tau)
        oc_closed.append(float(np.linalg.norm(d_closed)))
        mon_leak.append(float(np.linalg.norm(U.T @ d_closed)))
        b_new = float((h + d_closed) @ w_b)
        flipped.append(abs(b_new - tau) < 1e-6)
        # (b) independent projected-gradient descent on ||delta|| s.t. flip, U^T d=0
        d = np.zeros_like(h)
        target = tau - float(h @ w_b)
        for _ in range(400):
            resid = (float((h + d) @ w_b) - (tau + np.sign(target) * 1e-3))
            grad_flip = resid * w_b
            d = d - 0.5 * grad_flip
            d = d - U @ (U.T @ d)             # project onto monitor-blind subspace
        oc_pgd.append(float(np.linalg.norm(d)))
    oc_formula = np.array(oc_formula); oc_closed = np.array(oc_closed); oc_pgd = np.array(oc_pgd)
    rel = lambda a, b: float(np.median(np.abs(a - b) / (np.abs(b) + EPS)))
    return {
        "rel_formula_vs_closed": rel(oc_formula, oc_closed),
        "rel_pgd_vs_closed": rel(oc_pgd, oc_closed),
        "monitor_leak_max": float(np.max(mon_leak)),
        "flip_rate": float(np.mean(flipped)),
    }


def masking_index_through_prune(H0: np.ndarray, w_b: np.ndarray, U: np.ndarray,
                                tau: float, keep: float, n_check: int = 24,
                                seed: int = 0) -> Dict[str, float]:
    """Mini gradient-masking certification (plan Phase 3.3 / gate G2) on the
    DISCRETE prune. Attacker perturbs the PRE-compression read; behaviour is read
    after prune(top-m). Two attacks:

      naive-gradient: trusts d(b_pruned)/d(h0) with the mask held fixed -> it can
        only move coordinates that already survive -> OC_naive.
      gradient-free : also allowed to PROMOTE a currently-pruned coordinate into
        the kept set (a move whose gradient is exactly zero = masked) -> OC_free.

    masking index = OC_naive / OC_free. ~1 means no masking; >> 1 means the
    discrete selection faked robustness (the false "robust monitor at low N")."""
    rng = np.random.default_rng(seed)
    n, D = H0.shape
    m = max(1, int(round(keep * D)))
    idx = rng.choice(n, size=min(n_check, n), replace=False)
    oc_naive, oc_free = [], []
    for i in idx:
        h0 = H0[i]
        order = np.argsort(np.abs(h0))[::-1]
        kept = order[:m]
        pruned = order[m:]
        b0 = float(h0[kept] @ w_b[kept])
        target = tau - b0
        # --- naive-gradient attack: confined to kept coords, blind in U_perp ---
        wk = np.zeros(D); wk[kept] = w_b[kept]
        p_perp = wk - U @ (U.T @ wk)
        denom = float(p_perp @ p_perp) + EPS
        oc_naive.append(abs(target) / np.sqrt(denom) if denom > EPS else np.inf)
        # --- gradient-free: also try promoting each pruned coord above the cutoff ---
        best = oc_naive[-1]
        cutoff = abs(h0[kept][-1]) if m <= len(kept) else 0.0
        for j in pruned[:20]:                 # cheap: strongest pruned candidates
            promote = (cutoff - abs(h0[j])) + 1e-3   # cost to enter kept set
            if promote < 0:
                continue
            # once promoted, coord j contributes w_b[j]; remaining flip via kept
            contrib = abs(w_b[j]) * (abs(h0[j]) + promote)
            resid = max(abs(target) - contrib, 0.0)
            cost_rest = resid / np.sqrt(denom) if denom > EPS else 0.0
            cand = np.sqrt(promote ** 2 + cost_rest ** 2)
            best = min(best, cand)
        oc_free.append(best)
    oc_naive = np.array(oc_naive, float); oc_free = np.array(oc_free, float)
    finite = np.isfinite(oc_naive) & np.isfinite(oc_free) & (oc_free > EPS)
    ratio = float(np.median(oc_naive[finite] / oc_free[finite])) if finite.any() else np.inf
    return {"masking_index": ratio,
            "oc_naive_median": float(np.median(oc_naive[finite])) if finite.any() else np.inf,
            "oc_free_median": float(np.median(oc_free[finite])) if finite.any() else np.inf}


# --------------------------------------------------------------------------- #
# 6. small helpers
# --------------------------------------------------------------------------- #
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho without scipy (rank then Pearson)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum()) + EPS
    return float((rx * ry).sum() / denom)


def fit_loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope alpha of log y vs log x (least squares). Used for F_eff ~ m^alpha."""
    lx = np.log(np.asarray(x, float) + EPS); ly = np.log(np.asarray(y, float) + EPS)
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return float(sol[0])


DEFAULT_KEEPS: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10, 0.06)