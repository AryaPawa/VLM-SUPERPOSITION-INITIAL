"""
csc_vlm.solver  (hardened, v0.4)
================================

Safety oracle + full-range keep-sweep + per-model calibration + PID-Lagrangian
frontier solver + evaluation.

What changed vs v0.3 (and WHY), after the first real-GPU runs:

  * Full-range keep sweep (NEW).  v0.3 only ever sampled the deep-compression
    corner (keep <= 0.4), so the dramatic Gamma_r cliff near full budget
    (Gamma_r ~ 0 at keep=1 -> ~0.7 under compression) was never drawn. The
    descriptive frontier now comes from a direct sweep over the WHOLE keep range
    and cannot degenerate (it is pure measurement, no solver).

  * delta read off the measured curve (FIX for Qwen).  v0.3 calibrated delta on
    keep in [0.08,0.40] but the PID solver then fell to keep=0.05, into a FLAT
    saturated zone where S < every delta and dS/dkeep ~ 0, so it pinned at the box
    with lambda at the cap. We now pick target keeps across the informative
    interior band, set each delta to the MEASURED S at that keep, and warm-start
    the PID there. Each delta is therefore achievable at an interior keep by
    construction, and the solver starts next to its own solution.

  * Mediator = Gamma_r / A (FIX).  On real VLMs the token-merge averaging SHRINKS
    the covariance, so leverage ||a|| FALLS with compression while Gamma_r RISES
    and carries the whole effect. The H3 mechanism check now tests Gamma_r (and
    the log-alignment A = -log(1-Gamma_r^2)); leverage is reported as secondary.

  * Frozen-vs-refit monitor control (NEW).  At every operating point we also
    re-fit the monitor on the compressed activations. The gap between S_frozen and
    S_refit separates genuine superposition erosion from mere monitor staleness.

  * Scale-correct feasibility (FIX).  v0.3 used an absolute 0.4 tolerance floor,
    which trivially "passed" Qwen's S~0.19 scale. Tolerance is now relative to the
    measured per-point noise (sigma) and the model's own S scale.

  * Box-edge finite difference (FIX).  The 0.5 dS/dkeep floor manufactured a
    lambda=2.0 artifact in flat zones; replaced with a small epsilon so a truly
    flat region is reported honestly (and avoided by calibration anyway).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .backends import VLMBackend, VLMConfig
from .geometry import measured_geometry, geometry_frozen_and_refit

__all__ = ["VLMSafetyOracle", "sweep_keep", "calibrate", "restore_calibration",
           "pid_solve", "evaluate_frontier"]


def _isotonic_increasing(y) -> np.ndarray:
    """Pool-adjacent-violators isotonic regression (non-decreasing, L2). Robust
    monotone smoother for the measured S(keep) curve -- pools noisy spikes down
    with their neighbours instead of latching onto them the way a cumulative max
    does, so delta targets track the LOCAL achievable S, not an outlier."""
    y = np.asarray(y, dtype=float)
    stack = []                                   # each: [mean, count]
    for v in y:
        cur = [float(v), 1.0]
        while stack and stack[-1][0] > cur[0]:
            p = stack.pop()
            tot = p[1] + cur[1]
            cur = [(p[0] * p[1] + cur[0] * cur[1]) / tot, tot]
        stack.append(cur)
    out = []
    for mean, cnt in stack:
        out.extend([mean] * int(round(cnt)))
    return np.asarray(out[:len(y)], dtype=float)


def _safe_spear(x, y) -> float:
    """Spearman correlation that returns 0.0 (not NaN, no warning) when an input
    is constant, too short, or non-finite -- keeps the gate robust and the logs
    clean on degenerate cells."""
    from scipy import stats
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if (len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        return 0.0
    return float(stats.spearmanr(x, y).statistic)


class VLMSafetyOracle:
    def __init__(self, backend: VLMBackend, cfg: VLMConfig, w_b, U):
        self.backend = backend
        self.cfg = cfg
        self.w_b = w_b
        self.U = U
        self.evals = 0
        # calibration scales / band (set by calibrate / restore_calibration)
        self.S_ref = 1.0
        self.g_ref = 1.0
        self.keep_start = 0.3 * cfg.keep_max
        self.keep_lo = cfg.keep_min
        self.keep_hi = cfg.keep_max

    # -- single measurement (frozen only, cheap) ---------------------------- #
    def _collect(self, keep: float, seed: int):
        c = self.cfg
        keep = float(np.clip(keep, c.keep_min, c.keep_max))
        H, y = self.backend.collect(keep, c.n_pairs, seed)       # fresh samples = noise
        self.evals += 1
        return H, y

    def _S_at(self, keep: float, seed: int):
        H, _ = self._collect(keep, seed)
        g = measured_geometry(H, self.w_b, self.U, self.cfg.beta, self.cfg.shrink)
        return g["obf_cost"], g

    def safety(self, keep, seed):
        return self._S_at(keep, seed)[0]

    # -- full measurement at a point: S (median+sigma) + frozen & refit geo -- #
    def measure_point(self, keep: float, seed: int, reps: int = 3,
                      refit: bool = True) -> Dict:
        c = self.cfg
        reps = max(1, reps)
        mid = reps // 2
        Ss, g_froz, g_refit = [], None, None
        for j in range(reps):
            H, y = self._collect(keep, seed + 101 * j)
            if refit:
                gf, gr = geometry_frozen_and_refit(H, y, self.w_b, self.U,
                                                   c.monitor_rank, c.beta, c.shrink)
            else:
                gf = measured_geometry(H, self.w_b, self.U, c.beta, c.shrink)
                gr = gf
            Ss.append(gf["obf_cost"])
            if j == mid:
                g_froz, g_refit = gf, gr
        Ss = np.asarray(Ss, dtype=float)
        return {
            "keep": float(np.clip(keep, c.keep_min, c.keep_max)),
            "compression": 1.0 / float(np.clip(keep, c.keep_min, c.keep_max)),
            "S": float(np.median(Ss)),
            "S_sigma": float(np.std(Ss)) if len(Ss) > 1 else 0.0,
            "S_refit": float(g_refit["obf_cost"]),
            "leverage": float(g_froz["leverage"]),
            "gamma_r": float(g_froz["gamma_r"]),
            "gamma_refit": float(g_refit["gamma_r"]),
            "A": float(g_froz["log_alignment"]),
            "A_refit": float(g_refit["log_alignment"]),
            "d_eff": float(g_froz["d_eff"]),
        }

    def value_and_fd_grad(self, keep, h, seed, reps=3):
        """Median-denoised value + finite-difference gradient (frozen S only,
        for speed inside the PID loop). dS/dkeep is known-positive (more tokens =
        safer), clipped to a small positive epsilon so a flat/saturated region is
        reported honestly rather than pinned to a fake floor."""
        c = self.cfg

        def med(k, s0):
            out = [self._S_at(k, s0 + 101 * j) for j in range(reps)]
            return float(np.median([o[0] for o in out])), out[reps // 2][1]

        kp = min(c.keep_max, keep + h)
        km = max(c.keep_min, keep - h)
        S0, g = med(keep, seed)
        Sp, _ = med(kp, seed + 1)
        Sm, _ = med(km, seed + 2)
        span = max(1e-3, kp - km)
        dSdkeep = float(np.clip((Sp - Sm) / span, 1e-3, 1e3))
        return S0, dSdkeep, g


# --------------------------------------------------------------------------- #
# Full-range descriptive sweep (robust; cannot degenerate)
# --------------------------------------------------------------------------- #
def sweep_keep(oracle: VLMSafetyOracle, keeps: Sequence[float],
               seeds: Sequence[int] = (11, 13, 15, 17, 19)) -> List[Dict]:
    """Measure S, Gamma_r, A, leverage, d_eff (frozen AND refit) across a keep
    grid spanning the full compression range. Pure measurement -- no optimiser --
    so it always produces a clean curve, including the near-full-budget cliff."""
    rows: List[Dict] = []
    for k in keeps:
        per = [oracle.measure_point(float(k), seed=sd, reps=1, refit=True)
               for sd in seeds]
        S = np.array([p["S"] for p in per])
        Sr = np.array([p["S_refit"] for p in per])
        rows.append({
            "keep": float(np.clip(k, oracle.cfg.keep_min, oracle.cfg.keep_max)),
            "compression": 1.0 / float(np.clip(k, oracle.cfg.keep_min, oracle.cfg.keep_max)),
            "S": float(np.median(S)), "S_sigma": float(np.std(S)),
            "S_refit": float(np.median(Sr)),
            "gamma_r": float(np.median([p["gamma_r"] for p in per])),
            "gamma_refit": float(np.median([p["gamma_refit"] for p in per])),
            "A": float(np.median([p["A"] for p in per])),
            "leverage": float(np.median([p["leverage"] for p in per])),
            "d_eff": float(np.median([p["d_eff"] for p in per])),
        })
    return sorted(rows, key=lambda r: r["keep"])


# --------------------------------------------------------------------------- #
# Per-model calibration driven by the measured sweep
# --------------------------------------------------------------------------- #
def calibrate(oracle: VLMSafetyOracle, cfg: VLMConfig, n_deltas: int = 6,
              n_sweep: int = 14, band_keep=(None, None), verbose: bool = True) -> Dict:
    """Sweep the FULL keep range, then choose delta targets by reading S off the
    measured curve at interior keeps. This guarantees every delta is achievable at
    a non-degenerate interior operating point and gives the PID a warm start, which
    is what cures the Qwen collapse.

    band_keep = (k_lo, k_hi): the interior keep band the delta targets span. If
    None, defaults to [max(keep_min*1.3, keep_min+0.01), 0.55] (compression ~1.8x
    to ~15x), which stays clear of both the saturated floor and the near-budget
    plateau (where S blows up as Gamma_r -> 0)."""
    k_lo_def = max(cfg.keep_min * 1.3, cfg.keep_min + 0.01)
    k_hi_def = min(0.55, 0.9 * cfg.keep_max)
    k_lo = float(band_keep[0]) if band_keep[0] is not None else k_lo_def
    k_hi = float(band_keep[1]) if band_keep[1] is not None else k_hi_def

    # dense sweep grid: geometric spacing packs resolution toward low keep, where
    # the cliff and the saturation both live; include a near-budget point.
    keeps = np.unique(np.clip(
        np.r_[np.geomspace(cfg.keep_min, min(0.9, cfg.keep_max), n_sweep),
              np.linspace(k_lo, k_hi, 5)],
        cfg.keep_min, cfg.keep_max))
    sweep = sweep_keep(oracle, keeps)

    ks = np.array([r["keep"] for r in sweep])
    Ss = np.array([r["S"] for r in sweep])
    S_iso = _isotonic_increasing(Ss)               # monotone smoother (robust to spikes)

    # target keeps across the interior band (geometric -> even in compression).
    # delta_i = LOCAL achievable S at k_i (read off the smoothed curve), so the
    # solver meets it by compressing to ~k_i and never chases an unachievable S.
    k_targets = np.geomspace(k_lo, k_hi, n_deltas)
    # 3% headroom below the local achievable S so the solver meets delta by
    # compressing slightly past k_i, instead of drifting up into a plateau.
    deltas = [float(np.interp(k, ks, S_iso)) * 0.97 for k in k_targets]
    # warm start = INVERSE of the measured curve: the keep where S_iso == delta,
    # i.e. the feasibility boundary / max-compression operating point. This is what
    # cures the saturated-floor collapse -- the PID starts at the known solution and
    # is bracketed around it, so it can't overshoot into the flat zone.
    S_mono = S_iso + 1e-9 * np.arange(len(S_iso))      # strictly increasing for interp
    warm = [float(np.interp(dl, S_mono, ks)) for dl in deltas]

    in_band = (ks >= k_lo) & (ks <= k_hi)
    S_ref = float(np.median(S_iso[in_band])) if in_band.any() else float(np.median(S_iso))
    slopes = np.gradient(S_iso, ks)
    g_ref = float(np.clip(np.median(np.abs(slopes[in_band])) if in_band.any()
                          else np.median(np.abs(slopes)), 1e-2, 1e3))

    # solver search bounds: a little wider than the band, clear of the exact box.
    keep_lo = float(max(cfg.keep_min, k_lo * 0.6))
    keep_hi = float(min(cfg.keep_max, k_hi * 1.6))

    oracle.S_ref, oracle.g_ref = S_ref, g_ref
    oracle.keep_lo, oracle.keep_hi = keep_lo, keep_hi
    oracle.keep_start = float(np.mean(warm))

    if verbose:
        print(f"  [calibrated] sweep S in [{Ss.min():.3f},{Ss.max():.3f}] over "
              f"keep [{ks.min():.3f},{ks.max():.3f}]")
        print(f"  [calibrated] delta band {min(deltas):.3f}..{max(deltas):.3f} at "
              f"keeps {k_lo:.3f}..{k_hi:.3f}; S_ref={S_ref:.3f} g_ref={g_ref:.3f}")
    return {"deltas": deltas, "warm": warm, "S_ref": S_ref, "g_ref": g_ref,
            "keep_lo": keep_lo, "keep_hi": keep_hi,
            "keep_start": oracle.keep_start, "band_keep": [k_lo, k_hi],
            "sweep": sweep}


def restore_calibration(oracle: VLMSafetyOracle, calib: Dict):
    oracle.S_ref = float(calib["S_ref"])
    oracle.g_ref = float(calib["g_ref"])
    oracle.keep_start = float(calib.get("keep_start", oracle.keep_start))
    oracle.keep_lo = float(calib.get("keep_lo", oracle.cfg.keep_min))
    oracle.keep_hi = float(calib.get("keep_hi", oracle.cfg.keep_max))


# --------------------------------------------------------------------------- #
# PID-Lagrangian solve for one delta (warm-started, band-bounded)
# --------------------------------------------------------------------------- #
def pid_solve(oracle: VLMSafetyOracle, delta: float, cfg: VLMConfig,
              keep_start: float = None, keep_lo: float = None, keep_hi: float = None,
              Kp=1.2, Ki=0.2, Kd=0.2, steps=50, lr=0.03, h=None,
              margin=0.0, seed=0) -> Dict:
    """Minimise keep (maximise compression) s.t. S(keep) >= delta, with O(1)
    normalisation (S_ref, g_ref). Warm-started at keep_start and clamped to the
    informative band [keep_lo, keep_hi] so it cannot fall into the saturated box
    corner that degenerated the v0.3 Qwen runs."""
    S_ref = float(oracle.S_ref)
    g_ref = float(oracle.g_ref)
    keep0 = float(np.clip(keep_start if keep_start is not None else oracle.keep_start,
                          cfg.keep_min, cfg.keep_max))
    # tight LOCAL bracket around the warm start (the inverse-curve solution), so the
    # PID refines locally and cannot collapse into the saturated box corner.
    klo = float(keep_lo) if keep_lo is not None else max(cfg.keep_min, keep0 * 0.6)
    khi = float(keep_hi) if keep_hi is not None else min(cfg.keep_max, keep0 * 2.0)
    keep = float(np.clip(keep0, klo, khi))

    lam, integral, prev_ec = 0.0, 0.0, 0.0
    traj = {"keep": [], "S": [], "lam": []}
    for k in range(steps):
        hh = float(h if h is not None else np.clip(0.12 * keep, 0.01, 0.1))
        # reps=2 is enough inside the loop: the warm start sits at the inverse-curve
        # solution and feasibility restoration cleans up afterwards, so we don't pay
        # for heavy per-step denoising. (n_pairs carries the real noise control.)
        S, dSdk, _ = oracle.value_and_fd_grad(keep, hh, seed=seed + 7 * k, reps=2)
        ec = ((delta + margin) - S) / S_ref
        integral = float(np.clip(integral + ec, -6.0, 6.0))
        deriv = ec - prev_ec; prev_ec = ec
        lam = float(np.clip(Kp * ec + Ki * integral + Kd * deriv, 0.0, 6.0))
        grad = 1.0 - lam * (dSdk / g_ref)
        keep = float(np.clip(keep - lr * grad, klo, khi))
        traj["keep"].append(keep); traj["S"].append(S); traj["lam"].append(lam)

    tail = max(3, int(0.35 * steps))
    keep_star = float(np.mean(traj["keep"][-tail:]))

    # feasibility restoration: a penalty/PID controller settles slightly INSIDE
    # feasibility (S a touch below delta). Nudge keep up along the measured curve
    # until S >= delta, so every operating point is a genuine feasible frontier
    # point (the min keep meeting the floor), not an over-compressed overshoot.
    def _medS(k):
        return float(np.median([oracle.safety(k, sd) for sd in (4001, 4003, 4005)]))
    S_op = _medS(keep_star)
    tries = 0
    while S_op < delta and keep_star < khi - 1e-6 and tries < 15:
        keep_star = float(min(khi, keep_star * 1.12 + 0.004))
        S_op = _medS(keep_star)
        tries += 1

    # final operating point: median S + sigma, frozen AND refit geometry
    pts = [oracle.measure_point(keep_star, seed=sd, reps=1, refit=True)
           for sd in (9991, 9993, 9995)]
    S_star = float(np.median([p["S"] for p in pts]))
    S_sigma = float(np.std([p["S"] for p in pts]))
    S_refit = float(np.median([p["S_refit"] for p in pts]))
    gmid = pts[len(pts) // 2]
    # shadow price from the local slope at the operating point
    _, dSdk_star, _ = oracle.value_and_fd_grad(keep_star, np.clip(0.12 * keep_star, 0.01, 0.1),
                                               seed=7777)
    # shadow price proxy (local marginal compute per unit safety); clipped to a sane
    # range so a near-flat local slope can't produce a meaningless spike on the plot.
    lam_star = float(np.clip(1.0 / dSdk_star, 0.0, 50.0)) if dSdk_star > 1e-6 else float("nan")

    return {"delta": float(delta), "keep": keep_star, "compression": 1.0 / keep_star,
            "S": S_star, "S_sigma": S_sigma, "S_refit": S_refit,
            "lam": lam_star,
            "leverage": gmid["leverage"], "gamma_r": gmid["gamma_r"],
            "gamma_refit": gmid["gamma_refit"], "A": gmid["A"],
            "d_eff": gmid["d_eff"],
            "S_geom": float(cfg.beta / (gmid["gamma_r"] * gmid["leverage"] + 1e-12)),
            # at the global compression box (keep_min = max compression) -- the
            # meaningful "hit the ceiling" flag, not the local search bracket.
            "at_bound": bool(keep_star <= cfg.keep_min + 1e-6
                             or keep_star >= cfg.keep_max - 1e-6)}


# --------------------------------------------------------------------------- #
# Gate / evaluation
# --------------------------------------------------------------------------- #
def evaluate_frontier(rows: List[Dict], S_ref: float = 1.0,
                      feas_rel: float = 0.12, feas_sigma_k: float = 2.0,
                      feas_abs_frac: float = 0.12) -> Dict:
    """G-VLM gate, corrected for the real-VLM findings:

      frontier_monotone : more safety delta  => less compression (Spearman <= -0.8)
      mechanism_H3      : Gamma_r RISES with compression (Spearman >= 0.5) -- the
                          real-VLM mediator (leverage is reported, not gated)
      feasible          : each S >= delta within measurement noise, with a
                          tolerance scaled to the model's own S scale (not a fixed
                          absolute floor)
    Also reports the frozen-vs-refit staleness read so we can see how much of the
    erosion survives keeping the monitor current.
    """
    d = np.array([r["delta"] for r in rows])
    comp = np.array([r["compression"] for r in rows])
    lev = np.array([r["leverage"] for r in rows])
    gam = np.array([r["gamma_r"] for r in rows])
    A = np.array([r.get("A", np.nan) for r in rows])

    # feasibility: tolerance = max(relative, k*sigma, small fraction of S scale)
    tol = np.array([max(feas_rel * r["delta"],
                        feas_sigma_k * r.get("S_sigma", 0.0),
                        feas_abs_frac * float(S_ref)) for r in rows])
    feasible = bool(np.all([r["S"] >= r["delta"] - t for r, t in zip(rows, tol)]))

    mono = _safe_spear(d, comp)
    monotone = bool(mono <= -0.8)

    gam_rho = _safe_spear(comp, gam)
    A_rho = _safe_spear(comp, A)
    lev_rho = _safe_spear(comp, lev)
    mechanism = bool(gam_rho >= 0.5)

    track = float(np.median(np.abs([r["S"] - r["delta"] for r in rows])))
    ratios = [r["S"] / r["S_geom"] for r in rows if r.get("S_geom", 0) > 1e-9 and np.isfinite(r["S"])]
    prop1 = float(np.median(ratios)) if ratios else float("nan")

    # staleness control: how much erosion survives a current (refit) monitor
    refit_ok = all("S_refit" in r for r in rows)
    if refit_ok:
        gam_refit = np.array([r["gamma_refit"] for r in rows])
        refit_rho = _safe_spear(comp, gam_refit)
        stale_ratio = float(np.median([r["S_refit"] / max(r["S"], 1e-9) for r in rows]))
    else:
        refit_rho, stale_ratio = float("nan"), float("nan")

    return {"passed": bool(feasible and monotone and mechanism),
            "checks": {"frontier_monotone": monotone, "mechanism_H3": mechanism,
                       "feasible": feasible},
            "stats": {"spearman(delta,compression)": float(mono),
                      "spearman(compression,gamma_r)": float(gam_rho),
                      "spearman(compression,A)": float(A_rho),
                      "spearman(compression,leverage)": float(lev_rho),
                      "median|S-delta|": track,
                      "prop1_median_ratio_S/S_geom": prop1,
                      "refit_spearman(compression,gamma_r)": float(refit_rho),
                      "staleness_median_S_refit/S_frozen": stale_ratio,
                      "n_points": len(rows)}}
