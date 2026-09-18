"""
csc_vlm.solver
==============

Safety oracle + calibration + PID-Lagrangian frontier solver + evaluation.
Calibration is separated from the sweep so the runner can checkpoint it and the
per-delta operating points independently.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .backends import VLMBackend, VLMConfig
from .geometry import measured_geometry

__all__ = ["VLMSafetyOracle", "calibrate", "restore_calibration",
           "pid_solve", "evaluate_frontier"]


class VLMSafetyOracle:
    def __init__(self, backend: VLMBackend, cfg: VLMConfig, w_b, U):
        self.backend = backend
        self.cfg = cfg
        self.w_b = w_b
        self.U = U
        self.evals = 0
        self.S_ref = 1.0
        self.g_ref = 10.0
        self.keep_start = 0.3 * cfg.keep_max

    def _S_at(self, keep: float, seed: int):
        c = self.cfg
        keep = float(np.clip(keep, c.keep_min, c.keep_max))
        H, _ = self.backend.collect(keep, c.n_pairs, seed)      # fresh samples = noise
        self.evals += 1
        g = measured_geometry(H, self.w_b, self.U, c.beta, c.shrink)
        return g["obf_cost"], g

    def safety(self, keep, seed):
        return self._S_at(keep, seed)[0]

    def value_and_fd_grad(self, keep, h, seed, reps=3):
        """Median-denoised value + finite-difference gradient. dS/dkeep is
        known-positive (monotone: keeping more tokens is safer), so we clip it
        positive and bounded for a stable primal-dual step."""
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
        dSdkeep = float(np.clip((Sp - Sm) / span, 0.5, 40.0))
        return S0, dSdkeep, g


def calibrate(oracle: VLMSafetyOracle, cfg: VLMConfig, n_deltas: int = 6,
              verbose: bool = True) -> Dict:
    """Probe an INTERIOR keep grid (away from the box and the high-keep plateau
    where dS/dkeep -> 0), median over seeds, target the steep band, and set the
    O(1) reference scales. Returns a dict that fully restores calibration on
    resume."""
    grid = np.linspace(1.6 * cfg.keep_min, 0.40 * cfg.keep_max, 6)
    s_grid = [float(np.median([oracle.safety(k, seed=sd) for sd in (11, 13, 15, 17, 19)]))
              for k in grid]
    lo, hi = float(min(s_grid)), float(max(s_grid))
    deltas = list(np.linspace(lo + 0.12 * (hi - lo), hi - 0.12 * (hi - lo), n_deltas))
    S_ref = float(np.median(s_grid))
    gslopes = np.diff(s_grid) / np.diff(grid)
    g_ref = float(np.clip(np.median(gslopes), 1.0, 60.0))
    keep_start = float(np.mean(grid))
    oracle.S_ref, oracle.g_ref, oracle.keep_start = S_ref, g_ref, keep_start
    if verbose:
        print(f"  [calibrated] steep band S in [{lo:.3f},{hi:.3f}] over "
              f"keep in [{grid[0]:.3f},{grid[-1]:.3f}]; "
              f"S_ref={S_ref:.3f} g_ref={g_ref:.2f} keep_start={keep_start:.3f}")
    return {"deltas": deltas, "S_ref": S_ref, "g_ref": g_ref,
            "keep_start": keep_start, "s_grid": s_grid, "grid": grid.tolist()}


def restore_calibration(oracle: VLMSafetyOracle, calib: Dict):
    oracle.S_ref = float(calib["S_ref"])
    oracle.g_ref = float(calib["g_ref"])
    oracle.keep_start = float(calib["keep_start"])


def pid_solve(oracle: VLMSafetyOracle, delta: float, cfg: VLMConfig,
              Kp=1.2, Ki=0.2, Kd=0.2, steps=70, lr=0.02, h=0.08,
              margin=0.05, seed=0) -> Dict:
    """PID-Lagrangian over the keep-ratio knob: minimise keep (maximise
    compression) s.t. S(keep) >= delta, with O(1) normalisation (S_ref, g_ref)."""
    S_ref = float(oracle.S_ref)
    g_ref = float(oracle.g_ref)
    keep = float(oracle.keep_start)                     # warm start at band center
    lam, integral, prev_ec = 0.0, 0.0, 0.0
    traj = {"keep": [], "S": [], "lam": []}
    g = None
    for k in range(steps):
        S, dSdk, g = oracle.value_and_fd_grad(keep, h, seed=seed + 7 * k)
        ec = ((delta + margin) - S) / S_ref
        integral = float(np.clip(integral + ec, -6.0, 6.0))
        deriv = ec - prev_ec; prev_ec = ec
        lam = float(np.clip(Kp * ec + Ki * integral + Kd * deriv, 0.0, 6.0))
        grad = 1.0 - lam * (dSdk / g_ref)
        keep = float(np.clip(keep - lr * grad, cfg.keep_min, cfg.keep_max))
        traj["keep"].append(keep); traj["S"].append(S); traj["lam"].append(lam)
    tail = max(3, int(0.35 * steps))
    keep_star = float(np.mean(traj["keep"][-tail:]))
    S_reps, dS_reps, geos = [], [], []
    for sd in (9991, 9993, 9995):
        Sv, dsv, gv = oracle.value_and_fd_grad(keep_star, h, seed=sd)
        S_reps.append(Sv); dS_reps.append(dsv); geos.append(gv)
    S_star = float(np.median(S_reps))
    dSdk_star = float(np.median(dS_reps))
    g = geos[len(geos) // 2]
    lam_star = float(1.0 / dSdk_star) if dSdk_star > 1e-6 else float("nan")
    return {"delta": float(delta), "keep": keep_star, "compression": 1.0 / keep_star,
            "S": S_star, "lam": lam_star,
            "leverage": g["leverage"], "gamma_r": g["gamma_r"], "d_eff": g["d_eff"],
            "S_geom": float(cfg.beta / (g["gamma_r"] * g["leverage"] + 1e-12)),
            "at_bound": bool(keep_star <= cfg.keep_min + 1e-6 or
                             keep_star >= cfg.keep_max - 1e-6)}


def evaluate_frontier(rows: List[Dict], feas_k: float = 0.12,
                      feas_floor: float = 0.4) -> Dict:
    """Gate on the scientifically meaningful properties: monotone frontier,
    H3 mechanism (leverage rises with compression), and feasibility WITHIN the
    oracle's measurement noise (~1 sigma ~= feas_k*delta)."""
    from scipy import stats
    d = np.array([r["delta"] for r in rows])
    comp = np.array([r["compression"] for r in rows])
    lev = np.array([r["leverage"] for r in rows])
    tol = np.maximum(feas_floor, feas_k * d)
    feasible = bool(np.all([r["S"] >= r["delta"] - t for r, t in zip(rows, tol)]))
    mono = stats.spearmanr(d, comp).statistic if len(rows) > 2 else 0.0
    monotone = bool(mono <= -0.8)
    lev_rho = stats.spearmanr(comp, lev).statistic if len(rows) > 2 else 0.0
    mechanism = bool(lev_rho >= 0.5)
    track = float(np.median(np.abs([r["S"] - r["delta"] for r in rows])))
    ratios = [r["S"] / r["S_geom"] for r in rows if r["S_geom"] > 1e-9 and np.isfinite(r["S"])]
    prop1 = float(np.median(ratios)) if ratios else float("nan")
    return {"passed": bool(feasible and monotone and mechanism),
            "checks": {"frontier_monotone": monotone, "mechanism_H3": mechanism,
                       "feasible": feasible},
            "stats": {"spearman(delta,compression)": float(mono),
                      "spearman(compression,leverage)": float(lev_rho),
                      "median|S-delta|": track,
                      "prop1_median_ratio_S/S_geom": prop1,
                      "n_points": len(rows)}}
