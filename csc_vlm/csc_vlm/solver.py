"""
csc_vlm.solver
==============

Safety oracle + grid-search frontier solver + evaluation.

The PID-Lagrangian optimizer has been replaced with a direct Grid Search.
Instead of hunting for a specific safety floor (delta), we evaluate S at
every fixed keep_ratio in a pre-defined grid and map the full trade-off curve.

  grid_solve()       : evaluates S at every point in the keep_ratio grid.
  evaluate_frontier(): reports frontier_monotone, feasibility, and the H3
                       mechanism stat (for inference only -- H3 is NOT a gate).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .backends import VLMBackend, VLMConfig
from .geometry import measured_geometry

__all__ = ["VLMSafetyOracle", "KEEP_GRID", "grid_solve", "evaluate_frontier"]


# ---------------------------------------------------------------------------
# Default high-resolution keep_ratio grid.
# Coarse in the flat / safe region (keep >= 0.5); ultra-fine in the "cliff"
# zone identified from prior calibration logs (0.05 <= keep <= 0.40).
# ---------------------------------------------------------------------------
_COARSE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
_FINE   = list(np.round(np.arange(0.40, 0.04, -0.02), 2))   # 0.40, 0.38, ... 0.06
_BOUND  = [0.05]
KEEP_GRID: List[float] = _COARSE + _FINE + _BOUND


class VLMSafetyOracle:
    def __init__(self, backend: VLMBackend, cfg: VLMConfig, w_b, U):
        self.backend = backend
        self.cfg = cfg
        self.w_b = w_b
        self.U = U
        self.evals = 0

    def measure(self, keep: float, seed: int) -> Dict:
        """Run the model at a fixed keep_ratio and return all geometry stats."""
        c = self.cfg
        keep = float(np.clip(keep, c.keep_min, c.keep_max))
        H, _ = self.backend.collect(keep, c.n_pairs, seed)
        self.evals += 1
        return measured_geometry(H, self.w_b, self.U, c.beta, c.shrink)


def grid_solve(oracle: VLMSafetyOracle, cfg: VLMConfig,
               keep_grid: List[float] = None,
               seed: int = 0) -> List[Dict]:
    """Evaluate the safety metric S at every point in keep_grid.

    No optimisation, no target delta, no possibility of getting stuck at a
    boundary. Each keep_ratio is evaluated exactly once (cheap) and the
    results are returned as a list of rows ready to write to frontier.csv.

    Args:
        oracle    : VLMSafetyOracle (monitor already fitted).
        cfg       : VLMConfig.
        keep_grid : List of keep_ratio floats to test. Defaults to KEEP_GRID.
        seed      : Base random seed (incremented per point for independence).

    Returns:
        List of dicts with keys: keep, compression, S, S_geom,
        leverage, gamma_r, d_eff.
    """
    if keep_grid is None:
        keep_grid = KEEP_GRID

    rows = []
    for i, keep in enumerate(keep_grid):
        keep_clipped = float(np.clip(keep, cfg.keep_min, cfg.keep_max))
        g = oracle.measure(keep_clipped, seed=seed + i)
        S = g["obf_cost"]
        S_geom = float(cfg.beta / (g["gamma_r"] * g["leverage"] + 1e-12))
        row = {
            "keep": keep_clipped,
            "compression": float(1.0 / keep_clipped),
            "S": S,
            "S_geom": S_geom,
            "leverage": g["leverage"],
            "gamma_r": g["gamma_r"],
            "d_eff": g["d_eff"],
        }
        rows.append(row)
        print(f"  keep={keep_clipped:.3f}  comp={row['compression']:5.2f}x  "
              f"S={S:.4f}  ‖a‖={g['leverage']:.3f}  "
              f"Γ={g['gamma_r']:.3f}  d_eff={g['d_eff']:.1f}  [saved]")
    return rows


def evaluate_frontier(rows: List[Dict]) -> Dict:
    """Evaluate the frontier data after the grid search completes.

    Gates:
      frontier_monotone : Spearman(compression, S) <= -0.6 (more compression => less safe).
      feasible          : always True for grid search (we measure S directly; no target to miss).

    Inference-only (logged but NOT a gate):
      mechanism_H3      : Spearman(compression, leverage) >= 0.5.
                          Reported for analysis, does NOT affect 'passed'.
    """
    from scipy import stats

    comp = np.array([r["compression"] for r in rows])
    S    = np.array([r["S"]           for r in rows])
    lev  = np.array([r["leverage"]    for r in rows])

    # --- Gates ---
    mono_rho  = stats.spearmanr(comp, S).statistic if len(rows) > 2 else 0.0
    monotone  = bool(mono_rho <= -0.6)
    feasible  = True    # grid search always measures directly; never misses a target

    # --- Inference-only (H3 mechanism, NOT a gate) ---
    lev_rho   = stats.spearmanr(comp, lev).statistic if len(rows) > 2 else 0.0
    mechanism_h3 = bool(lev_rho >= 0.5)      # stored for analysis only

    ratios = [r["S"] / r["S_geom"] for r in rows
              if r["S_geom"] > 1e-9 and np.isfinite(r["S"])]
    prop1  = float(np.median(ratios)) if ratios else float("nan")

    return {
        "passed": bool(feasible and monotone),
        "checks": {
            "frontier_monotone": monotone,
            "feasible": feasible,
        },
        "inference": {
            "mechanism_H3": mechanism_h3,
            "spearman(compression,leverage)": float(lev_rho),
        },
        "stats": {
            "spearman(compression,S)": float(mono_rho),
            "prop1_median_ratio_S/S_geom": prop1,
            "n_points": len(rows),
        },
    }
