"""
csc.phase2
==========

Phase 2: run the solver chosen at the bake-off (PID-Lagrangian) against the REAL
attack on the real Variant-B token substrate -- the honest, VLM-like setting.

What changes from the bake-off
------------------------------
The bake-off used a smooth analytic safety surrogate. Here the safety signal is
the actual minimal-evading-perturbation S measured by the attack (delta_star) on
a rebuilt token geometry, and it has all three properties of the VLM setting:

  * measured on the real geometry (not a formula),
  * NON-differentiable in the compression knob (token count is discrete) -> the
    solver must estimate dS/dN by finite differences over attack calls (zeroth
    order),
  * NOISY -- each evaluation medians S over a fresh random set of planted pairs,
    so the estimate carries sampling variance (exactly what PID damping is for).

Problem
-------
Continuous compression knob N (token budget). More compression = fewer tokens.
    maximize    C(N) = N_ref / N        (compression ratio; optimiser drives log C)
    subject to  S(N) >= delta           (real-attack representational safety)
                N in [N_min, N_max]      (structural box)
Sweeping delta traces the empirical compression-safety Pareto frontier (T-BALANCE),
with the shadow price lambda recovered at each operating point and the Welch
coherence floor overlaid to show the L1 feasibility boundary is respected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .model import ToyConfig, ToyModel
from .attack import delta_star_analytic

__all__ = [
    "Phase2Config", "PIDGains", "RealSafetyOracle",
    "pid_solve", "pareto_sweep", "evaluate_phase2",
]


@dataclass
class Phase2Config:
    F_ref: int = 32
    N_ref: int = 32
    d_eff: int = 32
    k_per_token: int = 1
    monitor_rank: int = 4
    monitor_angle_deg: float = 30.0
    n_pairs: int = 16          # medians S over this many planted pairs (noise source)
    N_min: float = 3.0
    N_max: float = 32.0
    seed: int = 0


@dataclass
class PIDGains:
    Kp: float = 1.5
    Ki: float = 0.3
    Kd: float = 0.3


class RealSafetyOracle:
    """Safety known only by attacking a rebuilt token geometry. Returns a noisy
    median-S estimate and a finite-difference dS/dN; counts attack calls."""

    def __init__(self, cfg: Phase2Config):
        self.cfg = cfg
        self.attack_calls = 0
        self.builds = 0

    def _S_at(self, N_int: int, seed: int):
        c = self.cfg
        N_int = int(np.clip(round(N_int), 2, 100000))
        mcfg = ToyConfig(F=c.F_ref, d_eff=c.d_eff, N=N_int, N0=c.N_ref,
                         k_per_token=c.k_per_token, geometry="token",
                         monitor_rank=c.monitor_rank,
                         monitor_angle_deg=c.monitor_angle_deg, seed=seed)
        model = ToyModel(mcfg)
        self.builds += 1
        rng = np.random.default_rng(seed + 911)
        vals = []
        for _ in range(c.n_pairs):
            _, w_b, U = model.plant_pair(rng)
            r = delta_star_analytic(model.W, w_b, U)
            self.attack_calls += 1
            if np.isfinite(r.delta_input) and r.delta_input > 0:
                vals.append(r.delta_input)
        S = float(np.median(vals)) if vals else np.inf
        return S, model

    def safety(self, N: float, seed: int) -> float:
        return self._S_at(N, seed)[0]

    def value_and_fd_grad(self, N: float, h: float, seed: int):
        """Central finite difference over whole-token steps (zeroth-order)."""
        c = self.cfg
        nP = int(round(min(c.N_max, N + h)))
        nM = int(round(max(c.N_min, N - h)))
        if nP == nM:                                   # widen so the span is >= 1 token
            nP = min(int(c.N_max), nM + 1)
        S0, model = self._S_at(N, seed)
        Sp, _ = self._S_at(nP, seed + 1)
        Sm, _ = self._S_at(nM, seed + 2)
        span = max(1, nP - nM)
        dSdN = (Sp - Sm) / span                        # > 0: more tokens -> safer
        return S0, dSdN, model


def _comp_ratio(N: float, N_ref: int) -> float:
    return float(N_ref) / float(N)


def pid_solve(oracle: RealSafetyOracle, delta: float, cfg: Phase2Config,
              gains: PIDGains = PIDGains(), steps: int = 100, lr: float = 15.0,
              h: float = 2.0, margin: float = 0.05,
              N_init: float | None = None, seed: int = 0) -> Dict:
    """PID-Lagrangian on the real-attack constraint. Drives log-compression
    (objective f(N) = log N, f'(N) = 1/N) so objective and safety gradients are
    comparably scaled; reports the compression RATIO N_ref/N."""
    N = float(N_init if N_init is not None else 0.6 * cfg.N_max)
    lam, integral, prev_e = 0.0, 0.0, 0.0
    traj = {"N": [], "S": [], "C": [], "viol": [], "lam": []}
    model = None
    for k in range(steps):
        S, dSdN, model = oracle.value_and_fd_grad(N, h, seed=seed + 7 * k)
        e = (delta + margin) - S                        # tighten by margin for feasibility under noise
        integral = float(np.clip(integral + e, -50.0, 50.0))
        deriv = e - prev_e
        prev_e = e
        lam = max(0.0, gains.Kp * e + gains.Ki * integral + gains.Kd * deriv)
        grad = (1.0 / cfg.N_ref) - lam * dSdN          # min N/N_ref => O(1) multiplier
        N = float(np.clip(N - lr * grad, cfg.N_min, cfg.N_max))
        traj["N"].append(N); traj["S"].append(S)
        traj["C"].append(_comp_ratio(N, cfg.N_ref))
        traj["viol"].append(max(0.0, e)); traj["lam"].append(lam)
    # PID settles into a small limit cycle at the boundary -> report the tail
    # average as the operating point, and the shadow price from the KKT balance
    # lambda* = (dObjective/dN) / (dS/dN) = (1/N*) / (dS/dN) at N*.
    tail = max(3, int(0.3 * steps))
    N_star = float(np.mean(traj["N"][-tail:]))
    N_star = float(np.clip(N_star, cfg.N_min, cfg.N_max))
    S_final, dSdN_star, model = oracle.value_and_fd_grad(N_star, h, seed=seed + 9999)
    lam_star = float((1.0 / cfg.N_ref) / dSdN_star) if dSdN_star > 1e-9 else float("nan")
    at_bound = (N_star <= cfg.N_min + 1e-6) or (N_star >= cfg.N_max - 1e-6)
    return {
        "delta": float(delta), "N": N_star, "S": float(S_final),
        "C": _comp_ratio(N_star, cfg.N_ref), "lam": lam_star,
        "lam_pid_tail": float(np.mean(traj["lam"][-tail:])),
        "coherence": float(model.coherence()) if model else np.nan,
        "welch": float(model.welch_floor()) if model else np.nan,
        "at_bound": bool(at_bound),
        "traj": {k: np.asarray(v) for k, v in traj.items()},
    }


def pareto_sweep(cfg: Phase2Config, deltas: Sequence[float],
                 gains: PIDGains = PIDGains(), steps: int = 100, lr: float = 15.0,
                 h: float = 2.0, margin: float = 0.05, seed: int = 0, verbose: bool = True):
    oracle = RealSafetyOracle(cfg)
    rows = []
    for i, delta in enumerate(deltas):
        res = pid_solve(oracle, delta, cfg, gains, steps, lr, h, margin, seed=seed + 100 * i)
        rows.append(res)
        if verbose:
            flag = " (box)" if res["at_bound"] else ""
            print(f"  δ={delta:4.2f} -> N*={res['N']:5.2f}  ratio C*={res['C']:5.2f}  "
                  f"S={res['S']:.3f}  λ={res['lam']:.3f}  coh={res['coherence']:.3f}  "
                  f"welch={res['welch']:.3f}{flag}")
    return rows, oracle


def evaluate_phase2(rows: List[Dict], feas_tol: float = 0.12) -> Dict:
    """G-PHASE2: PID hits the safety boundary (or a box bound) at every δ; the
    frontier is monotone (more safety costs compression); Welch respected."""
    from scipy import stats

    deltas = np.array([r["delta"] for r in rows])
    Cs = np.array([r["C"] for r in rows])
    # (1) constraint met at each operating point (boundary S≈δ, unless box-limited)
    # feasibility-respecting: the operating point honours S >= delta (within a
    # small tolerance for oracle noise + whole-token quantisation)
    hit = [bool(r["S"] >= r["delta"] - feas_tol) for r in rows]
    converged = bool(np.all(hit))
    # (2) monotone frontier: compression falls as the safety demand rises
    mono_rho = stats.spearmanr(deltas, Cs).statistic if len(rows) > 2 else 0.0
    monotone = bool(mono_rho <= -0.8)
    # (3) Welch respected along the frontier
    welch_ok = bool(np.all([r["coherence"] >= r["welch"] - 1e-6 for r in rows]))
    # shadow-price sanity: positive where the safety constraint is active
    active = [r for r in rows if not r["at_bound"]]
    sp_positive = bool(np.all([r["lam"] > 0 for r in active])) if active else True

    passed = converged and monotone and welch_ok and sp_positive
    return {
        "passed": passed,
        "checks": {"boundary_converged": converged, "frontier_monotone": monotone,
                   "welch_respected": welch_ok, "shadow_price_positive": sp_positive},
        "stats": {"spearman(delta, C)": float(mono_rho),
                  "n_active": len(active), "n_points": len(rows)},
    }