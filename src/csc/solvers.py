"""
csc.solvers
===========

The solver bake-off at G0 (plan v3 §4). We do NOT pre-commit to ADMM; we race
five constrained-optimization strategies on a controlled toy problem whose safety
constraint has the two properties that matter at VLM scale:

  * NO cheap projection  -- feasibility is only known through an oracle
    (a stand-in for "run an adversarial attack"); solvers get first-order access
    (value + gradient) only, and may NOT project onto the safe set.
  * NOISY               -- the oracle returns an estimate + noise, mirroring the
    variance of an attack-based safety estimate.

Constrained problem (toy form)
------------------------------
Decision rho in R^L : per-block superposition pressure (the compression knob).
    maximize    C(rho) = sum_l rho_l                 (compression; more is better)
    subject to  S(rho) >= delta                      (representational safety; hard)
                0 <= rho_l <= rho_max                 (structural box; cheap prox)

Safety surrogate (Variant-B regime: Gamma_r steady, leverage drives S):
    S(rho) = beta / (gamma0 * ||D rho||_2),   D = diag(sqrt(c))
so {S >= delta} <=> ||D rho||^2 <= Rmax^2,  Rmax = beta/(gamma0 delta). To keep the
constraint well-conditioned (S ~ 1/leverage is very flat far inside the unsafe
region), the oracle exposes the NORMALISED safety margin
    m(rho) = 1 - ||D rho||^2 / Rmax^2   (>= 0 feasible),  grad m = -2 D^2 rho / Rmax^2
which is the same feasible set with an O(1), non-vanishing gradient. The linear
objective + ellipsoid + box has a known optimum and KKT multiplier, so we score
each solver against ground truth -- while the SOLVERS see only the noisy oracle.

Why this separates the field
-----------------------------
  penalty        : no multiplier; converges just OUTSIDE feasibility (classic).
  ALM            : multiplier + capped penalty schedule; robust, feasible.
  PID-Lagrangian : PID-damped dual; best under a NOISY constraint (targets the
                   primal-dual non-convergence CAID reports).
  ADMM (struct)  : z-update is a prox -> only legitimate on the box; safety has no
                   prox, so it is a fixed soft penalty on the x-min and UNDER-
                   enforces (demonstrates the plan-v3 §2 mismatch).
  hybrid         : ADMM for the box + CAID-style dual descent for safety; best.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

__all__ = [
    "BakeoffProblem", "SafetyOracle", "SolverConfig", "SolverResult",
    "solve_penalty", "solve_alm", "solve_pid", "solve_admm_structural",
    "solve_hybrid", "SOLVERS", "true_optimum", "run_bakeoff",
    "evaluate_solver_bakeoff",
]


# --------------------------------------------------------------------------- #
# Problem + oracle
# --------------------------------------------------------------------------- #
@dataclass
class BakeoffProblem:
    L: int = 8
    beta: float = 1.0
    gamma0: float = 0.4          # steady escape fraction (Variant-B regime)
    delta: float = 0.5           # safety floor  S >= delta
    rho_max: float = 3.0         # structural box upper bound
    c: np.ndarray = field(default=None)  # leverage weights (diag of D^2)

    def __post_init__(self):
        if self.c is None:
            self.c = np.linspace(0.5, 4.0, self.L)
        self.c = np.asarray(self.c, dtype=float)
        self.Rmax2 = (self.beta / (self.gamma0 * self.delta)) ** 2

    def leverage(self, rho):
        return float(np.sqrt(np.sum(self.c * rho ** 2)))

    def safety(self, rho):
        lev = self.leverage(rho)
        return float(self.beta / (self.gamma0 * lev)) if lev > 1e-12 else np.inf

    def margin(self, rho):
        """Normalised safety margin m(rho) >= 0 <=> S(rho) >= delta."""
        return float(1.0 - np.sum(self.c * rho ** 2) / self.Rmax2)

    def margin_grad(self, rho):
        return -2.0 * self.c * rho / self.Rmax2

    def compression(self, rho):
        return float(np.sum(rho))

    def box(self, rho):
        return np.clip(rho, 0.0, self.rho_max)


class SafetyOracle:
    """Feasibility is known only through this oracle (no projection). Returns a
    NOISY estimate of the safety margin and its gradient; counts calls."""

    def __init__(self, problem: BakeoffProblem, noise: float = 0.0, seed: int = 0):
        self.p = problem
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def value_and_grad(self, rho):
        self.calls += 1
        m = self.p.margin(rho)
        g = self.p.margin_grad(rho)
        if self.noise > 0:
            m = m + self.rng.normal(0.0, self.noise * (abs(m) + 0.1))
            g = g + self.rng.normal(0.0, self.noise * (np.abs(g) + 1e-3))
        return m, g


# --------------------------------------------------------------------------- #
# Solver plumbing
# --------------------------------------------------------------------------- #
@dataclass
class SolverConfig:
    steps: int = 800
    lr: float = 0.05
    mu0: float = 0.5             # ALM / penalty initial penalty
    mu_growth: float = 1.08
    mu_cap: float = 40.0
    dual_period: int = 40        # ALM multiplier update cadence
    Kp: float = 1.0              # PID gains
    Ki: float = 0.15
    Kd: float = 0.5
    rho_admm: float = 1.0        # ADMM consensus penalty
    admm_safety_penalty: float = 3.0   # fixed soft safety penalty (no prox)
    eta_dual: float = 0.3        # hybrid safety dual-descent step
    feas_tol: float = 1e-2
    seed: int = 0


@dataclass
class SolverResult:
    name: str
    rho: np.ndarray
    traj: Dict[str, np.ndarray]      # per-iter scalars: C, S, viol, lam
    lam_final: float

    def final_compression(self):
        return float(np.sum(self.rho))


def _new_traj():
    return {"C": [], "S": [], "viol": [], "lam": []}


def _record(p, traj, rho, lam):
    S = p.safety(rho)
    traj["C"].append(p.compression(rho)); traj["S"].append(S)
    traj["viol"].append(max(0.0, p.delta - S)); traj["lam"].append(lam)


def _finish(name, p, rho, traj, lam):
    return SolverResult(name=name, rho=p.box(rho),
                        traj={k: np.asarray(v) for k, v in traj.items()},
                        lam_final=float(lam))


# --------------------------------------------------------------------------- #
# The five contestants  (updates use ONLY the oracle margin m and its grad g)
# --------------------------------------------------------------------------- #
def solve_penalty(p, oracle, cfg):
    rho = 0.1 * np.ones(p.L)
    mu = cfg.mu0
    traj = _new_traj()
    for k in range(cfg.steps):
        m, g = oracle.value_and_grad(rho)
        viol = max(0.0, -m)                              # infeasible if m < 0
        grad = -np.ones(p.L)
        if viol > 0:
            grad = grad - mu * viol * g                  # (mu/2)viol^2, dviol/drho = -g
        rho = p.box(rho - cfg.lr * grad)
        if (k + 1) % (cfg.dual_period * 3) == 0:
            mu = min(cfg.mu_cap, mu * cfg.mu_growth)
        _record(p, traj, rho, mu * viol)
    return _finish("penalty", p, rho, traj, mu * max(0.0, -p.margin(rho)))


def solve_alm(p, oracle, cfg):
    rho = 0.1 * np.ones(p.L)
    lam, mu = 0.0, cfg.mu0
    traj = _new_traj()
    for k in range(cfg.steps):
        m, g = oracle.value_and_grad(rho)
        active = min(0.0, m)                             # penalise violation only
        grad = -np.ones(p.L) - lam * g + mu * active * g
        rho = p.box(rho - cfg.lr * grad)
        if (k + 1) % cfg.dual_period == 0:
            lam = max(0.0, lam - mu * m)                 # multiplier update
            mu = min(cfg.mu_cap, mu * cfg.mu_growth)
        _record(p, traj, rho, lam)
    return _finish("ALM", p, rho, traj, lam)


def solve_pid(p, oracle, cfg):
    rho = 0.1 * np.ones(p.L)
    lam, integral, prev_e = 0.0, 0.0, 0.0
    traj = _new_traj()
    for k in range(cfg.steps):
        m, g = oracle.value_and_grad(rho)
        e = -m                                           # violation signal
        integral = max(0.0, integral + e)                # anti-windup
        deriv = e - prev_e
        prev_e = e
        lam = max(0.0, cfg.Kp * e + cfg.Ki * integral + cfg.Kd * deriv)
        grad = -np.ones(p.L) - lam * g
        rho = p.box(rho - cfg.lr * grad)
        _record(p, traj, rho, lam)
    return _finish("PID-Lagrangian", p, rho, traj, lam)


def solve_admm_structural(p, oracle, cfg):
    """ADMM with the box as g(z) (cheap prox = clip). Safety has NO prox, so it is
    only a fixed soft penalty on the x-min -- deliberately handicapped to show the
    §2 mismatch: without a dedicated adapting safety multiplier it under-enforces."""
    x = 0.1 * np.ones(p.L)
    z = x.copy()
    u = np.zeros(p.L)
    mu = cfg.admm_safety_penalty
    traj = _new_traj()
    for k in range(cfg.steps):
        m, g = oracle.value_and_grad(x)
        viol = max(0.0, -m)
        grad = -np.ones(p.L) + (0.0 if viol == 0 else -mu * viol * g) \
            + cfg.rho_admm * (x - z + u)
        x = x - cfg.lr * grad
        z = p.box(x + u)                                 # z-update = prox_box (cheap)
        u = u + x - z                                    # dual ascent on consensus
        _record(p, traj, z, mu * viol)
    return _finish("ADMM(struct)", p, z, traj, mu * max(0.0, -p.margin(z)))


def solve_hybrid(p, oracle, cfg):
    """ADMM for the structural box + CAID-style dual descent for safety."""
    x = 0.1 * np.ones(p.L)
    z = x.copy()
    u = np.zeros(p.L)
    lam = 0.0
    traj = _new_traj()
    for k in range(cfg.steps):
        m, g = oracle.value_and_grad(x)
        grad = -np.ones(p.L) - lam * g + cfg.rho_admm * (x - z + u)
        x = x - cfg.lr * grad
        z = p.box(x + u)                                 # structural prox
        u = u + x - z
        lam = max(0.0, lam - cfg.eta_dual * m)           # safety dual descent
        _record(p, traj, z, lam)
    return _finish("hybrid", p, z, traj, lam)


SOLVERS: Dict[str, Callable] = {
    "penalty": solve_penalty,
    "ALM": solve_alm,
    "PID-Lagrangian": solve_pid,
    "ADMM(struct)": solve_admm_structural,
    "hybrid": solve_hybrid,
}


# --------------------------------------------------------------------------- #
# Ground truth + scoring
# --------------------------------------------------------------------------- #
def true_optimum(p: BakeoffProblem):
    """Analytic ground truth via SLSQP on the exact (noiseless) problem. Returns
    (rho_star, C_star, lam_star, price_delta) where lam_star is the KKT multiplier
    on the normalised margin, and price_delta = -dC*/d(delta) is the price of
    safety (both by construction / finite difference)."""
    from scipy.optimize import minimize

    def solve_at(delta):
        q = BakeoffProblem(L=p.L, beta=p.beta, gamma0=p.gamma0, delta=delta,
                           rho_max=p.rho_max, c=p.c.copy())
        res = minimize(lambda r: -np.sum(r), 0.1 * np.ones(p.L),
                       jac=lambda r: -np.ones(p.L),
                       bounds=[(0.0, p.rho_max)] * p.L,
                       constraints=[{"type": "ineq",
                                     "fun": lambda r, q=q: q.margin(r)}],
                       method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
        return res.x, float(np.sum(res.x))

    rho_star, C_star = solve_at(p.delta)
    # KKT multiplier on the normalised margin, from any interior (unclipped) block:
    #   -1 - lam * (d m / d rho_l) = 0  with  d m/d rho_l = -2 c_l rho_l / Rmax^2
    interior = [i for i in range(p.L) if 1e-3 < rho_star[i] < p.rho_max - 1e-3]
    lam_star = float(np.median([p.Rmax2 / (2 * p.c[i] * rho_star[i]) for i in interior])) \
        if interior else float("nan")
    eps = 1e-3 * max(1.0, p.delta)
    _, C_up = solve_at(p.delta + eps)
    price_delta = -(C_up - C_star) / eps
    return rho_star, C_star, lam_star, float(price_delta)


def _metrics(p, res, C_star, lam_star, cfg):
    S_final = p.safety(res.rho)
    viol = res.traj["viol"]
    tail = max(1, len(viol) // 5)
    feas_iter = -1
    ok = res.traj["S"] >= (p.delta - cfg.feas_tol)
    for i in range(len(ok)):
        if ok[i] and ok[i:].all():
            feas_iter = i
            break
    opt_gap = (C_star - res.final_compression()) / max(abs(C_star), 1e-9)
    sp_err = abs(res.lam_final - lam_star) / max(abs(lam_star), 1e-9)
    return {
        "solver": res.name,
        "final_C": res.final_compression(),
        "opt_gap": float(opt_gap),
        "final_violation": float(max(0.0, p.delta - S_final)),
        "iters_to_feasible": int(feas_iter),
        "tail_osc": float(np.std(viol[-tail:])),
        "lam_final": res.lam_final,
        "shadow_price_err": float(sp_err),
    }


def run_bakeoff(p, cfg=None, noises=(0.0, 0.05, 0.1, 0.2), seeds=(0, 1, 2)):
    """Race all solvers across a noise sweep. Returns (clean_rows, robust_rows, gt)."""
    cfg = cfg or SolverConfig()
    rho_star, C_star, lam_star, price = true_optimum(p)
    gt = {"C_star": C_star, "lam_star": lam_star, "price_delta": price,
          "rho_star": rho_star}

    clean_rows, robust_rows = [], []
    for name, fn in SOLVERS.items():
        for noise in noises:
            gaps, viols, feas = [], [], []
            rep = None
            for sd in seeds:
                oracle = SafetyOracle(p, noise=noise, seed=1000 * sd + 7)
                c2 = SolverConfig(**{**cfg.__dict__, "seed": sd})
                res = fn(p, oracle, c2)
                m = _metrics(p, res, C_star, lam_star, c2)
                gaps.append(m["opt_gap"]); viols.append(m["final_violation"])
                feas.append(1.0 if m["final_violation"] <= cfg.feas_tol else 0.0)
                if noise == 0.0 and sd == seeds[0]:
                    rep = m
            robust_rows.append({"solver": name, "noise": noise,
                                "median_opt_gap": float(np.median(gaps)),
                                "median_violation": float(np.median(viols)),
                                "feasible_rate": float(np.mean(feas))})
            if noise == 0.0 and rep is not None:
                clean_rows.append(rep)
    return clean_rows, robust_rows, gt


def evaluate_solver_bakeoff(clean_rows, robust_rows, gt,
                            max_iters_feasible=700, max_gap_clean=0.10,
                            max_sp_err=0.25, noisy_level=0.1, max_gap_noisy=0.20):
    """G-SOLVER: which solvers are viable, and which wins. Viable = (clean) reaches
    feasibility within budget, small opt_gap, shadow price recovered; AND (noisy)
    stays feasible with a still-small gap at the target noise. Winner = viable
    solver with the smallest clean opt_gap."""
    by_clean = {r["solver"]: r for r in clean_rows}
    noisy = {r["solver"]: r for r in robust_rows
             if abs(r["noise"] - noisy_level) < 1e-9}
    verdicts, viable = {}, []
    for name, c in by_clean.items():
        n = noisy.get(name, {})
        clean_ok = (0 <= c["iters_to_feasible"] <= max_iters_feasible
                    and abs(c["opt_gap"]) <= max_gap_clean
                    and c["final_violation"] <= 1e-2
                    and c["shadow_price_err"] <= max_sp_err)
        noisy_ok = (n.get("feasible_rate", 0.0) >= 0.5
                    and abs(n.get("median_opt_gap", 1e9)) <= max_gap_noisy)
        ok = bool(clean_ok and noisy_ok)
        verdicts[name] = {"viable": ok, "clean_ok": bool(clean_ok),
                          "noisy_ok": bool(noisy_ok)}
        if ok:
            viable.append((name, abs(c["opt_gap"])))
    winner = min(viable, key=lambda t: t[1])[0] if viable else None
    return {"winner": winner, "viable": [v for v, _ in viable], "verdicts": verdicts}