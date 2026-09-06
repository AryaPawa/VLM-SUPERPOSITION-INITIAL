#!/usr/bin/env python3
"""
run_solver_bakeoff.py -- Phase-2 prep: race five constrained-optimization solvers
at the G0 toy gate and select empirically (plan v3 §4).

The safety constraint is a NOISY oracle with NO cheap projection (a stand-in for
"run an adversarial attack"), which is the property that separates dual-style
solvers (PID-Lagrangian, hybrid) from penalty/ALM and from structural ADMM.

Outputs (results/):
    solver_bakeoff_clean.csv   -- clean-run metrics per solver
    solver_bakeoff_robust.csv  -- metrics across the noise sweep
    figure_bakeoff.png         -- convergence, feasibility, robustness, shadow price

Usage:
    python scripts/run_solver_bakeoff.py [--quick]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csc.solvers import (BakeoffProblem, SolverConfig, SOLVERS, SafetyOracle,   # noqa: E402
                         true_optimum, run_bakeoff, evaluate_solver_bakeoff,
                         _metrics)

COLORS = {"penalty": "#d62728", "ALM": "#ff7f0e", "PID-Lagrangian": "#2ca02c",
          "ADMM(struct)": "#9467bd", "hybrid": "#1f77b4"}


def _trajectories(p, cfg):
    """Re-run each solver once (noiseless) keeping full trajectories for plotting."""
    out = {}
    for name, fn in SOLVERS.items():
        oracle = SafetyOracle(p, noise=0.0, seed=0)
        out[name] = fn(p, oracle, cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    p = BakeoffProblem(L=8, beta=1.0, gamma0=0.4, delta=0.5, rho_max=3.0)
    cfg = SolverConfig(steps=400 if args.quick else 800, lr=0.02)
    noises = (0.0, 0.1) if args.quick else (0.0, 0.05, 0.1, 0.2)
    seeds = (0, 1) if args.quick else (0, 1, 2)

    rho_star, C_star, lam_star, price = true_optimum(p)
    print(f"[bakeoff] ground truth: C* = {C_star:.4f}   KKT multiplier λ* = "
          f"{lam_star:.4f}   price of safety (−dC*/dδ) = {price:.4f}   δ = {p.delta}")

    clean_rows, robust_rows, gt = run_bakeoff(p, cfg, noises=noises, seeds=seeds)

    clean_df = pd.DataFrame(clean_rows)
    robust_df = pd.DataFrame(robust_rows)
    clean_df.to_csv(os.path.join(args.outdir, "solver_bakeoff_clean.csv"), index=False)
    robust_df.to_csv(os.path.join(args.outdir, "solver_bakeoff_robust.csv"), index=False)

    print("\n=== clean run (noise = 0) ===")
    print(f"{'solver':16s} {'final_C':>8s} {'opt_gap':>8s} {'viol':>7s} "
          f"{'feas@it':>8s} {'tail_osc':>9s} {'lam':>7s} {'sp_err':>7s}")
    for r in clean_rows:
        print(f"{r['solver']:16s} {r['final_C']:8.3f} {r['opt_gap']:8.3f} "
              f"{r['final_violation']:7.3f} {r['iters_to_feasible']:8d} "
              f"{r['tail_osc']:9.4f} {r['lam_final']:7.3f} {r['shadow_price_err']:7.3f}")

    print("\n=== robustness (median opt_gap | feasible_rate) across noise ===")
    piv = robust_df.pivot(index="solver", columns="noise", values="median_opt_gap")
    feas = robust_df.pivot(index="solver", columns="noise", values="feasible_rate")
    for name in SOLVERS:
        cells = " ".join(f"n={c}:{piv.loc[name, c]:.2f}/{feas.loc[name, c]:.0%}"
                         for c in piv.columns)
        print(f"  {name:16s} {cells}")

    verdict = evaluate_solver_bakeoff(clean_rows, robust_rows, gt,
                                      noisy_level=0.1)
    print("\n=== G-SOLVER ===")
    for name, v in verdict["verdicts"].items():
        tag = "VIABLE" if v["viable"] else "----- "
        print(f"  [{tag}] {name:16s} clean_ok={v['clean_ok']} noisy_ok={v['noisy_ok']}")
    print(f"  WINNER: {verdict['winner']}")

    # ------------------------------------------------------------------ plot #
    trajs = _trajectories(p, cfg)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Solver bake-off at G0 — constrained compression under a noisy, "
                 "non-projectable safety oracle", fontsize=12, fontweight="bold")

    a = ax[0, 0]
    for name, res in trajs.items():
        a.plot(res.traj["C"], color=COLORS[name], label=name, lw=1.3)
    a.axhline(C_star, ls="--", color="k", lw=1, label="C* (optimum)")
    a.set_xlabel("iteration"); a.set_ylabel("compression C = Σρ")
    a.set_title("(a) convergence of the objective"); a.legend(fontsize=7); a.grid(alpha=0.3)

    b = ax[0, 1]
    for name, res in trajs.items():
        b.plot(res.traj["viol"], color=COLORS[name], label=name, lw=1.3)
    b.axhline(0, ls="--", color="k", lw=1)
    b.set_xlabel("iteration"); b.set_ylabel("safety violation  max(0, δ−S)")
    b.set_title("(b) feasibility & oscillation"); b.grid(alpha=0.3)

    c = ax[1, 0]
    for name in SOLVERS:
        sub = robust_df[robust_df["solver"] == name].sort_values("noise")
        c.plot(sub["noise"], sub["median_opt_gap"], "o-", color=COLORS[name],
               label=name, lw=1.3)
    c.set_xlabel("oracle noise σ (relative)"); c.set_ylabel("median opt_gap")
    c.set_title("(c) robustness to noisy safety estimates"); c.legend(fontsize=7); c.grid(alpha=0.3)

    d = ax[1, 1]
    names = list(SOLVERS.keys())
    sp = [next(r["shadow_price_err"] for r in clean_rows if r["solver"] == n) for n in names]
    d.bar(names, sp, color=[COLORS[n] for n in names])
    d.axhline(0.25, ls="--", color="k", lw=1, label="tolerance")
    d.set_ylabel("shadow-price error"); d.set_title("(d) shadow-price recovery")
    d.tick_params(axis="x", rotation=30); d.legend(fontsize=8); d.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = os.path.join(args.outdir, "figure_bakeoff.png")
    fig.savefig(figpath, dpi=130)
    plt.close(fig)

    print(f"\n[bakeoff] wrote {os.path.join(args.outdir, 'solver_bakeoff_clean.csv')}")
    print(f"[bakeoff] wrote {os.path.join(args.outdir, 'solver_bakeoff_robust.csv')}")
    print(f"[bakeoff] wrote {figpath}")
    return 0 if verdict["winner"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())