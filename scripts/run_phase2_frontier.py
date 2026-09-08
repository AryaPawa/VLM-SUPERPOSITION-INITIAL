#!/usr/bin/env python3
"""
run_phase2_frontier.py -- Phase 2: PID-Lagrangian on the REAL attack.

Runs the bake-off winner (PID-Lagrangian) against the actual delta_star attack on
the Variant-B token substrate, where the safety signal is measured (not a
formula), non-differentiable (finite-differenced over attack calls), and noisy
(medianed over random planted pairs). Sweeping the safety floor delta traces the
empirical compression-safety Pareto frontier (T-BALANCE), with the recovered
shadow price at each point and the Welch coherence floor overlaid.

Outputs (results/):
    phase2_frontier.csv     -- one operating point per delta
    figure_phase2.png       -- frontier, shadow price, a convergence trace, Welch

Usage:
    python scripts/run_phase2_frontier.py [--quick]
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

from csc.phase2 import (Phase2Config, PIDGains, pareto_sweep,   # noqa: E402
                        evaluate_phase2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.quick:
        cfg = Phase2Config(F_ref=32, N_ref=32, d_eff=32, n_pairs=14, N_min=3.0, N_max=32.0)
        deltas = [1.0, 1.3, 1.6, 1.9]
        steps = 80
    else:
        cfg = Phase2Config(F_ref=32, N_ref=32, d_eff=32, n_pairs=20, N_min=3.0, N_max=32.0)
        deltas = [0.7, 0.9, 1.1, 1.4, 1.7, 2.0]
        steps = 100

    print(f"[phase2] PID-Lagrangian vs the REAL delta_star attack "
          f"({cfg.n_pairs} pairs/eval, finite-difference dS/dN)")
    rows, oracle = pareto_sweep(cfg, deltas, steps=steps, seed=0)

    df = pd.DataFrame([{k: r[k] for k in
                        ("delta", "N", "C", "S", "lam", "coherence", "welch", "at_bound")}
                       for r in rows])
    df.to_csv(os.path.join(args.outdir, "phase2_frontier.csv"), index=False)

    ev = evaluate_phase2(rows)
    print("\n=== G-PHASE2 ===")
    for k, v in ev["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    print(f"  spearman(delta, C) = {ev['stats']['spearman(delta, C)']:+.3f}   "
          f"attack calls = {oracle.attack_calls}")
    print(f"  VERDICT: {'PASSED' if ev['passed'] else 'FAILED'}")

    # ------------------------------------------------------------------ plot #
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Phase 2 — compression–safety frontier on the REAL attack "
                 "(PID-Lagrangian, token substrate)", fontsize=12, fontweight="bold")

    d = np.array([r["delta"] for r in rows])
    C = np.array([r["C"] for r in rows])
    S = np.array([r["S"] for r in rows])
    lam = np.array([r["lam"] for r in rows])
    coh = np.array([r["coherence"] for r in rows])
    wel = np.array([r["welch"] for r in rows])

    # (a) the Pareto frontier: achieved safety vs compression -------------- #
    a = ax[0, 0]
    order = np.argsort(S)
    a.plot(S[order], C[order], "o-", color="#1f77b4", lw=1.6)
    for r in rows:
        a.annotate(f"δ={r['delta']:.1f}", (r["S"], r["C"]), fontsize=7,
                   textcoords="offset points", xytext=(5, 4))
    a.set_xlabel("achieved safety  S = ‖δ*‖"); a.set_ylabel("compression ratio  N_ref / N*")
    a.set_title("(a) T-BALANCE frontier: safety costs compression")
    a.grid(alpha=0.3)

    # (b) shadow price of safety ------------------------------------------ #
    b = ax[0, 1]
    b.plot(d, lam, "s-", color="#2ca02c", lw=1.6)
    b.set_xlabel("safety floor  δ"); b.set_ylabel(r"shadow price  λ = (1/N_ref)/(dS/dN)")
    b.set_title("(b) price of safety rises as you demand more")
    b.grid(alpha=0.3)

    # (c) a convergence trace under the noisy oracle ---------------------- #
    c = ax[1, 0]
    mid = rows[len(rows) // 2]
    t = mid["traj"]
    it = np.arange(len(t["S"]))
    c.plot(it, t["S"], color="#d62728", lw=1.2, label="measured S")
    c.axhline(mid["delta"], ls="--", color="k", lw=1, label=f"δ = {mid['delta']:.1f}")
    c2 = c.twinx()
    c2.plot(it, t["N"], color="#7f7f7f", lw=1.0, alpha=0.7, label="token budget N")
    c2.set_ylabel("token budget N", color="#7f7f7f")
    c.set_xlabel("PID iteration"); c.set_ylabel("safety S")
    c.set_title(f"(c) PID settles to the boundary (δ={mid['delta']:.1f}) under noise")
    c.legend(fontsize=7, loc="upper right"); c.grid(alpha=0.3)

    # (d) Welch floor respected along the frontier ------------------------ #
    e = ax[1, 1]
    x = np.arange(len(rows))
    e.bar(x - 0.2, coh, 0.4, label="coherence μ", color="#1f77b4")
    e.bar(x + 0.2, wel, 0.4, label="Welch floor (L1)", color="#d62728")
    e.set_xticks(x); e.set_xticklabels([f"δ{r['delta']:.1f}" for r in rows], fontsize=8)
    e.set_ylabel("coherence"); e.set_title("(d) L1 feasibility floor respected")
    e.legend(fontsize=8); e.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = os.path.join(args.outdir, "figure_phase2.png")
    fig.savefig(figpath, dpi=130)
    plt.close(fig)

    print(f"\n[phase2] wrote {os.path.join(args.outdir, 'phase2_frontier.csv')}")
    print(f"[phase2] wrote {figpath}")
    return 0 if ev["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())