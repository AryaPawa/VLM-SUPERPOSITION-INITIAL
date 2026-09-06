#!/usr/bin/env python3
"""
run_token_substrate.py -- the hardened Phase-A additions.

  1. Token N-sweep: vary the token budget N at fixed (F, d_eff) and show the
     empirical chain  N down -> rho_rd up -> S down, with the pooled Welch floor
     respected. Evaluates the G0-TOKEN gate. Writes figure_token.png.
  2. Gradient-masking probe: on several planted pairs, run the four attackers
     (no-gate reference, naive masked PGD, BPDA, gradient-free) and show that the
     soft-top-k gate fools only the naive attacker. Writes figure_masking.png.

Usage:
    python scripts/run_token_substrate.py [--quick]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csc import (token_sweep, evaluate_token_gate, evaluate_masking,   # noqa: E402
                 figure_token, figure_masking, ToyConfig, ToyModel,
                 masking_demo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sweep / fewer pairs")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------ 1 #
    print("[token] N-sweep: compression drives read pressure and safety")
    if args.quick:
        tok = token_sweep(F_ref=32, d_eff=32, N_ref=32,
                          N_values=(2, 4, 8, 16, 24, 32), n_pairs=25, seed=0)
    else:
        tok = token_sweep(F_ref=48, d_eff=48, N_ref=48,
                          N_values=(2, 4, 8, 16, 24, 32, 40, 48), n_pairs=40, seed=0)
    tok_path = os.path.join(args.outdir, "token_sweep.csv")
    tok.to_csv(tok_path, index=False)
    figt = figure_token(tok, os.path.join(args.outdir, "figure_token.png"))

    tgate = evaluate_token_gate(tok)
    print()
    print(tgate.report())

    # ------------------------------------------------------------------ 2 #
    print()
    print("[mask] gradient-masking probe (soft-top-k gate)")
    # a superposed token model so the behaviour is genuinely attackable
    # (F_ref=32, N_ref=32, N=8 -> F_rd = 32*32/8 = 128 features in d_eff=32)
    cfg = ToyConfig(F=32, d_eff=32, N=8, N0=32, k_per_token=1, geometry="token",
                    monitor_rank=4, monitor_angle_deg=30.0, seed=1)
    model = ToyModel(cfg)
    rng = np.random.default_rng(0)
    n_pairs = 8 if args.quick else 20
    rows = []
    for i in range(n_pairs):
        _, w_b, U = model.plant_pair(rng)
        res = masking_demo(model.W, w_b, U, seed=i)
        rows.append(dict(pair=i, S_nogate=res.S_nogate, S_naive=res.S_naive,
                         S_bpda=res.S_bpda, S_gradfree=res.S_gradfree,
                         masking_ratio=res.masking_ratio))
    mask_df = pd.DataFrame(rows)
    mask_path = os.path.join(args.outdir, "masking_probe.csv")
    mask_df.to_csv(mask_path, index=False)
    figm = figure_masking(mask_df, os.path.join(args.outdir, "figure_masking.png"))

    summary = evaluate_masking(mask_df)
    print("  median ||eta||:  no-gate={median_S_nogate:.3f}  naive={median_S_naive:.3f}  "
          "bpda={median_S_bpda:.3f}  gradient-free={median_S_gradfree:.3f}"
          .format(**summary))
    print(f"  median masking ratio (naive / min(bpda,gf)) = "
          f"{summary['median_masking_ratio']:.2f}  "
          f"-> masking {'DETECTED' if summary['masking_detected'] else 'not detected'}")

    print()
    print(f"[token] wrote {tok_path}")
    print(f"[token] wrote {figt}")
    print(f"[mask]  wrote {mask_path}")
    print(f"[mask]  wrote {figm}")
    return 0 if tgate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())