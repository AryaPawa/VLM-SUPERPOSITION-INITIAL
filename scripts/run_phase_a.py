#!/usr/bin/env python3
"""
run_phase_a.py -- execute the Phase-A / G0 gate end to end.

  1. sweep the toy geometry grid (+ a few PGD cross-checks),
  2. aggregate to per-cell statistics,
  3. evaluate the G0 gate,
  4. write results/phase_a_pairs.csv, results/phase_a_cells.csv, results/figure0.png,
  5. print the gate report.

Usage:
    python scripts/run_phase_a.py [--quick] [--pgd] [--trained]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csc import (SweepGrid, run_sweep, aggregate_cells, evaluate_gate,
                 GateThresholds, figure0, delta_star_analytic)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grid for a fast smoke run")
    ap.add_argument("--pgd", action="store_true", help="include PGD cross-checks")
    ap.add_argument("--trained", action="store_true", help="add trained-Elhage geometry")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    geometry = ("random",)
    if args.trained:
        geometry = ("random", "trained")

    if args.quick:
        grid = SweepGrid(
            F=(32, 64, 128), d_eff=(8, 16, 32, 64),
            s=(0.05,), geometry=geometry, monitor_angle_deg=(30.0,),
            n_pairs=25, pgd_check=args.pgd, seed=0,
        )
    else:
        grid = SweepGrid(
            F=(32, 64, 128, 256, 512), d_eff=(8, 16, 32, 64, 128),
            s=(0.02, 0.05, 0.1), geometry=geometry,
            monitor_angle_deg=(15.0, 30.0, 45.0),
            n_pairs=40, pgd_check=args.pgd, seed=0,
        )

    print(f"[phase-a] running sweep: {len(list(grid.cells()))} cells "
          f"x {grid.n_pairs} pairs, geometry={geometry}")
    pair_df = run_sweep(grid)
    cell_df = aggregate_cells(pair_df)

    pair_path = os.path.join(args.outdir, "phase_a_pairs.csv")
    cell_path = os.path.join(args.outdir, "phase_a_cells.csv")
    pair_df.to_csv(pair_path, index=False)
    cell_df.to_csv(cell_path, index=False)

    # ---- PGD cross-check summary (X1: analytic vs empirical agree) ------- #
    if args.pgd and "delta_read_pgd" in pair_df.columns:
        chk = pair_df.dropna(subset=["delta_read_pgd"])
        chk = chk[np.isfinite(chk["delta_read"]) & np.isfinite(chk["delta_read_pgd"])]
        if len(chk):
            ratio = (chk["delta_read_pgd"] / chk["delta_read"]).median()
            print(f"[phase-a] PGD/analytic median ratio = {ratio:.3f} "
                  f"(should be ~1; empirical confirms closed form)")

    fig_path = figure0(pair_df, cell_df, os.path.join(args.outdir, "figure0.png"))

    result = evaluate_gate(cell_df, pair_df, GateThresholds())
    print()
    print(result.report())
    print()
    print(f"[phase-a] wrote {pair_path}")
    print(f"[phase-a] wrote {cell_path}")
    print(f"[phase-a] wrote {fig_path}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
