"""
csc.sweep
=========

Run the Phase-A grid over the toy geometry and, per cell, over many planted
(behaviour, monitor) probe pairs. Produces a tidy per-pair DataFrame and a
per-cell aggregate DataFrame with every quantity the G0 gate needs.

A "cell" is one (F, d_eff, N, s, geometry, angle) configuration. Within a cell
we plant ``n_pairs`` random probes and record, for each, gamma_r, log-alignment
A, and the exact minimal evading perturbation ||delta*||. The safety factor per
pair is S = ||delta*_read||.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import estimators as est
from .attack import delta_star_analytic, delta_star_pgd
from .model import ToyConfig, ToyModel

__all__ = ["SweepGrid", "run_sweep", "aggregate_cells"]


class SweepGrid:
    """Container for the swept axes; defaults follow review X1 ranges."""

    def __init__(
        self,
        F: Sequence[int] = (32, 64, 128, 256, 512),
        d_eff: Sequence[int] = (8, 16, 32, 64, 128),
        N: Sequence[int] | None = None,
        s: Sequence[float] = (0.02, 0.05, 0.1),
        geometry: Sequence[str] = ("random",),
        monitor_angle_deg: Sequence[float] = (15.0, 30.0, 45.0),
        monitor_rank: int = 4,
        n_pairs: int = 40,
        beta: float = 1.0,
        tau: float = 0.0,
        seed: int = 0,
        pgd_check: bool = False,
    ):
        self.F = list(F)
        self.d_eff = list(d_eff)
        self.N = list(N) if N is not None else list(d_eff)  # default: N tracks d_eff
        self.s = list(s)
        self.geometry = list(geometry)
        self.monitor_angle_deg = list(monitor_angle_deg)
        self.monitor_rank = monitor_rank
        self.n_pairs = n_pairs
        self.beta = beta
        self.tau = tau
        self.seed = seed
        self.pgd_check = pgd_check

    def cells(self) -> Iterable[tuple]:
        return itertools.product(
            self.F, self.d_eff, self.s, self.geometry, self.monitor_angle_deg
        )


def run_sweep(grid: SweepGrid, verbose: bool = True) -> pd.DataFrame:
    """Execute the sweep; return a per-pair tidy DataFrame."""
    rows = []
    rng_master = np.random.default_rng(grid.seed)
    cells = list(grid.cells())
    for ci, (F, d, s, geom, angle) in enumerate(cells):
        cfg = ToyConfig(
            F=F, d_eff=d, N=d, s=s, geometry=geom,
            coupling="identity", monitor_rank=grid.monitor_rank,
            monitor_angle_deg=angle, seed=int(rng_master.integers(1 << 31)),
        )
        model = ToyModel(cfg)
        coh = model.coherence()
        wf = model.welch_floor()
        gap = coh - wf
        # F_eff / d_eff from sampled read activations (endogenous, measured)
        acts = model.read_activations(2000)
        feats = model.sample_features(2000)
        d_eff_meas = est.d_eff(acts)
        f_eff_meas = est.f_eff(feats, floor=1e-4)
        rho = est.rho_rd(f_eff_meas, d_eff_meas)
        superposed = F > d

        pair_rng = np.random.default_rng(cfg.seed + 7)
        for p in range(grid.n_pairs):
            b_idx, w_b, U = model.plant_pair(pair_rng)
            res = delta_star_analytic(model.W, w_b, U, beta=grid.beta, tau=grid.tau)
            A = est.log_alignment(res.gamma_r)
            row = dict(
                cell=ci, F=F, d_eff_nominal=d, N=d, s=s, geometry=geom,
                monitor_angle_deg=angle, monitor_rank=grid.monitor_rank,
                superposed=superposed, load=F / d,
                coherence=coh, welch_floor=wf, coherence_gap=gap,
                d_eff_meas=d_eff_meas, f_eff_meas=f_eff_meas, rho_rd=rho,
                pair=p, gamma_r=res.gamma_r, log_alignment=A,
                leverage=res.leverage, delta_read=res.delta_input,  # S == input-space
                delta_input=res.delta_input, feasible=res.feasible,
            )
            if grid.pgd_check and p < 3:  # cheap: check a few pairs per cell
                row["delta_read_pgd"] = delta_star_pgd(
                    model.W, w_b, U, beta=grid.beta, tau=grid.tau, seed=cfg.seed + p
                )
            rows.append(row)
        if verbose and (ci % max(len(cells) // 10, 1) == 0):
            print(f"  cell {ci+1}/{len(cells)}  F={F} d={d} load={F/d:.1f} "
                  f"coh={coh:.3f} welch={wf:.3f} rho={rho:.2f}")
    return pd.DataFrame(rows)


def aggregate_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-pair frame to per-cell medians/means for gate checks."""
    # finite delta only (drop infeasible/robust-monitor rows for central stats)
    finite = df.replace([np.inf, -np.inf], np.nan)
    g = finite.groupby("cell")
    agg = g.agg(
        F=("F", "first"), d_eff_nominal=("d_eff_nominal", "first"),
        N=("N", "first"), s=("s", "first"), geometry=("geometry", "first"),
        monitor_angle_deg=("monitor_angle_deg", "first"),
        superposed=("superposed", "first"), load=("load", "first"),
        coherence=("coherence", "first"), welch_floor=("welch_floor", "first"),
        coherence_gap=("coherence_gap", "first"),
        rho_rd=("rho_rd", "first"), d_eff_meas=("d_eff_meas", "first"),
        f_eff_meas=("f_eff_meas", "first"),
        gamma_r_mean=("gamma_r", "mean"),
        log_alignment_mean=("log_alignment", "mean"),
        delta_read_median=("delta_read", "median"),
        delta_read_mean=("delta_read", "mean"),
        delta_input_median=("delta_input", "median"),
    )
    # per-cell infeasible fraction from the raw frame (robust-monitor rate)
    infeasible_frac = df.assign(
        inf=lambda x: ~np.isfinite(x["delta_read"])
    ).groupby("cell")["inf"].mean()
    agg["infeasible_frac"] = infeasible_frac
    return agg.reset_index()
