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


# --------------------------------------------------------------------------- #
# Token-substrate N-sweep: the honest N -> rho_rd -> S bridge.
# --------------------------------------------------------------------------- #
def token_sweep(
    F_ref: int = 32,
    d_eff: int = 32,
    N_ref: int = 32,        # reference (uncompressed) token budget; F_rd(N_ref)=F_ref
    k_per_token: int = 1,
    N_values: Sequence[int] = (2, 4, 8, 16, 24, 32),
    s: float = 0.05,
    monitor_rank: int = 4,
    monitor_angle_deg: float = 30.0,
    n_pairs: int = 40,
    beta: float = 1.0,
    tau: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Vary the token budget N at FIXED d_eff (Variant B); measure how compression
    drives read pressure and safety. One row per planted pair.

    Compression axis (Variant B): N down => more features merged into the fixed
    read, F_rd = round(F_ref * N_ref / N) up => rho_rd ~ F_rd/d_eff up, coherence
    up, leverage ||a|| up, while the fixed-width monitor keeps Gamma_r steady =>
    S = beta/(Gamma_r*||a||) down. (Contrast Variant A, which shrank the subspace
    and inverted the sign once the monitor over-covered it.)
    """
    rows = []
    for N in N_values:
        cfg = ToyConfig(
            F=F_ref, d_eff=d_eff, N=N, N0=N_ref, k_per_token=k_per_token, s=s,
            geometry="token", monitor_rank=monitor_rank,
            monitor_angle_deg=monitor_angle_deg, seed=seed + N,
        )
        model = ToyModel(cfg)
        F_rd = model.F                          # effective read features at this N
        coh = model.coherence()
        wf = model.welch_floor()                # welch_floor(F_rd, d_eff)
        acts = model.read_activations(2000)
        feats = model.sample_features(2000)
        d_eff_meas = est.d_eff(acts)
        f_eff_meas = est.f_eff(feats, floor=1e-4)
        rho = est.rho_rd(f_eff_meas, d_eff_meas)

        pair_rng = np.random.default_rng(seed + N + 7)
        for p in range(n_pairs):
            _, w_b, U = model.plant_pair(pair_rng)
            res = delta_star_analytic(model.W, w_b, U, beta=beta, tau=tau)
            rows.append(dict(
                N=N, k_per_token=k_per_token, F_rd=F_rd, d_eff=d_eff,
                load=F_rd / d_eff, rho_blk=model.rho_blk,
                coherence=coh, welch_floor=wf,
                d_eff_meas=d_eff_meas, f_eff_meas=f_eff_meas, rho_rd=rho,
                pair=p, gamma_r=res.gamma_r, leverage=res.leverage,
                log_alignment=est.log_alignment(res.gamma_r),
                S=res.delta_input, feasible=res.feasible,
            ))
        if verbose:
            import numpy as _np
            svals = [r["S"] for r in rows if r["N"] == N and _np.isfinite(r["S"])]
            med = _np.nanmedian(svals) if svals else float("nan")
            print(f"  N={N:3d}  F_rd={F_rd:4d}  load={F_rd/d_eff:5.1f}  "
                  f"coh={coh:.3f}  d_eff_meas={d_eff_meas:5.1f}  rho_rd={rho:6.2f}  "
                  f"medianS={med:.3f}")
    return pd.DataFrame(rows)


__all__ += ["token_sweep"]