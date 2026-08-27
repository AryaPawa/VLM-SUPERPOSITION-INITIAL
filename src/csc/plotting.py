"""
csc.plotting
============

Figure 0 -- the figure that makes a theory-inclined reviewer relax. Four panels:

  (a) COLLAPSE : ||delta*|| vs rho_rd across every sweep, coloured by geometry.
                 The prediction is a single decreasing curve.
  (b) WELCH    : empirical coherence vs load F/d, overlaid on the Welch floor.
  (c) MEDIATE  : ||delta*|| vs log-alignment A (should be the tight law beta/Gamma).
  (d) CEILING  : max achievable safety (delta_max) vs read dimension d_eff --
                 the feasibility ceiling L1 predicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def figure0(pair_df: pd.DataFrame, cell_df: pd.DataFrame, path: str) -> str:
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Figure 0 — Phase-A toy substrate: the compression→auditability chain",
                 fontsize=13, fontweight="bold")

    pp = pair_df.replace([np.inf, -np.inf], np.nan)

    # (a) collapse: delta* vs rho_rd -------------------------------------- #
    a = ax[0, 0]
    for geom, sub in pp.groupby("geometry"):
        a.scatter(sub["rho_rd"], sub["delta_read"], s=8, alpha=0.3, label=geom)
    # median trend
    csup = cell_df[cell_df["superposed"]].sort_values("rho_rd")
    a.plot(csup["rho_rd"], csup["delta_read_median"], "k-o", ms=4, lw=1.5,
           label="per-cell median")
    a.set_xlabel(r"read-position pressure  $\rho_{rd}=F_{eff}/d_{eff}$")
    a.set_ylabel(r"safety  $S=\|\delta^*\|$")
    a.set_yscale("log"); a.set_title("(a) OC collapses onto ρ_rd")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # (b) Welch: coherence vs load --------------------------------------- #
    b = ax[0, 1]
    cc = cell_df.sort_values("load")
    b.plot(cc["load"], cc["coherence"], "o", ms=4, label="empirical coherence")
    b.plot(cc["load"], cc["welch_floor"], "r--", lw=1.5, label="Welch floor")
    b.set_xlabel(r"load  $F / d_{eff}$")
    b.set_ylabel(r"coherence  $\mu$")
    b.set_title("(b) coherence tracks the Welch floor (L1)")
    b.legend(fontsize=8); b.grid(alpha=0.3)

    # (c) mediation: delta* vs A ----------------------------------------- #
    c = ax[1, 0]
    sc = c.scatter(pp["log_alignment"], pp["delta_read"], s=8, alpha=0.3,
                   c=pp["rho_rd"], cmap="viridis")
    c.set_xlabel(r"log-alignment  $A=-\log(1-\Gamma_r^2)$")
    c.set_ylabel(r"safety  $S=\|\delta^*\|$")
    c.set_yscale("log"); c.set_title("(c) A mediates the effect (S = β/Γ_r)")
    plt.colorbar(sc, ax=c, label=r"$\rho_{rd}$"); c.grid(alpha=0.3)

    # (d) feasibility ceiling: delta_max vs d_eff ------------------------ #
    d = ax[1, 1]
    ceil = pp.groupby("d_eff_nominal")["delta_read"].max().reset_index()
    d.plot(ceil["d_eff_nominal"], ceil["delta_read"], "s-", ms=5)
    d.set_xlabel(r"read dimension  $d_{eff}$")
    d.set_ylabel(r"max achievable safety  $\delta_{max}$")
    d.set_yscale("log"); d.set_title("(d) feasibility ceiling δ_max(d_eff)")
    d.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
