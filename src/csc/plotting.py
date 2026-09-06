"""
csc.plotting
============

Figure 0 -- the figure that makes a theory-inclined reviewer relax (random
substrate). Plus token-substrate and gradient-masking figures for the hardened
Phase A.

figure0 panels:
  (a) COLLAPSE : ||delta*|| vs rho_rd across every sweep.
  (b) WELCH    : empirical coherence vs load F/d, overlaid on the Welch floor.
  (c) MEDIATE  : ||delta*|| vs log-alignment A, coloured by rho_rd.
  (d) CEILING  : feasibility ceiling delta_max vs rho_rd (per-cell max S, on a
                 log-log axis) -- monotone because we bin by pressure, not by a
                 single dimension. Replaces the earlier d_eff-pooled panel.
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
    c.set_yscale("log"); c.set_title("(c) A + leverage mediate the effect")
    plt.colorbar(sc, ax=c, label=r"$\rho_{rd}$"); c.grid(alpha=0.3)

    # (d) feasibility ceiling: delta_max vs rho_rd ----------------------- #
    d = ax[1, 1]
    sup = cell_df[cell_df["superposed"]].replace([np.inf, -np.inf], np.nan)
    # per-cell ceiling = max achievable S among its pairs (already aggregated as
    # delta_read_mean/median; use the per-pair max via the pair frame instead)
    ceil = (pp[pp["superposed"]]
            .groupby("cell")
            .agg(rho_rd=("rho_rd", "first"),
                 delta_max=("delta_read", "max"),
                 delta_med=("delta_read", "median"))
            .sort_values("rho_rd"))
    d.plot(ceil["rho_rd"], ceil["delta_max"], "s-", ms=4, lw=1.2,
           label=r"$\delta_{max}$ (ceiling)")
    d.plot(ceil["rho_rd"], ceil["delta_med"], "o--", ms=3, lw=1.0, alpha=0.7,
           label="median")
    d.set_xlabel(r"read-position pressure  $\rho_{rd}$")
    d.set_ylabel(r"achievable safety  $S$")
    d.set_xscale("log"); d.set_yscale("log")
    d.set_title("(d) feasibility ceiling δ_max(ρ_rd)")
    d.legend(fontsize=8); d.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_token(tok_df: pd.DataFrame, path: str) -> str:
    """Token-substrate figure: compression (fewer N) drives the whole chain."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Figure T — token substrate: compression (N↓) drives ρ_rd↑ and S↓",
                 fontsize=13, fontweight="bold")

    df = tok_df.replace([np.inf, -np.inf], np.nan)
    per_N = df.groupby("N").agg(
        rho_rd=("rho_rd", "first"), coherence=("coherence", "first"),
        welch=("welch_floor", "first"), F_rd=("F_rd", "first"),
        S_med=("S", "median")).reset_index()

    # (a) rho_rd vs N ---------------------------------------------------- #
    a = ax[0, 0]
    a.plot(per_N["N"], per_N["rho_rd"], "o-", ms=5)
    a.set_xlabel("token budget  N"); a.set_ylabel(r"read pressure  $\rho_{rd}$")
    a.set_title("(a) fewer tokens → higher read pressure")
    a.invert_xaxis(); a.grid(alpha=0.3)

    # (b) coherence & Welch vs N ----------------------------------------- #
    b = ax[0, 1]
    b.plot(per_N["N"], per_N["coherence"], "o-", ms=5, label="coherence")
    b.plot(per_N["N"], per_N["welch"], "r--", label="Welch floor (F_rd, d_eff)")
    b.set_xlabel("token budget  N"); b.set_ylabel(r"coherence  $\mu$")
    b.set_title("(b) coherence rises as tokens shrink")
    b.invert_xaxis(); b.legend(fontsize=8); b.grid(alpha=0.3)

    # (c) safety vs N ---------------------------------------------------- #
    c = ax[1, 0]
    c.plot(per_N["N"], per_N["S_med"], "o-", ms=5)
    c.set_xlabel("token budget  N"); c.set_ylabel(r"median safety  $S$")
    c.set_yscale("log"); c.set_title("(c) compression erodes safety")
    c.invert_xaxis(); c.grid(alpha=0.3)

    # (d) collapse: S vs rho_rd, coloured by N --------------------------- #
    d = ax[1, 1]
    sc = d.scatter(df["rho_rd"], df["S"], s=10, alpha=0.4, c=df["N"], cmap="plasma")
    d.set_xlabel(r"read pressure  $\rho_{rd}$"); d.set_ylabel(r"safety  $S$")
    d.set_yscale("log"); d.set_title("(d) same collapse law, driven by N")
    plt.colorbar(sc, ax=d, label="N"); d.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_masking(mask_df: pd.DataFrame, path: str) -> str:
    """Bar chart of the four attackers' perturbation norms across planted pairs.

    The naive (honest-gradient) attacker is fooled by the soft-top-k gate and
    reports a large perturbation; BPDA and the gradient-free search recover the
    true (small) one, matching the no-gate reference.
    """
    df = mask_df.replace([np.inf, -np.inf], np.nan)
    labels = ["no-gate\n(reference)", "naive PGD\n(masked)",
              "BPDA\n(unmasked)", "gradient-free"]
    cols = ["S_nogate", "S_naive", "S_bpda", "S_gradfree"]
    meds = [np.nanmedian(df[c]) for c in cols]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, meds, color=["#888", "#d62728", "#2ca02c", "#1f77b4"])
    ax.set_ylabel(r"median read-space perturbation  $\|\eta\|$")
    ax.set_title("Gradient masking: naive attacker over-reports safety;\n"
                 "BPDA / gradient-free reveal the true erosion")
    for bar, v in zip(bars, meds):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path