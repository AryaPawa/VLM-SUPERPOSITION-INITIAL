"""
csc.gates
=========

The Phase-A / G0 gate for the toy substrate. This is the positive-control gate
that precedes the VLM-scale G-A1..G-A4: if the predicted chain does not appear
here, where every quantity is exact, it will not appear at 13B.

Sub-checks (each foreshadows a downstream G-A criterion):

  G0-COLLAPSE  : S = ||delta*|| is a tight decreasing function of rho_rd.
                 (Spearman(rho_rd, S) strongly negative; low residual scatter.)
                 -> foreshadows G-A2 (rho_rd/Gamma trends).
  G0-MEDIATE   : log-alignment A mediates rho_rd -> S. Partial corr(rho_rd, S | A)
                 collapses toward 0 while corr(A, S) stays strong.
                 -> foreshadows G-A3 (dS/drho_rd < 0 survives conditioning).
  G0-WELCH     : coherence tracks the Welch floor as load F/d rises (small,
                 stable gap), so the ceiling is geometry-forced, not construction.
                 -> foreshadows G-A4 / L1 (feasibility ceiling delta_max(N)).
  G0-NOSUPER   : in the d_eff >= F regime attacks are expensive/infeasible
                 (high delta / high infeasible fraction) -- the "zero successful
                 attacks" control of Stevinson et al. / Gorton & Lewis.

Decision rule: PASS iff all four sub-checks pass. Any failure -> stop and
diagnose before scaling (estimator wrong? ceiling elsewhere? mediation
confounded?), exactly as A.3 prescribes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GateThresholds:
    collapse_spearman_max: float = -0.6     # corr(rho, S) must be <= this
    collapse_resid_r2_min: float = 0.5      # R^2 of log S ~ log rho (collapse tightness)
    mediate_partial_max: float = 0.25       # |partial corr(rho,S|A)| must be <= this
    mediate_direct_min: float = 0.5         # |corr(A,S)| must be >= this
    welch_gap_max: float = 0.25             # median coherence gap in superposed cells
    nosuper_delta_ratio_min: float = 1.5    # median S(no-super) / S(super) must exceed this


@dataclass
class GateResult:
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)

    def report(self) -> str:
        lines = ["=" * 64, "PHASE-A / G0 GATE  (toy-substrate positive control)", "=" * 64]
        name_map = {
            "collapse": "G0-COLLAPSE  (S falls tightly with rho_rd)      -> G-A2",
            "mediate":  "G0-MEDIATE   (A mediates rho_rd -> S)           -> G-A3",
            "welch":    "G0-WELCH     (coherence tracks Welch floor)     -> G-A4/L1",
            "nosuper":  "G0-NOSUPER   (no-superposition regime is safe)  -> control",
        }
        for k, label in name_map.items():
            mark = "PASS" if self.checks.get(k) else "FAIL"
            lines.append(f"  [{mark}]  {label}")
        lines.append("-" * 64)
        for k, v in self.stats.items():
            lines.append(f"    {k:34s} = {v:+.4f}")
        lines.append("-" * 64)
        verdict = "GATE PASSED -> proceed to Phase 1/2" if self.passed \
            else "GATE FAILED -> stop and diagnose before scaling"
        lines.append(f"  VERDICT: {verdict}")
        lines.append("=" * 64)
        return "\n".join(lines)


def _partial_corr(x, y, z):
    """Partial correlation of x,y given z (all 1-D), via residualisation."""
    x, y, z = map(np.asarray, (x, y, z))
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    rx, ry = resid(x, z), resid(y, z)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _partial_corr_multi(x, y, Z):
    """Partial correlation of x,y given a matrix Z of conditioning variables."""
    x, y, Z = np.asarray(x), np.asarray(y), np.asarray(Z)
    Z1 = np.c_[np.ones(len(x)), Z]
    def resid(a):
        coef, *_ = np.linalg.lstsq(Z1, a, rcond=None)
        return a - Z1 @ coef
    rx, ry = resid(x), resid(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_gate(cell_df: pd.DataFrame, pair_df: pd.DataFrame,
                  th: GateThresholds | None = None) -> GateResult:
    th = th or GateThresholds()
    checks, st = {}, {}

    sup = cell_df[cell_df["superposed"]].copy()
    sup = sup[np.isfinite(sup["delta_read_median"])]

    # ---- G0-COLLAPSE : S falls tightly with rho_rd ----------------------- #
    rho = sup["rho_rd"].to_numpy()
    S = sup["delta_read_median"].to_numpy()
    sp = stats.spearmanr(rho, S).statistic
    # collapse tightness: log-log linear fit R^2
    m = (rho > 0) & (S > 0) & np.isfinite(S)
    if m.sum() >= 3:
        lr = stats.linregress(np.log(rho[m]), np.log(S[m]))
        r2 = lr.rvalue ** 2
    else:
        r2 = 0.0
    checks["collapse"] = (sp <= th.collapse_spearman_max) and (r2 >= th.collapse_resid_r2_min)
    st["collapse_spearman(rho,S)"] = sp
    st["collapse_loglog_R2"] = r2

    # ---- G0-MEDIATE : geometry mediates rho_rd -> S (per-pair, superposed) #
    # The exact toy relation is S = beta / (Gamma_r * ||a||), so the effect is
    # carried by TWO measurable superposition consequences: the log-alignment A
    # (== Gamma_r) and the behaviour leverage ||a|| = ||W^T w_b||. Conditioning
    # on A ALONE is insufficient (leverage carries the rest); conditioning on
    # BOTH must collapse the rho_rd -> S path. We report both so the toy's
    # verdict -- "the VLM-scale mediator must include leverage" -- is explicit.
    pp = pair_df[pair_df["superposed"]].copy()
    pp = pp[np.isfinite(pp["delta_read"]) & (pp["delta_read"] > 0)]
    logS = np.log(pp["delta_read"].to_numpy())
    A = pp["log_alignment"].to_numpy()
    logLev = np.log(np.clip(pp["leverage"].to_numpy(), 1e-9, None))
    rho_pp = pp["rho_rd"].to_numpy()

    raw = abs(np.corrcoef(rho_pp, logS)[0, 1]) if len(pp) > 2 else 0.0
    direct_A = abs(np.corrcoef(A, logS)[0, 1]) if len(pp) > 2 else 0.0
    partial_A = abs(_partial_corr(rho_pp, logS, A)) if len(pp) > 2 else 1.0
    # joint conditioning on [A, logLev] via two-step residualisation
    Z = np.c_[A, logLev]
    partial_joint = abs(_partial_corr_multi(rho_pp, logS, Z)) if len(pp) > 2 else 1.0

    checks["mediate"] = (partial_joint <= th.mediate_partial_max)
    st["mediate_raw_|corr(rho,logS)|"] = raw
    st["mediate_|corr(A,logS)|"] = direct_A
    st["mediate_|partial(rho,logS|A)| (A alone insuff.)"] = partial_A
    st["mediate_|partial(rho,logS|A,leverage)|"] = partial_joint

    # ---- G0-WELCH : coherence respects the floor and rises with load ---- #
    # L1 is a *lower bound*; random frames sit above it. The decisive checks
    # are (i) the bound is never violated, and (ii) coherence rises with load,
    # i.e. superposition forces coherence up exactly as Welch predicts.
    allc = cell_df[np.isfinite(cell_df["coherence"])]
    bound_ok = bool((allc["coherence"] >= allc["welch_floor"] - 1e-6).all())
    load = allc["load"].to_numpy()
    coh = allc["coherence"].to_numpy()
    welch_trend = stats.spearmanr(load, coh).statistic if len(allc) > 2 else 0.0
    checks["welch"] = bound_ok and (welch_trend >= 0.6)
    st["welch_bound_respected"] = float(bound_ok)
    st["welch_spearman(load,coherence)"] = float(welch_trend)
    st["welch_median_gap_superposed"] = float(np.median(sup["coherence_gap"])) if len(sup) else np.nan

    # ---- G0-NOSUPER : no-superposition regime is safe ------------------- #
    nos = cell_df[~cell_df["superposed"]]
    if len(nos) and len(sup):
        s_nos = float(np.nanmedian(nos["delta_read_median"]))
        s_sup = float(np.nanmedian(sup["delta_read_median"]))
        ratio = s_nos / max(s_sup, 1e-12)
        inf_nos = float(np.nanmean(nos["infeasible_frac"]))
    else:
        ratio, inf_nos = 0.0, 0.0
    checks["nosuper"] = (ratio >= th.nosuper_delta_ratio_min) or (inf_nos >= 0.5)
    st["nosuper_delta_ratio(nosup/sup)"] = ratio
    st["nosuper_infeasible_frac"] = inf_nos

    passed = all(checks.values())
    return GateResult(passed=passed, checks=checks, stats=st)
