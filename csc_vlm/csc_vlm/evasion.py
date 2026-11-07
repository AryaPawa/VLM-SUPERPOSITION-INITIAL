"""
csc_vlm.evasion  --  Stage 1 v2: per-image evasion geometry in carrier space
============================================================================

For each image, at each (method, keep), we take gradients with respect to the
k DISTINCT compressed visual carriers v (the leaf from ``compression``). Then:

  b            = LSE(pos answer logits) - LSE(neg answer logits)    (behaviour)
  m            = |b|                                                (margin)
  g_b          = db/dv                                              (behaviour gradient)
  G_U          = d(U^T h_read)/dv   (rows = monitor directions; U = rank-r probe
                                     basis on the last-token read, refit per keep)
  cap(r)       = ||P_{span G_U[:r]} g_b||^2 / ||g_b||^2   (share of the gradient the
                                                           monitor can see; nested in r)
  Gamma(r)     = sqrt(1 - cap(r))                         (escape fraction)
  T(r)         = 1 / Gamma(r)                             (OBFUSCATION TAX: how many times
                                                           more an attacker pays to stay hidden)
  A(r)         = -log cap(r)                              (log-alignment; the plan's
                                                           A = -log(1 - Gamma^2))
  OC           = m / (||g_b|| Gamma) = (m / ||g_b||) * T

  log OC = log m  - log||g_b||  + log T
           (R1 info)  (robustness)  (auditability: the claim)

Why T and A, and not only Gamma (first-principles review)
----------------------------------------------------------
With r = 8 monitor directions and d ~ 2-4k read dimensions, cap is small and
Gamma ~ 0.99, so log T = -0.5 log(1-cap) ~ cap/2. The auditability term can then
barely move OC, even when monitor alignment changes by an order of magnitude.
A (log-alignment) spreads that regime out. T states the deployment meaning: if
T ~ 1, the monitor imposes no tax at all, whatever the compression.

Nulls (geometry-matched, replacing the vacuous isotropic placebo)
-----------------------------------------------------------------
  placebo_out   a random answer pair (logit[a] - logit[c]). It follows the same
                network path as b but has a random output direction.
  placebo_read  a random direction in the read space, pulled back through J.
A trend in A that also shows up in the placebos is generic geometry, e.g. the
coverage effect E[cap] ~ r/d_eff. Only the real-minus-placebo slope counts as a
behaviour-specific effect.

Read space vs carrier space
---------------------------
We also log cap_read(r) = ||U_r^T c||^2/||c||^2 with c = db/dh_read[last]. It is
free (no extra backward pass) and goes to high rank (rank-to-audit curve).
Carrier-space cap differs from read-space cap for two reasons:
  (a) J = dh_read/dv reshapes the metric;
  (b) b also depends on v through channels that bypass h_read[last] (later layers
      attend to other positions), and a last-token monitor cannot see those.

No gradient masking here: the selection (drop_*) happens BEFORE the leaf, so the
attacker's mask is frozen (review X4 arm B). Masking is a pixel-space (Stage 2)
issue.

Monitors and nulls added for the cluster run (v2.1)
---------------------------------------------------
  * CANONICAL rank-1 probe = the class-mean-difference axis u0 (first basis
    column) with the midpoint threshold. It is the HEADLINE (R["summary"], r=1):
    no rank or PC choice to defend. r = r_max is reported alongside
    (R["summary_rmax"]); every capture is nested, so any r can be re-analysed.
  * Full-capacity (Mahalanobis-optimal) read-space detector: the minimal
    whitened move that flips b is |b| / sqrt(c^T Sigma c). Logged as c_var and
    analysed as log_full = log m - 0.5 log c^T Sigma c. If this slope is ~0 while
    log T moves, the effect is about WHICH monitor, not about detectability.
  * Incidental detection: does the canonical probe flag the PLAIN (monitor-
    unaware) minimal attack delta_unc = -1.25 b g/||g||^2? First order on every
    image (probe.detected_fo), and in the true model on n_check images
    (check.unc_probe_flipped).
  * Read-path pullback J^T c (cap_readpath) and matched placebo (cap_match:
    same monitored component as c, random unmonitored part of the same norm):
      A = [A - A_readpath] + [A_readpath - A_match] + A_match
          (bypass)          (behaviour-specific)     (generic coverage)

First-order validity (cheap preview of the Stage-2 gate): on ``n_check`` images
per cell we apply the first-order minimal evading perturbation
delta = -b * P_perp g / ||P_perp g||^2 (scaled by 1.25) in the TRUE model and
record whether the answer flips and how far the monitor moves (relative to the
class-mean gap).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import stats as st
from .stage0 import BEHAVIOURS, Stage0Config, behaviour_for  # noqa: F401  (re-export)

__all__ = ["build_monitor_basis", "nested_capture", "run_stage1", "analyze_stage1",
           "Stage1VLM", "Stage1Mock", "DEFAULT_KEEPS", "METHODS", "READ_RANKS"]

EPS = 1e-12
DEFAULT_KEEPS: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
METHODS: Tuple[str, ...] = ("pool", "drop_norm", "drop_random", "merge_dup")
READ_RANKS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


# --------------------------------------------------------------------------- #
# numpy core
# --------------------------------------------------------------------------- #
def build_monitor_basis(H: np.ndarray, y: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal [D, rank] basis: class-mean-difference (detection) axis first,
    then the top PCs orthogonalised against it. Behaviour is NOT inserted, so the
    monitor and the behaviour are different objects."""
    H = np.asarray(H, float); y = np.asarray(y)
    mu1 = H[y == 1].mean(0) if (y == 1).any() else H.mean(0)
    mu0 = H[y == 0].mean(0) if (y == 0).any() else H.mean(0)
    p = mu1 - mu0
    basis = [p / (np.linalg.norm(p) + EPS)]
    Hc = H - H.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    for v in Vt:
        for b in basis:
            v = v - (v @ b) * b
        n = np.linalg.norm(v)
        if n > 1e-6:
            basis.append(v / n)
        if len(basis) >= rank:
            break
    return np.stack(basis[:rank], axis=1)


def nested_capture(K: np.ndarray, z: np.ndarray, gn2: float, tol: float = 1e-10) -> np.ndarray:
    """cap(r) = z_r^T K_r^+ z_r / ||g||^2 for r = 1..R, where K = G G^T and
    z = G g. This is the captured fraction of g by span(first r rows of G). Uses a
    pseudo-inverse, so dependent monitor directions are handled."""
    R = len(z)
    out = np.zeros(R)
    if gn2 <= 0:
        return out
    for r in range(1, R + 1):
        Kr = K[:r, :r]
        w, V = np.linalg.eigh(0.5 * (Kr + Kr.T))
        keep = w > tol * max(w.max(), EPS)
        zr = V.T @ z[:r]
        out[r - 1] = float(np.sum(zr[keep] ** 2 / w[keep]) / gn2)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_stage1(collector, keeps=DEFAULT_KEEPS, methods=METHODS, r_max: int = 8,
               r_frozen: int = 8, read_ranks: Sequence[int] = READ_RANKS, n_images: int = 64,
               n_check: int = 4, staleness: bool = True, verbose: bool = True) -> Dict:
    keeps = sorted(set([1.0] + [float(k) for k in keeps]), reverse=True)
    R: Dict = {"keeps": keeps, "methods": list(methods), "r_max": r_max, "r_frozen": r_frozen,
               "read_ranks": list(read_ranks), "staleness": staleness,
               "config": collector.describe(), "per_image": [], "cells": []}
    for method in methods:
        for keep in keeps:
            info = collector.fit_monitors(keep, method, r_max=r_max,
                                          r_basis=max(r_max, max(read_ranks)), r_frozen=r_frozen)
            recs = collector.per_image(keep, method, n_images=n_images, r_max=r_max,
                                       r_frozen=r_frozen, read_ranks=read_ranks,
                                       staleness=staleness, n_check=n_check)
            for rec in recs:
                rec.update({"method": method, "keep": keep})
            R["per_image"].extend(recs)
            R["cells"].append({"method": method, "keep": keep, **info})
            if verbose:
                ok = [r for r in recs if r.get("ok")]
                med = lambda k: float(np.median([r[k] for r in ok])) if ok else float("nan")
                cap8 = float(np.median([r["cap_in"][min(r_max, len(r["cap_in"])) - 1] for r in ok])) if ok else float("nan")
                print(f"[stage1] {method:11s} keep={keep:.4f} n={len(ok)} |b|={med('m'):.3f} "
                      f"||g||={math.sqrt(max(med('gn2'), 0)):.3g} cap@{r_max}={cap8:.4f}", flush=True)
    # headline = the CANONICAL rank-1 probe (no rank / PC choices); r_max shown alongside
    R["summary"] = analyze_stage1(R, r=1)
    R["summary_rmax"] = analyze_stage1(R, r=r_max) if r_max > 1 else R["summary"]
    return R


def _nanmed(vals) -> float:
    a = np.array([np.nan if v is None else float(v) for v in vals], float)
    return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")


def _cap_at(v, r: int) -> Optional[float]:
    """Capture at rank r from a nested list (or a legacy scalar = top rank)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return float(v[min(r, len(v)) - 1])
    return float(v)


def _A(cap: Optional[float]) -> float:
    return float("nan") if cap is None else -math.log(float(np.clip(cap, 1e-12, 1.0)))


def _derived(rec: Dict, r: int) -> Dict:
    cap = float(np.clip(_cap_at(rec["cap_in"], r), 1e-12, 1 - 1e-12))
    out = {"log_m": math.log(rec["m"] + 1e-12),
           "neg_log_g": -0.5 * math.log(rec["gn2"] + 1e-30),
           "log_T": -0.5 * math.log(1.0 - cap),
           "A": -math.log(cap)}
    out["log_oc"] = out["log_m"] + out["neg_log_g"] + out["log_T"]
    for key in ("cap_pl_out", "cap_pl_read", "cap_frozen", "cap_readpath", "cap_match"):
        if rec.get(key) is not None:
            out["A_" + key[4:]] = _A(_cap_at(rec[key], r))
    if rec.get("cap_read"):
        rr = rec["read_ranks"]
        if r in rr:
            out["A_read"] = _A(rec["cap_read"][rr.index(r)])
    if rec.get("c_var") and rec["c_var"] > 0:
        # full-capacity detector: margin in units of the natural spread of the behaviour logit
        out["log_full"] = math.log(rec["m"] + 1e-12) - 0.5 * math.log(rec["c_var"])
    if rec.get("c_norm"):
        out["log_read_robust"] = math.log(rec["m"] + 1e-12) - math.log(rec["c_norm"] + 1e-30)
    return out


_SLOPE_KEYS = ("A", "A_match", "A_pl_out", "A_pl_read", "A_readpath", "A_frozen", "A_read",
               "log_full", "log_read_robust")


def analyze_stage1(R: Dict, r: Optional[int] = None, B: int = 2000) -> Dict:
    """Paired (within-image) slopes vs log(actual kept fraction) with cluster-bootstrap
    CIs, the exact signed decomposition of log OC at monitor rank r, placebo contrasts,
    the full-capacity detector, incidental detection, and per-keep curves. All
    verdicts are two-sided."""
    r = int(r or R["r_max"])
    S: Dict = {"r": r, "methods": {}}
    for method in R["methods"]:
        recs = [x for x in R["per_image"] if x["method"] == method and x.get("ok")]
        if not recs:
            continue
        x = np.array([math.log(z["n_carriers"] / z["n_visual"]) if z.get("n_visual") else math.log(z["keep"])
                      for z in recs])
        g = np.array([z["image"] for z in recs])
        D = [_derived(z, r) for z in recs]
        col = lambda k: np.array([d.get(k, np.nan) for d in D], float)
        dec = st.decompose(x, {"log_oc": col("log_oc"), "log_m": col("log_m"),
                               "neg_log_g": col("neg_log_g"), "log_T": col("log_T")}, g, B=B)
        sl = {k: st.paired_slope(x, col(k), g, B=B) for k in _SLOPE_KEYS if np.isfinite(col(k)).any()}
        diffs = {}
        # A - A_x for every null. The cleanest split of the real slope is
        #   A = [A - A_readpath]  (bypass channels: paths the last-token read cannot see)
        #     + [A_readpath - A_match]  (specificity: both are pullbacks of read directions with
        #                                the SAME read-space capture at every r <= r_max, so no
        #                                ceiling/floor asymmetry)
        #     + A_match  (generic coverage geometry)
        for pk in ("A_match", "A_pl_out", "A_pl_read"):
            if pk in sl:
                diffs["A_minus_" + pk[2:]] = st.paired_slope_diff(x, col("A"), col(pk), g, B=B)
        if "A_readpath" in sl:
            diffs["bypass(A - A_readpath)"] = st.paired_slope_diff(x, col("A"), col("A_readpath"), g, B=B)
            if "A_match" in sl:
                diffs["specific(A_readpath - A_match)"] = st.paired_slope_diff(
                    x, col("A_readpath"), col("A_match"), g, B=B)
        rows = []
        for keep in R["keeps"]:
            cell = [z for z in recs if abs(z["keep"] - keep) < 1e-12]
            if not cell:
                continue
            Dk = [_derived(z, r) for z in cell]
            capin = np.array([z["cap_in"] for z in cell])
            ri = min(r, capin.shape[1]) - 1
            row = {"keep": keep, "n": len(cell),
                   "m": float(np.median([z["m"] for z in cell])),
                   "g": float(np.median([math.sqrt(z["gn2"]) for z in cell])),
                   "Gamma": float(np.median([math.sqrt(1 - min(z["cap_in"][ri], 1 - 1e-12)) for z in cell])),
                   "T": float(np.median([math.exp(d["log_T"]) for d in Dk])),
                   "OC": float(np.median([math.exp(d["log_oc"]) for d in Dk])),
                   "OC_rel": float(np.median([math.exp(d["log_oc"]) / (z["carrier_fro"] + 1e-12)
                                              for d, z in zip(Dk, cell)])),
                   "accuracy": float(np.mean([z["correct"] for z in cell])),
                   "n_carriers": float(np.median([z["n_carriers"] for z in cell])),
                   "cap_in_curve": np.median(capin, 0).tolist(),
                   "cap": float(np.median(capin[:, ri])),
                   "cap_match": _nanmed([_cap_at(z.get("cap_match"), r) for z in cell]),
                   "cap_readpath": _nanmed([_cap_at(z.get("cap_readpath"), r) for z in cell]),
                   "cap_pl_out": _nanmed([_cap_at(z.get("cap_pl_out"), r) for z in cell]),
                   "cap_pl_read": _nanmed([_cap_at(z.get("cap_pl_read"), r) for z in cell]),
                   "full_cost": _nanmed([math.exp(d["log_full"]) if "log_full" in d else None for d in Dk])}
            pr = [z["probe"] for z in cell if z.get("probe")]
            if pr:
                row["incidental_detect_fo"] = float(np.mean([p["detected_fo"] for p in pr]))
                row["probe_shift_rel"] = float(np.median([p["shift_rel"] for p in pr]))
            if cell[0].get("cap_read"):
                cr = np.array([z["cap_read"] for z in cell])
                row["cap_read_curve"] = np.median(cr, 0).tolist()
                rr = cell[0]["read_ranks"]
                hit = [rk for rk, c in zip(rr, row["cap_read_curve"]) if c >= 0.75]
                row["rank_to_audit_T2_read"] = int(hit[0]) if hit else None
            chk = [z for z in cell if z.get("check")]
            if chk:
                row["check_flip_rate"] = float(np.mean([z["check"]["flipped"] for z in chk]))
                row["check_monitor_drift"] = float(np.median([z["check"]["monitor_drift"] for z in chk]))
                row["check_b_ratio"] = float(np.median([z["check"]["b_ratio"] for z in chk]))
                unc = [z["check"] for z in chk if "unc_flipped" in z["check"]]
                if unc:
                    row["unc_flip_rate"] = float(np.mean([c["unc_flipped"] for c in unc]))
                    row["unc_probe_detect"] = float(np.mean([c["unc_probe_flipped"] for c in unc]))
            rows.append(row)
        S["methods"][method] = {"decomposition": dec, "slopes": sl, "placebo_diffs": diffs,
                                "rows": rows}
    return S


def format_stage1(S: Dict) -> str:
    r = S["r"]
    L = [f"Stage 1 v2 summary -- monitor rank r={r}"
         + ("  (CANONICAL rank-1 probe)" if r == 1 else "")
         + "\n  beta = d log X / d log(actual kept fraction); beta>0 means X FALLS under compression"]
    for method, M in S["methods"].items():
        d, sl = M["decomposition"], M["slopes"]
        L.append(f"\n[{method}]")
        L.append("  " + st.describe_slope("log OC (total)", d["log_oc"],
                                          "OC FALLS with compression (cheaper hidden attack)",
                                          "OC RISES with compression"))
        L.append("  " + st.describe_slope("  log m (R1 info loss)", d["log_m"],
                                          "margin falls (info loss)", "margin rises"))
        L.append("  " + st.describe_slope("  -log||g|| (robustness)", d["neg_log_g"],
                                          "sensitivity rises (less robust)", "sensitivity falls"))
        L.append("  " + st.describe_slope("  log T (auditability)", d["log_T"],
                                          "monitor tax FALLS (worse auditability)",
                                          "monitor tax RISES (better auditability)"))
        c = d.get("contribution", {})
        L.append(f"  signed shares of the log-OC slope: m {c.get('log_m', float('nan')):+.2f}, "
                 f"robust {c.get('neg_log_g', float('nan')):+.2f}, tax {c.get('log_T', float('nan')):+.2f}"
                 f"  (exact recon err {d.get('recon_error', float('nan')):.1e}; shares explode if total ~ 0)")
        if "log_full" in sl:
            L.append("  " + st.describe_slope("full-capacity detector cost", sl["log_full"],
                                              "falls (even an optimal detector loses ground)",
                                              "rises"))
        for k in ("A", "A_match", "A_readpath", "A_frozen", "A_read", "A_pl_out", "A_pl_read"):
            if k in sl:
                L.append("  " + st.describe_slope(f"{k} (log-alignment)", sl[k],
                                                  "monitor sees MORE under compression",
                                                  "monitor sees LESS under compression"))
        for k, v in M["placebo_diffs"].items():
            L.append("  " + st.describe_slope(k, v, "real: monitor sees MORE than this null",
                                              "real: monitor sees LESS than this null"))
        L.append("   keep    n   |b|    ||g||   cap    capMatch  T      OC      acc  carr  "
                 "r2audit  incidental  flip/drift  unc_flip/probe")
        for w in M["rows"]:
            L.append(f"  {w['keep']:.4f} {w['n']:3d} {w['m']:6.3f} {w['g']:8.3g} {w['cap']:6.3f}  "
                     f"{w['cap_match']:6.3f}  {w['T']:6.3f} {w['OC']:8.3g} {w['accuracy']:4.2f} "
                     f"{w['n_carriers']:5.0f}  {str(w.get('rank_to_audit_T2_read')):>6s}   "
                     f"{w.get('incidental_detect_fo', float('nan')):5.2f}      "
                     f"{w.get('check_flip_rate', float('nan')):.2f}/{w.get('check_monitor_drift', float('nan')):.3f}  "
                     f"{w.get('unc_flip_rate', float('nan')):.2f}/{w.get('unc_probe_detect', float('nan')):.2f}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# VLM collector
# --------------------------------------------------------------------------- #
class Stage1VLM:
    """Carrier-space Jacobians on a real VLM. Reuses Stage0VLM (model handle,
    prompts, answer tokens, caches) so both stages measure the same thing."""

    def __init__(self, s0cfg: Stage0Config, images, labels, handle=None, pos_ids=None,
                 neg_ids=None, seed: int = 0):
        import torch
        from .stage0 import Stage0VLM
        self.torch = torch
        self.s0 = Stage0VLM(s0cfg, images, labels, handle=handle, pos_ids=pos_ids, neg_ids=neg_ids)
        self.h = self.s0.h
        if self.h.dtype == torch.float16:
            print("[stage1][warn] float16 gradients can underflow (rival R3). Prefer float32/bfloat16.")
        rng = np.random.default_rng(seed + 4242)
        V = int(self.h.model.get_output_embeddings().weight.shape[0]) \
            if self.h.model.get_output_embeddings() is not None else 1000
        banned = set(self.s0.pos_ids) | set(self.s0.neg_ids)
        cand = [i for i in rng.permutation(min(V, 30000)).tolist() if i not in banned]
        self.placebo_pair = (int(cand[0]), int(cand[1]))
        c = rng.standard_normal(self.h.hidden)
        self.c_rand = torch.tensor(c / np.linalg.norm(c), dtype=torch.float32, device=self.h.device)
        self._U0: Optional[np.ndarray] = None
        self.U_refit: Optional[np.ndarray] = None
        self.gap = 1.0

    def describe(self):
        d = self.s0.describe()
        d.update({"kind": "vlm-stage1-v2", "placebo_pair": self.placebo_pair})
        return d

    def calibrate_polarity(self, n: int = 16) -> float:
        return self.s0.calibrate_polarity(n)

    def fit_monitors(self, keep: float, method: str, r_max: int, r_basis: int,
                     r_frozen: int = 8) -> Dict:
        mon = self.s0.collect("monitor", "none" if keep >= 1.0 else method, keep)
        ok = mon["ok"]
        H, y = mon["last"][ok], mon["y"][ok]
        rb = int(min(r_basis, H.shape[0] - 1))
        self.U_refit = build_monitor_basis(H, y, rb)
        if self._U0 is None or self._U0.shape[1] < r_frozen:
            # frozen monitor = the keep=1.0 (uncompressed) fit; method-independent
            m0 = mon if keep >= 1.0 else self.s0.collect("monitor", "none", 1.0)
            self._U0 = build_monitor_basis(m0["last"][m0["ok"]], m0["y"][m0["ok"]],
                                           int(min(r_frozen, m0["ok"].sum() - 1)))
        mu1, mu0 = H[y == 1].mean(0), H[y == 0].mean(0)
        self.gap = float(np.linalg.norm(mu1 - mu0)) + EPS
        # canonical rank-1 probe = class-mean-difference axis u0 (first basis column) with the
        # midpoint threshold; and centred reads for the full-capacity (Mahalanobis-optimal) cost
        u0 = self.U_refit[:, 0]
        self.probe_tau = float(u0 @ (0.5 * (mu1 + mu0)))
        self.Hc = (H - H.mean(0, keepdims=True)).astype(np.float64)
        from .stage0 import coverage_curve
        cov = coverage_curve(H)
        return {"basis_rank": rb, "class_gap": self.gap, "cov_r90": cov["r90"], "cov_top8": cov["top8"]}

    def _grad(self, y, x, retain=True):
        g = self.torch.autograd.grad(y, x, retain_graph=retain, allow_unused=True)[0]
        return None if g is None else g.reshape(-1).float()

    def one(self, i: int, method: str, keep: float, r_max: int, r_frozen: int,
            read_ranks: Sequence[int], staleness: bool, do_check: bool) -> Dict:
        torch = self.torch
        s0, h = self.s0, self.h
        inp = s0.inputs(i)
        rl = h.read_layer
        mth = "none" if keep >= 1.0 else method
        sc = s0.scores(i, inp) if mth == "drop_query" else None
        o = h.compressed_run(inp, mth, keep, scores=sc, image_key=int(i), seed=s0.cfg.seed,
                             make_leaf=True, capture=(rl,), grad=True)
        v, hraw, logits = o["leaf"], o["hs_raw"][rl], o["logits"]
        b = torch.logsumexp(logits[s0.pos_ids], 0) - torch.logsumexp(logits[s0.neg_ids], 0)
        if s0.polarity_flipped:
            b = -b
        gb, cfull = torch.autograd.grad(b, [v, hraw], retain_graph=True, allow_unused=True)
        if gb is None:
            return {"image": int(i), "ok": False, "why": "carriers disconnected from the behaviour"}
        gb = gb.reshape(-1).float()
        c = cfull[0, -1].float() if cfull is not None else None
        hlast = hraw[0, -1].float()
        Ur = torch.tensor(self.U_refit[:, :r_max], dtype=torch.float32, device=hlast.device)
        rows = []
        for j in range(Ur.shape[1]):
            gj = self._grad((Ur[:, j] * hlast).sum(), v)
            rows.append(gj if gj is not None else torch.zeros_like(gb))
        G = torch.stack(rows, 0)
        Gf = None
        if staleness:
            Uf = torch.tensor(self._U0[:, :r_frozen], dtype=torch.float32, device=hlast.device)
            fr = []
            for j in range(Uf.shape[1]):
                gj = self._grad((Uf[:, j] * hlast).sum(), v)
                fr.append(gj if gj is not None else torch.zeros_like(gb))
            Gf = torch.stack(fr, 0)
        g_pr = self._grad((self.c_rand.to(hlast.device) * hlast).sum(), v)
        g_rp = g_mt = None
        if c is not None:
            cdet = c.detach()
            # (i) read-path pullback J^T c: the behaviour's sensitivity THROUGH the monitored read
            #     only. g_b - J^T c are bypass channels (other positions / later layers) that a
            #     last-token monitor cannot see.
            g_rp = self._grad((cdet * hlast).sum(), v)
            # (ii) matched placebo: same monitored component as c, random unmonitored component
            #      of the same norm. Starts at the SAME read-space capture as the behaviour, so the
            #      real-vs-placebo comparison has no ceiling/floor asymmetry.
            pc = Ur @ (Ur.T @ cdet)
            gen = torch.Generator(device="cpu").manual_seed(int(i) * 7919 + 17)
            w = torch.randn(cdet.shape[0], generator=gen).to(cdet.device, cdet.dtype)
            w = w - Ur @ (Ur.T @ w)
            w = w / (w.norm() + 1e-30) * (cdet - pc).norm()
            g_mt = self._grad(((pc + w) * hlast).sum(), v)
        a, cc = self.placebo_pair
        g_po = self._grad(logits[a] - logits[cc], v, retain=False)
        # ---- small Gram algebra in float64 on CPU ----
        K = (G @ G.T).double().cpu().numpy()
        gn2 = float((gb @ gb).double())
        zb = (G @ gb).double().cpu().numpy()
        cap_in = nested_capture(K, zb, gn2)
        rec = {"image": int(i), "ok": True, "b": float(b.detach()), "m": abs(float(b.detach())),
               "gn2": gn2, "cap_in": cap_in.tolist(), "y": int(s0.labels[int(i)]),
               "correct": int((float(b.detach()) > 0) == (int(s0.labels[int(i)]) == 1)),
               "n_carriers": int(o["plan"].n_carriers), "n_visual": int(o["plan"].n_visual),
               "carrier_fro": float(v.detach().float().norm()),
               "read_ranks": list(read_ranks)}
        rP = min(r_max, G.shape[0])
        # every capture below is NESTED (index r-1 = rank r), so any rank can be analysed later
        for name, gp in (("cap_pl_read", g_pr), ("cap_pl_out", g_po), ("cap_readpath", g_rp),
                         ("cap_match", g_mt)):
            if gp is None or float(gp @ gp) <= 0:
                rec[name] = None
                continue
            rec[name] = nested_capture(K, (G @ gp).double().cpu().numpy(),
                                       float((gp @ gp).double())).tolist()
        if g_rp is not None:
            rec["gn2_readpath"] = float((g_rp @ g_rp).double())
        if Gf is not None:
            Kf = (Gf @ Gf.T).double().cpu().numpy()
            rec["cap_frozen"] = nested_capture(Kf, (Gf @ gb).double().cpu().numpy(), gn2).tolist()
        # canonical rank-1 probe score, and the first-order INCIDENTAL detection of the plain
        # (monitor-unaware) minimal attack delta_unc = -1.25 b g / ||g||^2: does the probe's
        # decision flip along with the answer?
        bb0 = float(b.detach())
        s_probe = float(np.dot(self.U_refit[:, 0], hlast.detach().double().cpu().numpy()))
        ds_unc = -1.25 * bb0 * float(zb[0]) / max(gn2, 1e-30)
        rec["probe"] = {"s": s_probe, "tau": self.probe_tau, "ds_unc": ds_unc,
                        "detected_fo": bool((s_probe > self.probe_tau) != (s_probe + ds_unc > self.probe_tau)),
                        "shift_rel": abs(ds_unc) / self.gap}
        if c is not None:
            cn = c.double().cpu().numpy()
            cn2 = float(cn @ cn)
            proj = self.U_refit.T @ cn
            rec["cap_read"] = [float(np.sum(proj[:min(rk, len(proj))] ** 2) / max(cn2, 1e-30))
                               for rk in read_ranks]
            rec["c_norm"] = math.sqrt(cn2)
            # full-capacity (Mahalanobis-optimal) detector cost in read space: the minimal
            # whitened move that flips b is |b| / sqrt(c^T Sigma c) -- "margin in units of the
            # natural spread of the behaviour logit". Needs only Sigma (no inverse), and the
            # quadratic form of a fixed c is unbiased for n < d.
            Hc = self.Hc
            rec["c_var"] = float(np.sum((Hc @ cn) ** 2) / max(1, Hc.shape[0] - 1))
        # ---- first-order validity check in the true model ----
        if do_check and gn2 > 0:
            with torch.no_grad():
                Gp = G[:rP].double()
                Kp = Gp @ Gp.T
                zp = Gp @ gb.double()
                coef = torch.linalg.lstsq(Kp, zp[:, None]).solution[:, 0]
                g_perp = gb.double() - Gp.T @ coef
                gp2 = float(g_perp @ g_perp)
                if gp2 > 0:
                    delta = (-1.25 * float(b.detach()) / gp2) * g_perp
                    v_new = v.detach().double() + delta.view_as(v).double()
                    o2 = h.compressed_run(inp, mth, keep, scores=sc, image_key=int(i),
                                          seed=s0.cfg.seed, capture=(rl,),
                                          carriers_override=v_new.to(v.dtype))
                    b2 = float(torch.logsumexp(o2["logits"][s0.pos_ids], 0)
                               - torch.logsumexp(o2["logits"][s0.neg_ids], 0)) * (-1 if s0.polarity_flipped else 1)
                    dh = (o2["hs"][rl][-1].float() - hlast.detach()).double().cpu().numpy()
                    drift = float(np.abs(self.U_refit[:, :rP].T @ dh).max() / self.gap)
                    bb = float(b.detach())
                    rec["check"] = {"b_before": bb, "b_after": b2, "b_ratio": b2 / (bb + 1e-12),
                                    "flipped": bool(np.sign(b2) != np.sign(bb)),
                                    "monitor_drift": drift, "delta_norm": math.sqrt(float(delta @ delta))}
                    # plain attack (no monitor term): does the canonical probe flag it in the true model?
                    du = (-1.25 * bb / gn2) * gb.double()
                    o3 = h.compressed_run(inp, mth, keep, scores=sc, image_key=int(i),
                                          seed=s0.cfg.seed, capture=(rl,),
                                          carriers_override=(v.detach().double() + du.view_as(v)).to(v.dtype))
                    b3 = float(torch.logsumexp(o3["logits"][s0.pos_ids], 0)
                               - torch.logsumexp(o3["logits"][s0.neg_ids], 0)) * (-1 if s0.polarity_flipped else 1)
                    s3 = float(np.dot(self.U_refit[:, 0], o3["hs"][rl][-1].double().cpu().numpy()))
                    rec["check"].update({"unc_flipped": bool(np.sign(b3) != np.sign(bb)),
                                         "unc_probe_flipped": bool((s3 > self.probe_tau) != (s_probe > self.probe_tau)),
                                         "unc_delta_norm": math.sqrt(float(du @ du))})
        return rec

    def per_image(self, keep, method, n_images, r_max, r_frozen, read_ranks, staleness, n_check):
        out, errs = [], 0
        for c, i in enumerate(self.s0.eval_idx[:n_images]):
            try:
                out.append(self.one(int(i), method, keep, r_max, r_frozen, read_ranks, staleness,
                                    do_check=c < n_check))
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # noqa: BLE001
                errs += 1
                if errs == c + 1 and errs >= 3:
                    raise RuntimeError(f"first {errs} images failed in Stage 1 ({method}@{keep}): "
                                       f"{type(e).__name__}: {e}") from e
                print(f"[stage1][warn] image {int(i)} {method}@{keep}: {type(e).__name__}: {str(e)[:120]}")
                out.append({"image": int(i), "ok": False, "why": str(e)[:200]})
            finally:
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
        return out


# --------------------------------------------------------------------------- #
# mock collector (no torch)
# --------------------------------------------------------------------------- #
class Stage1Mock:
    """Synthetic per-image records with KNOWN slopes, for checking the analysis.
      evasive : monitor alignment falls with compression (cap ~ keep^0.8), margin
                and ||g|| flat, placebo flat. Expect A slope < 0, log T slope > 0,
                and A beats the placebo.
      infoloss: the margin falls with keep, alignment and placebo flat.
    Logic fixture only."""

    def __init__(self, n=64, R=8, seed=0):
        self.n, self.R, self.seed = n, R, seed

    def describe(self):
        return {"kind": "mock-stage1-v2", "n": self.n}

    def fit_monitors(self, keep, method, r_max, r_basis, r_frozen=8):
        return {"basis_rank": r_basis, "class_gap": 1.0, "cov_r90": 10, "cov_top8": 0.5}

    def per_image(self, keep, method, n_images, r_max, r_frozen, read_ranks, staleness, n_check):
        out = []
        for i in range(min(n_images, self.n)):
            rng = np.random.default_rng([self.seed, i, int(keep * 1e4), len(method)])
            off = np.random.default_rng([self.seed, i]).normal()
            base_cap = 0.3 * math.exp(0.3 * off)
            if method == "evasive":
                cap = base_cap * keep ** 0.8
                m = math.exp(0.2 * off) * (1 + 0.05 * rng.normal())
            else:
                cap = base_cap
                m = math.exp(0.2 * off) * keep ** 0.6 * (1 + 0.05 * rng.normal())
            cap *= math.exp(0.05 * rng.normal())
            curve = [min(0.99, cap * (r / self.R) ** 0.5) for r in range(1, self.R + 1)]
            # matched placebo: starts at the behaviour's keep=1 capture, then stays flat
            mcurve = [min(0.99, base_cap * (r / self.R) ** 0.5 * math.exp(0.05 * rng.normal()))
                      for r in range(1, self.R + 1)]
            nest = lambda c0: [c0 * (r / self.R) ** 0.5 for r in range(1, self.R + 1)]
            s_pr = 0.5 * (1 if i % 2 else -1) + 0.1 * rng.normal()
            rec = {"image": i, "ok": True, "b": m, "m": m, "gn2": 4.0 * math.exp(0.1 * rng.normal()),
                   "cap_in": curve, "cap_readpath": list(curve), "cap_match": mcurve,
                   "cap_pl_out": nest(0.02 * math.exp(0.05 * rng.normal())),
                   "cap_pl_read": nest(0.01 * math.exp(0.05 * rng.normal())),
                   "cap_frozen": [0.9 * c_ for c_ in curve], "y": i % 2, "correct": 1,
                   "n_carriers": max(1, int(64 * keep)), "n_visual": 64, "carrier_fro": 10.0,
                   "read_ranks": list(read_ranks), "c_norm": 1.0, "c_var": 0.25,
                   "probe": {"s": s_pr, "tau": 0.0, "ds_unc": -0.2 * s_pr,
                             "detected_fo": False, "shift_rel": 0.2},
                   "cap_read": [min(0.99, curve[-1] * (rk / self.R) ** 0.5) for rk in read_ranks]}
            if i < n_check:
                rec["check"] = {"b_before": m, "b_after": -0.25 * m, "b_ratio": -0.25, "flipped": True,
                                "monitor_drift": 0.01, "delta_norm": 1.0, "unc_flipped": True,
                                "unc_probe_flipped": False, "unc_delta_norm": 0.5}
            out.append(rec)
        return out
