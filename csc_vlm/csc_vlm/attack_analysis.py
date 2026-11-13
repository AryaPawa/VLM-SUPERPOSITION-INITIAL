"""
csc_vlm.attack_analysis  --  analysis of the pipeline-v3 attack runs (numpy only)
================================================================================

Four questions, each with a pre-registered reading.

1. INSTRUMENT (patch leaf, frozen selection). Does the first-order Stage-1 formula
   track the true minimum-norm attack?
     PASS iff, for each keep with >= 8 successful attacks, Spearman(lin, true) >= 0.5,
     AND the 95 % CI of the paired slope of log(true / lin) vs log(kept fraction) lies
     inside [-0.1, +0.1] (an equivalence test: the linearisation error may not shift
     any cost slope by more than 0.1, small against the 0.3-0.6 effects interpreted).
     INSUFFICIENT when a cell has < 8 images with successful attacks.
   If it fails, OC_lin is diagnostic only and the true costs are the primary measure.

2. COSTS in the fixed space (patch or pixel). Per method:
     OC_free (robustness), OC_thr (hidden attack), T_true = OC_thr / OC_free.
     log OC_thr = log OC_free + log T_true holds exactly per image, so the paired
     slopes add up. beta > 0 means the quantity FALLS under compression.
   Images whose clean answer is already wrong at a keep ("prebroken") cost 0, and
   images whose tripwire already fires ("prefired", thr only) cannot be attacked
   silently. Both enter the success-at-budget curves; the slopes use images with a
   finite positive cost at every keep they appear in.
   Everything is also split by attack direction (y = 1: hide the object / answer
   "no"; y = 0: induce it / answer "yes"), because compression shifts answers toward
   the default label and the two directions get different margins.

3. INCIDENTAL DETECTION. Among successful plain (free) attacks, how often does the
   tripwire fire (refit and frozen probes)? Per keep, with Wilson CIs, plus a paired
   keep-1 vs smallest-keep difference over images that succeeded at both.

4. STEERING (R2; selection-dependent methods, live vs frozen selection). Per image,
   log(OC_frozen / OC_live) > 0 means a live selection makes the attack cheaper.
   R2 for auditability = steering(thr) - steering(free): does steering help the
   HIDDEN attack more than the plain one? Mean with a bootstrap CI over images. The
   live attack uses BPDA, so this is a lower bound on the steering advantage.
   CONTROL: frozen and live attacks follow different optimiser paths, so their costs
   differ slightly even when the final kept set is unchanged. The ratio is therefore
   split by whether the live selection at the adversarial point differs from the
   clean one (Jaccard < 1). Images with an unchanged selection give the optimiser
   noise floor; steering is the excess in images whose selection changed.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from . import stats as st

__all__ = ["rows_of", "analyze_attack", "format_attack", "spearman", "wilson"]

SPEARMAN_MIN = 0.5
MIN_N_SPEARMAN = 8
SLOPE_EQUIV = 0.10


# --------------------------------------------------------------------------- #
# small statistics
# --------------------------------------------------------------------------- #
def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    vals = a[order]
    i = 0
    while i < len(a):                       # average ranks over ties
        j = i
        while j + 1 < len(a) and vals[j + 1] == vals[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry / den) if den > 0 else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> Dict:
    if n == 0:
        return {"p": float("nan"), "lo": float("nan"), "hi": float("nan"), "k": 0, "n": 0}
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"p": p, "lo": max(0.0, c - w), "hi": min(1.0, c + w), "k": int(k), "n": int(n)}


def _boot_mean(v, B=2000, seed=0) -> Dict:
    return st.bootstrap_mean_ci(np.asarray(v, float), B=B, seed=seed)


def _excl0(d: Dict) -> bool:
    return bool(np.isfinite(d.get("lo", np.nan)) and (d["lo"] > 0 or d["hi"] < 0))


# --------------------------------------------------------------------------- #
# flatten records
# --------------------------------------------------------------------------- #
def rows_of(R: Dict) -> List[Dict]:
    """One row per (record, mode). cost_eff: ok -> cost; prebroken -> 0;
    failed / prefired -> inf (the attacker did not get a silent success)."""
    out = []
    for r in R["records"]:
        if "modes" not in r:
            continue
        for mode, mr in r["modes"].items():
            st_ = mr["status"]
            cost = mr.get("cost")
            eff = (float(cost) if st_ == "ok" else 0.0 if st_ == "prebroken" else math.inf)
            out.append({"image": int(r["image"]), "y": int(r["y"]), "method": r["method"],
                        "keep": float(r["keep"]), "ak": float(r["actual_keep"]),
                        "sel": r["selection"], "mode": mode, "status": st_,
                        "cost": (float(cost) if cost is not None else None), "cost_eff": eff,
                        "cost_rel": mr.get("cost_rel"),
                        "lin": (r.get("lin") or {}).get(mode),
                        "fires": mr.get("fires"), "fires_frozen": mr.get("fires_frozen"),
                        "jac": mr.get("kept_jaccard"), "margin0": r.get("margin0"),
                        "fires_q": mr.get("fires_q"), "converged": mr.get("converged"),
                        "iters": mr.get("iters_run"), "pgd_beat": mr.get("pgd_beat"),
                        "blocked": r.get("blocked_grad_frac")})
    return out


def _view(rows, method, sel, mode=None, y=None):
    """Rows of one method/selection plus the shared keep-1 baseline (method 'none')."""
    v = [r for r in rows if ((r["method"] == method and r["sel"] == sel) or
                             (r["method"] == "none" and r["keep"] >= 1.0))]
    if mode is not None:
        v = [r for r in v if r["mode"] == mode]
    if y is not None:
        v = [r for r in v if r["y"] == y]
    return v


# --------------------------------------------------------------------------- #
# 1. instrument
# --------------------------------------------------------------------------- #
def _instrument(rows, methods, modes, keeps) -> Dict:
    out = {}
    for m in methods:
        for mode in modes:
            v = [r for r in _view(rows, m, "frozen", mode)
                 if r["status"] == "ok" and r["lin"] is not None and r["lin"] > 0 and r["cost"] > 0]
            if not v:
                continue
            per = []
            for k in keeps:
                vk = [r for r in v if abs(r["keep"] - k) < 1e-12]
                if not vk:
                    continue
                lin = np.array([r["lin"] for r in vk]); tru = np.array([r["cost"] for r in vk])
                rho = spearman(lin, tru)
                per.append({"keep": k, "n": len(vk), "spearman": rho,
                            "ratio_median": float(np.median(tru / lin)),
                            "ratio_iqr": [float(np.percentile(tru / lin, 25)), float(np.percentile(tru / lin, 75))],
                            "pass": bool(len(vk) < MIN_N_SPEARMAN or (np.isfinite(rho) and rho >= SPEARMAN_MIN))})
            x = np.log([r["ak"] for r in v]); g = [r["image"] for r in v]
            yv = np.log([r["cost"] / r["lin"] for r in v])
            sl = st.paired_slope(x, yv, g)
            slope_pass = bool(np.isfinite(sl["lo"]) and -SLOPE_EQUIV <= sl["lo"] and sl["hi"] <= SLOPE_EQUIV)
            enough = bool(sl.get("n_groups", 0) >= MIN_N_SPEARMAN and
                          sum(p["n"] >= MIN_N_SPEARMAN for p in per) >= 2)
            ok = bool(all(p["pass"] for p in per) and slope_pass)
            out[f"{m}|{mode}"] = {"per_keep": per, "ratio_slope": sl, "slope_pass": slope_pass,
                                  "status": ("PASS" if ok else "FAIL") if enough else "INSUFFICIENT",
                                  "pass": ok and enough}
    return out


# --------------------------------------------------------------------------- #
# 2. costs
# --------------------------------------------------------------------------- #
def _cost_block(rows, m, sel, modes, keeps, budgets) -> Dict:
    res = {"per_keep": [], "slopes": {}, "by_direction": {}}
    for k in keeps:
        row = {"keep": k}
        for mode in modes:
            v = [r for r in _view(rows, m, sel, mode) if abs(r["keep"] - k) < 1e-12]
            if not v:
                continue
            eff = np.array([r["cost_eff"] for r in v])
            n = len(v)
            row[mode] = {"n": n,
                         "prebroken": float(np.mean([r["status"] == "prebroken" for r in v])),
                         "failed": float(np.mean([r["status"] == "failed" for r in v])),
                         "prefired": float(np.mean([r["status"] == "prefired" for r in v])),
                         "median": float(np.median(eff)),
                         "median_rel": float(np.median([r["cost_rel"] if r["cost_rel"] is not None
                                                        else (0.0 if r["status"] == "prebroken" else np.inf)
                                                        for r in v])),
                         "success_at": {f"{q}": float(np.mean(eff <= b)) for q, b in budgets.items()}}
            row["ak"] = float(np.median([r["ak"] for r in v]))
        for hm in ("thr", "thrq"):
            if {"free", hm} <= set(modes):
                T = _ratio_rows(rows, m, sel, k, hm)
                if T:
                    T = np.asarray(T)
                    row[f"T_{hm}"] = {"median": float(np.median(T)), "n": int(len(T)),
                                      # the tripwire BINDS (costs the attacker > 1 %) in this share
                                      "binding": float(np.mean(T > 1.01)),
                                      # true minima satisfy OC_thr >= OC_free; T < 1 = sub-optimal free attack
                                      "below_1": float(np.mean(T < 1 - 1e-3))}
        ok = [r for r in _view(rows, m, sel) if abs(r["keep"] - k) < 1e-12 and r["status"] == "ok"]
        if ok:
            row["converged"] = float(np.mean([bool(r["converged"]) for r in ok if r["converged"] is not None]
                                             or [np.nan]))
            pb = [bool(r["pgd_beat"]) for r in ok if r["pgd_beat"] is not None]
            row["pgd_beat"] = float(np.mean(pb)) if pb else float("nan")
        res["per_keep"].append(row)
    # paired slopes on finite positive costs
    for mode in modes:
        v = [r for r in _view(rows, m, sel, mode) if r["status"] == "ok" and r["cost"] > 0]
        if v:
            res["slopes"][f"log_{mode}"] = st.paired_slope(np.log([r["ak"] for r in v]),
                                                           np.log([r["cost"] for r in v]),
                                                           [r["image"] for r in v])
            for yv in (0, 1):
                vy = [r for r in v if r["y"] == yv]
                if vy:
                    res["by_direction"].setdefault(f"y={yv}", {})[f"log_{mode}"] = st.paired_slope(
                        np.log([r["ak"] for r in vy]), np.log([r["cost"] for r in vy]),
                        [r["image"] for r in vy])
    for hm in ("thr", "thrq"):
        if {"free", hm} <= set(modes):
            dec = _decomposition(rows, m, sel, hm)
            if dec:
                res[f"decomposition_{hm}"] = dec
    return res


def _pairs(rows, m, sel, k=None):
    """(image, keep) -> {mode: row} for ok rows with positive cost."""
    d = {}
    for r in _view(rows, m, sel):
        if k is not None and abs(r["keep"] - k) > 1e-12:
            continue
        if r["status"] == "ok" and r["cost"] > 0:
            d.setdefault((r["image"], r["keep"]), {})[r["mode"]] = r
    return d


def _ratio_rows(rows, m, sel, k, hm="thr") -> List[float]:
    return [p[hm]["cost"] / p["free"]["cost"] for p in _pairs(rows, m, sel, k).values()
            if hm in p and "free" in p]


def _decomposition(rows, m, sel, hm="thr") -> Optional[Dict]:
    P = [p for p in _pairs(rows, m, sel).values() if hm in p and "free" in p]
    if len(P) < 3:
        return None
    x = np.log([p["free"]["ak"] for p in P]); g = [p["free"]["image"] for p in P]
    lf = np.log([p["free"]["cost"] for p in P]); lt = np.log([p[hm]["cost"] for p in P])
    return st.decompose(x, {"log_oc": lt, "log_free": lf, "log_T": lt - lf}, g)


def _budgets(rows) -> Dict:
    base = [r["cost"] for r in rows if r["method"] == "none" and r["mode"] == "free"
            and r["status"] == "ok" and r["cost"] > 0]
    if not base:
        return {}
    return {f"q{q}": float(np.percentile(base, q)) for q in (25, 50, 75)}


# --------------------------------------------------------------------------- #
# 3. incidental detection
# --------------------------------------------------------------------------- #
def _incidental(rows, m, sel, keeps) -> Dict:
    v = [r for r in _view(rows, m, sel, "free") if r["status"] == "ok"]
    per = []
    for k in keeps:
        vk = [r for r in v if abs(r["keep"] - k) < 1e-12]
        if not vk:
            continue
        per.append({"keep": k,
                    "refit": wilson(sum(bool(r["fires"]) for r in vk), len(vk)),
                    "refit_q": wilson(sum(bool(r["fires_q"]) for r in vk if r["fires_q"] is not None),
                                      sum(r["fires_q"] is not None for r in vk)),
                    "frozen": wilson(sum(bool(r["fires_frozen"]) for r in vk), len(vk))})
    out = {"per_keep": per}
    if len(keeps) > 1:
        k1, kmin = max(keeps), min(keeps)
        a = {r["image"]: r for r in v if abs(r["keep"] - k1) < 1e-12}
        b = {r["image"]: r for r in v if abs(r["keep"] - kmin) < 1e-12}
        common = sorted(set(a) & set(b))
        if common:
            out["paired_diff_refit"] = _boot_mean([float(b[i]["fires"]) - float(a[i]["fires"]) for i in common])
            out["paired_diff_frozen"] = _boot_mean([float(b[i]["fires_frozen"]) - float(a[i]["fires_frozen"])
                                                    for i in common])
    return out


# --------------------------------------------------------------------------- #
# 4. steering (R2)
# --------------------------------------------------------------------------- #
def _steering(rows, m, modes, keeps, budgets) -> Dict:
    out = {}
    for k in keeps:
        if k >= 1.0:
            continue
        blk = {}
        logs = {}
        for mode in modes:
            fr = {r["image"]: r for r in rows if r["method"] == m and r["sel"] == "frozen"
                  and r["mode"] == mode and abs(r["keep"] - k) < 1e-12}
            lv = {r["image"]: r for r in rows if r["method"] == m and r["sel"] == "live"
                  and r["mode"] == mode and abs(r["keep"] - k) < 1e-12}
            common = sorted(set(fr) & set(lv))
            if not common:
                continue
            both = [i for i in common if fr[i]["status"] == "ok" and lv[i]["status"] == "ok"
                    and fr[i]["cost"] > 0 and lv[i]["cost"] > 0]
            lr = {i: math.log(fr[i]["cost"] / lv[i]["cost"]) for i in both}
            logs[mode] = lr
            changed = [i for i in both if lv[i]["jac"] is not None and lv[i]["jac"] < 1 - 1e-9]
            same = [i for i in both if i not in changed]
            blk[mode] = {"log_ratio": _boot_mean(list(lr.values())),
                         "log_ratio_selection_changed": _boot_mean([lr[i] for i in changed]),
                         "log_ratio_selection_same": _boot_mean([lr[i] for i in same]),
                         "frac_selection_changed": (len(changed) / len(both)) if both else float("nan"),
                         "n_pairs": len(both), "n_common": len(common),
                         "success_at": {q: {"frozen": float(np.mean([fr[i]["cost_eff"] <= b for i in common])),
                                            "live": float(np.mean([lv[i]["cost_eff"] <= b for i in common]))}
                                        for q, b in budgets.items()},
                         "jaccard_live_median": float(np.median([lv[i]["jac"] for i in common
                                                                 if lv[i]["jac"] is not None]))
                         if any(lv[i]["jac"] is not None for i in common) else float("nan")}
        for hm in ("thr", "thrq"):
            if hm in logs and "free" in logs:
                com = sorted(set(logs[hm]) & set(logs["free"]))
                blk[f"R2_index_{hm}"] = _boot_mean([logs[hm][i] - logs["free"][i] for i in com])
        if blk:
            out[f"{k}"] = blk
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def analyze_attack(R: Dict) -> Dict:
    rows = rows_of(R)
    keeps = sorted({r["keep"] for r in rows}, reverse=True)
    methods = [m for m in R["methods"] if any(r["method"] == m for r in rows)]
    modes = [m for m in R["modes"] if any(r["mode"] == m for r in rows)]
    budgets = _budgets(rows)
    S = {"leaf": R["leaf"], "keeps": keeps, "methods": methods, "modes": modes, "budgets": budgets,
         "n_records": len([r for r in R["records"] if "modes" in r]),
         "n_errors": len([r for r in R["records"] if "error" in r]),
         "complete": R.get("complete", False)}
    bl = [r["blocked"] for r in rows if r["mode"] == "free" and r["blocked"] is not None]
    S["diagnostics"] = {
        "converged": float(np.mean([bool(r["converged"]) for r in rows
                                    if r["status"] == "ok" and r["converged"] is not None] or [np.nan])),
        "failed": float(np.mean([r["status"] == "failed" for r in rows])),
        "pgd_beat": float(np.mean([bool(r["pgd_beat"]) for r in rows if r["pgd_beat"] is not None] or [np.nan])),
        "blocked_grad_frac_median": float(np.median(bl)) if bl else float("nan")}
    S["instrument"] = _instrument(rows, methods, modes, keeps)
    S["instrument_pass"] = {}
    for mode in modes:
        st_ = [v["status"] for k_, v in S["instrument"].items() if k_.endswith("|" + mode)]
        if st_:
            S["instrument_pass"][mode] = ("FAIL" if "FAIL" in st_ else
                                          "PASS" if all(x == "PASS" for x in st_) else "INSUFFICIENT")
    S["costs"], S["incidental"], S["steering"] = {}, {}, {}
    for m in methods:
        for sel in ("frozen", "live"):
            if not any(r["method"] == m and r["sel"] == sel for r in rows):
                continue
            S["costs"][f"{m}|{sel}"] = _cost_block(rows, m, sel, modes, keeps, budgets)
            S["incidental"][f"{m}|{sel}"] = _incidental(rows, m, sel, keeps)
        if any(r["method"] == m and r["sel"] == "live" for r in rows):
            S["steering"][m] = _steering(rows, m, modes, keeps, budgets)
    return S


def _fs(d: Dict) -> str:
    if not d or not np.isfinite(d.get("slope", np.nan)):
        return "   n/a        "
    star = "*" if _excl0(d) else " "
    return f"{d['slope']:+.2f}{star}[{d['lo']:+.2f},{d['hi']:+.2f}]"


def _fm(d: Dict) -> str:
    if not d or not np.isfinite(d.get("mean", np.nan)):
        return "n/a"
    star = "*" if _excl0(d) else " "
    return f"{d['mean']:+.3f}{star}[{d['lo']:+.3f},{d['hi']:+.3f}] (n={d['n']})"


def format_attack(S: Dict) -> str:
    L = [f"ATTACK v3 summary  leaf={S['leaf']}  records={S['n_records']}  errors={S['n_errors']}  "
         "complete=" + str(S["complete"]),
         "  beta = d log X / d log(kept fraction); beta>0: X FALLS under compression; * = 95% CI excludes 0",
         "  budgets (keep-1 OC_free quantiles): " + ", ".join(f"{k}={v:.3g}" for k, v in S["budgets"].items())]
    dg = S["diagnostics"]
    L.append(f"  OPTIMISER: converged {dg['converged']:.2f} of successes | failed {dg['failed']:.2f} of attacks | "
             f"PGD beat min-norm in {dg['pgd_beat']:.2f} of free attacks | median blocked-gradient share "
             f"{dg['blocked_grad_frac_median']:.2f}  (trust slopes only if converged ~1, failed ~0, pgd_beat ~0)")
    L.append("\n[1] INSTRUMENT (frozen selection): Spearman(lin, true) per keep; slope of log(true/lin)")
    for key, v in S["instrument"].items():
        per = "  ".join(f"k={p['keep']:.3g}:rho={p['spearman']:.2f},ratio={p['ratio_median']:.2f}(n={p['n']})"
                        for p in v["per_keep"])
        L.append(f"  {key:22s} {v['status']:12s} {per}  slope {_fs(v['ratio_slope'])}")
    L.append("  overall (FAIL if any cell fails; INSUFFICIENT if any cell lacks data): "
             + ", ".join(f"{k}={v}" for k, v in S["instrument_pass"].items()))
    L.append("\n[2] COSTS in the fixed space (median incl. prebroken=0 / failed=inf; slopes on finite costs)")
    for key, c in S["costs"].items():
        L.append(f"  [{key}]")
        for row in c["per_keep"]:
            parts = [f"k={row['keep']:.4g} ak={row.get('ak', float('nan')):.3f}"]
            for mode in S["modes"]:
                if mode in row:
                    q = row[mode]
                    parts.append(f"{mode}: med={q['median']:.3g} pre={q['prebroken']:.2f} fail={q['failed']:.2f}"
                                 + (f" fired={q['prefired']:.2f}" if mode in ("thr", "thrq") else "")
                                 + " P(<=q50)=" + f"{q['success_at'].get('q50', float('nan')):.2f}")
            for hm in ("thr", "thrq"):
                if f"T_{hm}" in row:
                    q = row[f"T_{hm}"]
                    parts.append(f"T_{hm}={q['median']:.3f}(n={q['n']}, binds:{q['binding']:.2f}, T<1:{q['below_1']:.2f})")
            if "converged" in row:
                parts.append(f"conv={row['converged']:.2f} pgd_beat={row['pgd_beat']:.2f}")
            L.append("    " + " | ".join(parts))
        sl = c["slopes"]
        L.append("    slopes: " + "  ".join(f"{k}={_fs(v)}" for k, v in sl.items()))
        for hm in ("thr", "thrq"):
            if f"decomposition_{hm}" in c:
                d = c[f"decomposition_{hm}"]
                L.append(f"    exact split: log OC_{hm} {_fs(d['log_oc'])} = log OC_free {_fs(d['log_free'])} "
                         f"+ log T_{hm} {_fs(d['log_T'])}  (recon err {d['recon_error']:.1e})")
        for dkey, dv in c["by_direction"].items():
            L.append(f"    {dkey}: " + "  ".join(f"{k}={_fs(v)}" for k, v in dv.items()))
    L.append("\n[3] INCIDENTAL DETECTION: P(tripwire fires | plain attack succeeded)")
    for key, inc in S["incidental"].items():
        per = "  ".join(f"k={p['keep']:.3g}: refit {p['refit']['p']:.2f}[{p['refit']['lo']:.2f},{p['refit']['hi']:.2f}]"
                        f"(n={p['refit']['n']}) calibrated {p['refit_q']['p']:.2f} frozen {p['frozen']['p']:.2f}"
                        for p in inc["per_keep"])
        L.append(f"  [{key}] {per}")
        if "paired_diff_refit" in inc:
            L.append(f"      paired (k_min - k_1): refit {_fm(inc['paired_diff_refit'])}  "
                     f"frozen {_fm(inc['paired_diff_frozen'])}")
    if S["steering"]:
        L.append("\n[4] STEERING (R2): log(OC_frozen / OC_live) > 0 => live selection makes the attack cheaper")
        for m, blk in S["steering"].items():
            for k, b in blk.items():
                parts = [f"{mode}: {_fm(b[mode]['log_ratio'])} jac={b[mode]['jaccard_live_median']:.2f}"
                         for mode in S["modes"] if mode in b]
                L.append(f"  [{m} k={k}] " + " | ".join(parts))
                for mode in S["modes"]:
                    if mode in b:
                        L.append(f"      {mode}: selection changed in {b[mode]['frac_selection_changed']:.2f} of pairs;"
                                 f" changed {_fm(b[mode]['log_ratio_selection_changed'])}"
                                 f" vs unchanged (noise floor) {_fm(b[mode]['log_ratio_selection_same'])}")
                for hm in ("thr", "thrq"):
                    if f"R2_index_{hm}" in b:
                        L.append(f"      R2 index {hm} (steering helps the HIDDEN attack beyond the plain one): "
                                 f"{_fm(b[f'R2_index_{hm}'])}")
    return "\n".join(L)
