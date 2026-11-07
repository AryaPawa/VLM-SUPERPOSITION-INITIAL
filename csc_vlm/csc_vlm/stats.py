"""
csc_vlm.stats  --  paired slope estimation, cluster bootstrap, exact decomposition
==================================================================================

The p2 verdicts took a least-squares slope through ~6 cell MEDIANS. That has two
problems:
  (i)  no uncertainty at all;
  (ii) the decomposition  log OC = log m - log||g|| - log Gamma  is exact per image
       but NOT on medians (median(OC) != median(m) / (median(g) * median(Gamma))),
       so the reported "share" did not add up.

Here every quantity is per image. The design is paired (the same images at every
keep), so we estimate the within-image (image fixed-effect) slope of y on x = log
keep. That is the OLS slope after demeaning x and y inside each image, and it is
the standard random-intercept estimator for a balanced design. Because the slope
is linear in y, the slopes of log m, -log||g|| and log T add up EXACTLY to the
slope of log OC on the same rows. Uncertainty comes from a cluster bootstrap over
images.

Sign convention (matches the proof plan): beta = d log X / d log keep.
beta > 0  =>  X FALLS as compression increases (keep decreases).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["within_slope", "paired_slope", "paired_slope_diff", "decompose",
           "bootstrap_mean_ci", "auroc", "auroc_ci", "describe_slope"]


def _group_stats(x, y, groups):
    """Per-group sufficient statistics (Sxx_g, Sxy_g) after within-group demeaning."""
    x = np.asarray(x, float); y = np.asarray(y, float); groups = np.asarray(groups)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[ok], y[ok], groups[ok]
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    cnt = np.bincount(inv, minlength=G).astype(float)
    mx = np.bincount(inv, weights=x, minlength=G) / np.maximum(cnt, 1)
    my = np.bincount(inv, weights=y, minlength=G) / np.maximum(cnt, 1)
    dx = x - mx[inv]; dy = y - my[inv]
    sxx = np.bincount(inv, weights=dx * dx, minlength=G)
    sxy = np.bincount(inv, weights=dx * dy, minlength=G)
    return uniq, sxx, sxy


def within_slope(x, y, groups) -> float:
    _, sxx, sxy = _group_stats(x, y, groups)
    den = sxx.sum()
    return float(sxy.sum() / den) if den > 0 else float("nan")


def paired_slope(x, y, groups, B: int = 2000, seed: int = 0, level: float = 0.95) -> Dict:
    """Within-image slope of y on x with a cluster-bootstrap percentile CI."""
    uniq, sxx, sxy = _group_stats(x, y, groups)
    G = len(uniq)
    den = sxx.sum()
    est = float(sxy.sum() / den) if den > 0 else float("nan")
    if G < 3 or den <= 0:
        return {"slope": est, "lo": float("nan"), "hi": float("nan"), "n_groups": int(G)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(B, G))
    bx = sxx[idx].sum(1); by = sxy[idx].sum(1)
    bs = np.where(bx > 0, by / np.where(bx > 0, bx, 1.0), np.nan)
    a = (1 - level) / 2
    lo, hi = np.nanquantile(bs, [a, 1 - a])
    return {"slope": est, "lo": float(lo), "hi": float(hi), "n_groups": int(G)}


def paired_slope_diff(x, y1, y2, groups, **kw) -> Dict:
    """Slope of (y1 - y2): e.g. real log-alignment minus placebo, paired by image."""
    return paired_slope(x, np.asarray(y1, float) - np.asarray(y2, float), groups, **kw)


def decompose(x, terms: Dict[str, Sequence[float]], groups, total_key: str = "log_oc",
              **kw) -> Dict:
    """Paired slopes for each additive term plus the total. When the terms sum to
    the total per row, the point slopes sum exactly (``recon_error`` ~ 0). Each
    term's signed contribution is reported (no absolute values, so cancellation
    stays visible)."""
    out = {k: paired_slope(x, v, groups, **kw) for k, v in terms.items()}
    parts = [k for k in terms if k != total_key]
    if total_key in out:
        recon = sum(out[k]["slope"] for k in parts)
        out["recon_error"] = float(out[total_key]["slope"] - recon)
        tot = out[total_key]["slope"]
        out["contribution"] = {k: (out[k]["slope"] / tot if np.isfinite(tot) and abs(tot) > 1e-12
                                   else float("nan")) for k in parts}
    return out


def bootstrap_mean_ci(v, B: int = 2000, seed: int = 0, level: float = 0.95) -> Dict:
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(B, len(v)))].mean(1)
    a = (1 - level) / 2
    return {"mean": float(v.mean()), "lo": float(np.quantile(bs, a)),
            "hi": float(np.quantile(bs, 1 - a)), "n": int(len(v))}


def auroc(scores, y) -> float:
    """Rank-based AUROC (Mann-Whitney), tie-aware."""
    s = np.asarray(scores, float); y = np.asarray(y)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    ranks = ((csum - cnt) + csum + 1) / 2.0
    r = ranks[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_ci(scores, y, B: int = 1000, seed: int = 0, level: float = 0.95) -> Dict:
    s = np.asarray(scores, float); y = np.asarray(y)
    est = auroc(s, y)
    rng = np.random.default_rng(seed)
    n = len(s)
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        bs.append(auroc(s[i], y[i]))
    if not bs:
        return {"auroc": est, "lo": float("nan"), "hi": float("nan")}
    a = (1 - level) / 2
    return {"auroc": est, "lo": float(np.quantile(bs, a)), "hi": float(np.quantile(bs, 1 - a))}


def describe_slope(name: str, d: Dict, meaning_pos: str, meaning_neg: str) -> str:
    """One-line, two-sided reading of a paired slope."""
    s, lo, hi = d["slope"], d["lo"], d["hi"]
    if not np.isfinite(lo):
        verdict = "undetermined (too few images)"
    elif lo > 0:
        verdict = meaning_pos
    elif hi < 0:
        verdict = meaning_neg
    else:
        verdict = "no detectable change (CI spans 0)"
    return f"{name:<28s} beta={s:+.3f} [{lo:+.3f},{hi:+.3f}]  -> {verdict}"
