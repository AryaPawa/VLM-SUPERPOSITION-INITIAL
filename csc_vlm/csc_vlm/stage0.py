"""
csc_vlm.stage0  --  Stage 0 v2: forward-only VLM sanity (no attack, no gradients)
=================================================================================

Stage 0 answers the VLM-specific questions that must be settled before any
evasion number means anything:

  R1   Does the concept survive compression? Accuracy and the log-margin slope,
       per method, with CIs.
  H2b  Merge-family vs prune-family at matched carriers: ``pool`` keeps
       information in fewer carriers; ``drop_*`` deletes it.
  G1   Random drop (``drop_random``) vs norm drop: "how many" vs "which".
  R2   Query-conditioned selection (``drop_query``) vs norm selection.
  X3   ``merge_dup`` (legacy): N stays fixed while input information falls.
  Read artifact  ``readout_only``: full forward, span read restricted to top-norm
       tokens. A d_eff collapse here means the collapse comes from the read.
  Staleness  Frozen (keep=1) vs refit linear probe at the last-token read.
  Coverage (Problem D)  How many read-space PCs explain 90% of the monitor-train
       reads at each keep. If ~r=8 already covers most of it, a fixed-rank
       monitor "sees more" of a compressed model just from geometry.
  Geometry (Problem C)  Block-level participation ratio of the carriers at the
       read layer (with the top-1% norm tokens removed too), next to the
       read-position d_eff.

Every per-image number is kept, so slopes come with cluster-bootstrap CIs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import stats as st

__all__ = ["BehaviourSpec", "BEHAVIOURS", "behaviour_for", "fit_linear_probe", "dprime",
           "d_eff", "participation_ratio", "coverage_curve", "keep_metrics", "run_stage0",
           "summarize_stage0", "Stage0Config", "Stage0VLM", "Stage0Mock", "DEFAULT_KEEPS",
           "METHODS", "auroc"]

EPS = 1e-12
DEFAULT_KEEPS: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
METHODS: Tuple[str, ...] = ("pool", "drop_norm", "drop_random", "drop_query", "merge_dup")
auroc = st.auroc


# --------------------------------------------------------------------------- #
# behaviours: the yes/no task that gives the model a real answer
# --------------------------------------------------------------------------- #
@dataclass
class BehaviourSpec:
    name: str
    prompt: str
    pos_words: List[str]          # answers for label == 1
    neg_words: List[str]          # answers for label == 0


BEHAVIOURS: Dict[str, BehaviourSpec] = {
    # hf:microsoft/cats_vs_dogs binarises to label 1 = DOG (data.py maps the first
    # HF label, cat=0, to 0). The p2 spec had pos="cat" and only "worked" through
    # the polarity flip.
    "cats_vs_dogs": BehaviourSpec("cats_vs_dogs", "Is this a cat or a dog? Answer with one word.",
                                  ["dog"], ["cat"]),
    "coco_person": BehaviourSpec("coco_person", "Is there a person in this image? Answer yes or no.",
                                 ["yes"], ["no"]),
}


def behaviour_for(dataset_spec: str) -> BehaviourSpec:
    s = (dataset_spec or "").lower()
    if s.startswith("coco:"):
        cat = s.split(":")[1].replace("_", " ")
        return BehaviourSpec(f"coco_{cat.replace(' ', '_')}",
                             f"Is there a {cat} in this image? Answer yes or no.", ["yes"], ["no"])
    if "person" in s:
        return BEHAVIOURS["coco_person"]
    return BEHAVIOURS["cats_vs_dogs"]


# --------------------------------------------------------------------------- #
# pure-numpy metrics
# --------------------------------------------------------------------------- #
def fit_linear_probe(H: np.ndarray, y: np.ndarray, lam: float = 0.1) -> Tuple[np.ndarray, float]:
    """Ridge probe to +/-1 targets, solved in the dual (n x n) so it is cheap when
    n << d. ``lam`` is RELATIVE to the mean squared read norm (scale-free across
    models/layers). Returns (w, b); the bias is unpenalised (we centre first)."""
    H = np.asarray(H, float); y = np.asarray(y, float)
    mu = H.mean(0)
    Hc = H - mu
    t = 2.0 * y - 1.0
    tm = t.mean()
    K = Hc @ Hc.T
    lam_abs = lam * float(np.trace(K)) / max(1, len(H)) + 1e-12
    alpha = np.linalg.solve(K + lam_abs * np.eye(len(H)), t - tm)
    w = Hc.T @ alpha
    b = float(tm - mu @ w)
    return w, b


def dprime(H, y, direction) -> float:
    s = np.asarray(H, float) @ direction
    s1, s0 = s[y == 1], s[y == 0]
    if len(s1) < 2 or len(s0) < 2:
        return 0.0
    pooled = math.sqrt(0.5 * (s1.var(ddof=1) + s0.var(ddof=1))) + EPS
    return float(abs(s1.mean() - s0.mean()) / pooled)


def participation_ratio(X: np.ndarray) -> float:
    X = np.asarray(X, float)
    if X.shape[0] < 2:
        return 1.0
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    tot = ev.sum()
    return float(tot ** 2 / (ev ** 2).sum()) if tot > 0 else 1.0


d_eff = participation_ratio


def coverage_curve(H: np.ndarray, ranks=(1, 2, 4, 8, 16, 32)) -> Dict:
    """Explained-variance fractions of the top-r PCs and r90 (PCs for 90%)."""
    H = np.asarray(H, float)
    Hc = H - H.mean(0, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    ev = s ** 2
    frac = np.cumsum(ev) / max(ev.sum(), EPS)
    out = {f"top{r}": float(frac[min(r, len(frac)) - 1]) for r in ranks}
    out["r90"] = int(np.searchsorted(frac, 0.90) + 1)
    out["r50"] = int(np.searchsorted(frac, 0.50) + 1)
    return out


def _probe_block(eval_H, eval_y, mon_H, mon_y, mon_H0, mon_y0, lam, B, seed):
    w_r, b_r = fit_linear_probe(mon_H, mon_y, lam)
    w_f, b_f = fit_linear_probe(mon_H0, mon_y0, lam)
    ar = st.auroc_ci(eval_H @ w_r + b_r, eval_y, B=B, seed=seed)
    af = st.auroc_ci(eval_H @ w_f + b_f, eval_y, B=B, seed=seed + 1)
    return {"auroc_refit": ar["auroc"], "auroc_refit_lo": ar["lo"], "auroc_refit_hi": ar["hi"],
            "auroc_frozen": af["auroc"], "auroc_frozen_lo": af["lo"], "auroc_frozen_hi": af["hi"],
            "dprime": dprime(eval_H, eval_y, w_r / (np.linalg.norm(w_r) + EPS)),
            "d_eff": participation_ratio(eval_H)}


def keep_metrics(ev: Dict, mon: Dict, mon0: Dict, lam: float = 0.1, B: int = 500,
                 seed: int = 0) -> Dict:
    ok = ev["ok"]
    acc = st.bootstrap_mean_ci(ev["correct"][ok], B=B, seed=seed)
    m = {"n_eval": int(ok.sum()), "accuracy": acc["mean"], "accuracy_lo": acc["lo"],
         "accuracy_hi": acc["hi"],
         "margin_median": float(np.median(ev["margin"][ok])) if ok.any() else float("nan"),
         "abs_margin_median": float(np.median(np.abs(ev["margin"][ok]))) if ok.any() else float("nan"),
         "n_carriers": float(np.mean(ev["n_carriers"][ok])) if ok.any() else float("nan"),
         "actual_keep": float(np.mean(ev["actual_keep"][ok])) if ok.any() else float("nan"),
         "seq_len": float(np.mean(ev["seq_len"][ok])) if ok.any() else float("nan"),
         "blk_pr": float(np.nanmean(ev["blk_pr"][ok])) if ok.any() else float("nan"),
         "blk_pr_trim": float(np.nanmean(ev["blk_pr_trim"][ok])) if ok.any() else float("nan")}
    mok, m0ok = mon["ok"], mon0["ok"]
    for read in ("last", "span"):
        r = _probe_block(ev[read][ok], ev["y"][ok], mon[read][mok], mon["y"][mok],
                         mon0[read][m0ok], mon0["y"][m0ok], lam, B, seed)
        for k, v in r.items():
            m[f"{read}_{k}"] = v
    cov = coverage_curve(mon["last"][mok])
    for k, v in cov.items():
        m[f"cov_{k}"] = v
    return m


# --------------------------------------------------------------------------- #
# orchestration (collector-agnostic)
# --------------------------------------------------------------------------- #
def run_stage0(collector, keeps=DEFAULT_KEEPS, methods=METHODS, include_readout_only=True,
               lam: float = 0.1, B: int = 500, verbose: bool = True) -> Dict:
    """collector.collect(split, method, keep) -> pack with keys
        last[n,D], span[n,D], margin[n], correct[n], y[n], ok[n], ids[n],
        n_carriers[n], actual_keep[n], seq_len[n], blk_pr[n], blk_pr_trim[n]
    for split in {"eval", "monitor"}."""
    keeps = [float(k) for k in keeps]
    if 1.0 not in keeps:
        keeps = [1.0] + keeps
    keeps = sorted(set(keeps), reverse=True)
    R: Dict = {"keeps": keeps, "methods": [], "config": collector.describe(), "per_image": {}}
    mon0 = collector.collect("monitor", "none", 1.0)
    ev0 = collector.collect("eval", "none", 1.0)
    all_methods = list(methods) + (["readout_only"] if include_readout_only else [])
    for method in all_methods:
        rows, per = [], []
        for keep in keeps:
            if keep >= 1.0:
                ev, mon = ev0, mon0
            else:
                ev = collector.collect("eval", method, keep)
                mon = collector.collect("monitor", method, keep)
            row = {"keep": keep}
            row.update(keep_metrics(ev, mon, mon0, lam=lam, B=B))
            rows.append(row)
            per.append({"keep": keep, "ids": ev["ids"].tolist(), "margin": ev["margin"].tolist(),
                        "actual_keep": ev["actual_keep"].tolist(),
                        "correct": ev["correct"].tolist(), "y": ev["y"].tolist(),
                        "ok": ev["ok"].tolist()})
            if verbose:
                print(f"[stage0] {method:12s} keep={keep:.4f} acc={row['accuracy']:.3f} "
                      f"lastAUROC(refit)={row['last_auroc_refit']:.3f} "
                      f"carriers={row['n_carriers']:.0f} r90={row['cov_r90']}", flush=True)
        R[method] = rows
        R["per_image"][method] = per
        R["methods"].append(method)
    R["summary"] = summarize_stage0(R, B=max(B, 1000))
    return R


def _paired_logmargin_slope(per: List[Dict], B: int) -> Dict:
    x, y, g = [], [], []
    for cell in per:
        ak = cell.get("actual_keep", [cell["keep"]] * len(cell["ids"]))
        for i, mg, ok, k in zip(cell["ids"], cell["margin"], cell["ok"], ak):
            if ok and np.isfinite(mg) and np.isfinite(k) and k > 0:
                x.append(math.log(k)); y.append(math.log(abs(mg) + 1e-6)); g.append(i)
    return st.paired_slope(np.array(x), np.array(y), np.array(g), B=B)


def summarize_stage0(R: Dict, B: int = 1000) -> Dict:
    S: Dict = {"logmargin_slope": {}, "notes": []}
    for m in R["methods"]:
        S["logmargin_slope"][m] = _paired_logmargin_slope(R["per_image"][m], B)
    k0 = R[R["methods"][0]][0]
    if k0.get("last_auroc_refit_lo", 0) > 0.98:
        S["notes"].append("last-token probe AUROC is saturated (>0.98) at keep=1: AUROC cannot show "
                          "degradation for this concept; use a harder concept (e.g. --dataset coco:<cat>).")
    if k0.get("accuracy", 1) < 0.6:
        S["notes"].append("uncompressed behaviour accuracy < 0.6: the answer tokens/prompt are not "
                          "measuring the concept for this model; fix before Stage 1.")
    if k0.get("cov_top8", 0) > 0.8:
        S["notes"].append(f"top-8 read PCs already explain {k0['cov_top8']:.0%} of read variance at keep=1: "
                          "a rank-8 monitor is in the high-coverage regime (Problem D). Read Stage 1 "
                          "Gamma(r) curves, not one rank.")
    return S


# --------------------------------------------------------------------------- #
# VLM collector
# --------------------------------------------------------------------------- #
@dataclass
class Stage0Config:
    model: str = "qwen-3b"
    model_id: str = ""
    dataset: str = "hf:microsoft/cats_vs_dogs"
    read_layer_frac: float = 0.70
    query_layer: int = 2
    device: str = "cuda"
    dtype: str = "auto"
    load_4bit: bool = False
    n_eval: int = 256
    n_monitor: int = 144
    seed: int = 0


def _split_indices(N: int, n_eval: int, n_mon: int, seed: int):
    if n_eval + n_mon > N:
        frac = n_mon / max(1, n_eval + n_mon)
        n_mon = max(2, min(n_mon, int(round(frac * N)), N - 2))
        n_eval = max(2, N - n_mon)
        print(f"[stage0][warn] only {N} images: using n_eval={n_eval}, n_monitor={n_mon}")
    perm = np.random.default_rng(seed).permutation(N)
    return perm[:n_eval], perm[n_eval:n_eval + n_mon]


class Stage0VLM:
    """Forward-only collector over a real VLM via VLMHandle (pipeline v2)."""

    def __init__(self, cfg: Stage0Config, images: List, labels: np.ndarray, handle=None,
                 pos_ids: Optional[List[int]] = None, neg_ids: Optional[List[int]] = None):
        from .vlm import VLMHandle
        self.cfg = cfg
        self.behaviour = behaviour_for(cfg.dataset)
        self.images = images
        self.labels = np.asarray(labels).astype(int)
        self.h = handle if handle is not None else VLMHandle.from_pretrained(
            cfg.model, model_id=cfg.model_id, dtype=cfg.dtype, device=cfg.device,
            load_4bit=cfg.load_4bit, read_layer_frac=cfg.read_layer_frac,
            query_layer=cfg.query_layer)
        self.pos_ids = pos_ids if pos_ids is not None else self.h.answer_ids(self.behaviour.pos_words)
        self.neg_ids = neg_ids if neg_ids is not None else self.h.answer_ids(self.behaviour.neg_words)
        overlap = set(self.pos_ids) & set(self.neg_ids)
        self.pos_ids = [i for i in self.pos_ids if i not in overlap]
        self.neg_ids = [i for i in self.neg_ids if i not in overlap]
        if not self.pos_ids or not self.neg_ids:
            raise ValueError(f"answer token sets are empty after removing overlap {overlap}; "
                             f"change the answer words in BEHAVIOURS for this tokenizer")
        self.eval_idx, self.mon_idx = _split_indices(len(images), cfg.n_eval, cfg.n_monitor, cfg.seed)
        self._inp_cache: Dict[int, Dict] = {}
        self._score_cache: Dict[int, np.ndarray] = {}
        self.polarity_flipped = False

    def describe(self) -> Dict:
        d = self.h.describe()
        d.update({"kind": "vlm-stage0-v2", "dataset": self.cfg.dataset,
                  "behaviour": self.behaviour.name, "prompt": self.behaviour.prompt,
                  "pos_ids": self.pos_ids, "neg_ids": self.neg_ids,
                  "n_eval": int(len(self.eval_idx)), "n_monitor": int(len(self.mon_idx)),
                  "polarity_flipped": self.polarity_flipped})
        return d

    # ---- caching ----
    def inputs(self, i: int) -> Dict:
        import torch
        if i not in self._inp_cache:
            inp = self.h.prepare(self.images[int(i)], self.behaviour.prompt)
            self._inp_cache[i] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inp.items()}
        return {k: (v.to(self.h.device) if torch.is_tensor(v) else v)
                for k, v in self._inp_cache[i].items()}

    def scores(self, i: int, inp: Dict) -> np.ndarray:
        if i not in self._score_cache:
            self._score_cache[i] = self.h.query_scores(inp)
        return self._score_cache[i]

    def margin_of(self, logits) -> float:
        import torch
        lp = torch.logsumexp(logits[self.pos_ids], 0)
        ln = torch.logsumexp(logits[self.neg_ids], 0)
        return float(lp - ln)

    # ---- one image ----
    def one(self, i: int, method: str, keep: float) -> Dict:
        import torch
        inp = self.inputs(i)
        rl = self.h.read_layer
        mth = "none" if keep >= 1.0 else method
        sc = self.scores(i, inp) if (mth == "drop_query") else None
        o = self.h.compressed_run(inp, mth, keep, scores=sc, image_key=int(i),
                                  seed=self.cfg.seed, capture=(rl,))
        hs = o["hs"][rl].float()
        plan = o["plan"]
        if plan is None:                       # uncompressed
            span = [p for r in o["runs"] for p in r.positions]
            rows = torch.as_tensor(span, device=hs.device)
            n_car, n_vis = len(span), len(span)
        else:
            rows = plan.read_restrict if plan.read_restrict is not None else plan.carrier_rows
            n_car, n_vis = plan.n_carriers, plan.n_visual
            if plan.read_restrict is not None:          # readout_only: the READ is what shrinks
                n_car = int(plan.read_restrict.numel())
        # multi-GPU (device_map="auto"): plan tensors live on the EMBEDDING device, while the
        # read layer's hidden states live on that layer's device -> index on hs's device
        blk = hs[rows.to(hs.device)].cpu().numpy().astype(np.float64)
        last = hs[-1].cpu().numpy().astype(np.float64)
        pr = participation_ratio(blk)
        if blk.shape[0] >= 10:
            nrm = np.linalg.norm(blk, axis=1)
            cut = max(1, int(round(0.01 * blk.shape[0])))
            pr_t = participation_ratio(blk[np.argsort(nrm)[:-cut]])
        else:
            pr_t = pr
        mg = self.margin_of(o["logits"])
        if self.polarity_flipped:
            mg = -mg
        y = int(self.labels[int(i)])
        return {"last": last, "span": blk.mean(0), "margin": mg, "correct": int((mg > 0) == (y == 1)),
                "y": y, "n_carriers": n_car, "actual_keep": n_car / max(1, n_vis),
                "seq_len": o["new_len"], "blk_pr": pr, "blk_pr_trim": pr_t}

    def collect(self, split: str, method: str, keep: float) -> Dict:
        idx = self.eval_idx if split == "eval" else self.mon_idx
        recs, oks, errs = [], [], 0
        for c, i in enumerate(idx):
            try:
                recs.append(self.one(int(i), method, keep)); oks.append(True)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # noqa: BLE001  (one bad image must not kill the run)
                errs += 1
                if errs == c + 1 and errs >= 3:
                    raise RuntimeError(f"first {errs} images all failed for {method}@{keep}: "
                                       f"{type(e).__name__}: {e}") from e
                print(f"[stage0][warn] image {int(i)} {method}@{keep}: {type(e).__name__}: {str(e)[:120]}")
                recs.append(None); oks.append(False)
        D = next(r["last"].shape[0] for r in recs if r is not None)
        nan_vec = np.full(D, np.nan)
        def col(k, default):
            return np.array([r[k] if r is not None else default for r in recs])
        pack = {"last": np.stack([r["last"] if r is not None else nan_vec for r in recs]),
                "span": np.stack([r["span"] if r is not None else nan_vec for r in recs]),
                "margin": col("margin", np.nan).astype(float), "correct": col("correct", 0),
                "y": col("y", 0).astype(int), "n_carriers": col("n_carriers", np.nan).astype(float),
                "actual_keep": col("actual_keep", np.nan).astype(float),
                "seq_len": col("seq_len", np.nan).astype(float),
                "blk_pr": col("blk_pr", np.nan).astype(float),
                "blk_pr_trim": col("blk_pr_trim", np.nan).astype(float),
                "ok": np.array(oks, bool), "ids": np.asarray(idx, int)}
        return pack

    def calibrate_polarity(self, n: int = 16) -> float:
        """Sanity check at keep=1: accuracy with the declared label->answer mapping.
        If it is < 0.5 the mapping is flipped. We flip it, warn loudly, and record it
        in the results header."""
        idx = self.eval_idx[:n]
        acc = float(np.mean([self.one(int(i), "none", 1.0)["correct"] for i in idx]))
        if acc < 0.5:
            self.polarity_flipped = True
            print(f"[stage0][WARN] accuracy {acc:.2f} < 0.5 with pos={self.behaviour.pos_words}: "
                  "label polarity looks reversed for this dataset; flipping and recording it.")
            acc = 1.0 - acc
        return acc


# --------------------------------------------------------------------------- #
# mock collector (no torch): known effects, for the self-check
# --------------------------------------------------------------------------- #
class Stage0Mock:
    """Synthetic collector with KNOWN effects: pool keeps the concept, drop_* lose
    it with keep, drop_random loses more, readout_only collapses the span-read
    rank. A logic fixture for the self-check, not science."""

    def __init__(self, d=48, n_eval=160, n_monitor=96, seed=0):
        self.d, self.n_eval, self.n_monitor, self.seed = d, n_eval, n_monitor, seed

    def describe(self):
        return {"kind": "mock-stage0-v2", "d": self.d}

    def collect(self, split, method, keep):
        n = self.n_eval if split == "eval" else self.n_monitor
        rng = np.random.default_rng(self.seed + (1 if split == "eval" else 2) + int(1000 * keep)
                                    + sum(map(ord, method)))
        base = np.random.default_rng(self.seed + (1 if split == "eval" else 2))
        y = (base.random(n) < 0.5).astype(int)
        sig = {"none": 1.0, "pool": 0.9 + 0.1 * keep, "merge_dup": 0.85 + 0.15 * keep,
               "drop_norm": 0.4 + 0.6 * keep, "drop_query": 0.5 + 0.5 * keep,
               "drop_random": 0.3 + 0.7 * keep, "readout_only": 1.0}[method]
        w = np.zeros(self.d); w[0] = 1.0
        last = sig * (2 * y - 1)[:, None] * w + 0.7 * rng.standard_normal((n, self.d))
        rank = 3 if (method == "readout_only" and keep < 0.5) else self.d
        noise = rng.standard_normal((n, rank)) @ rng.standard_normal((rank, self.d)) / math.sqrt(rank)
        span = 0.8 * sig * (2 * y - 1)[:, None] * w + 0.7 * noise
        offs = base.standard_normal(n) * 0.3
        margin = (1.5 * sig) * (2 * y - 1) + offs + 0.4 * rng.standard_normal(n)
        k = max(1, int(round(keep * 64)))
        return {"last": last, "span": span, "margin": margin,
                "correct": ((margin > 0) == (y == 1)).astype(int), "y": y,
                "ok": np.ones(n, bool), "ids": np.arange(n) + (0 if split == "eval" else 10_000),
                "n_carriers": np.full(n, k, float), "actual_keep": np.full(n, k / 64),
                "seq_len": np.full(n, 80 - 64 + k, float),
                "blk_pr": np.full(n, min(k, 20) * 0.8), "blk_pr_trim": np.full(n, min(k, 20) * 0.9)}
