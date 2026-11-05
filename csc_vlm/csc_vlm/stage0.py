"""
csc_vlm.stage0  --  Stage 0: forward-passes-only VLM sanity (plan Phase 5.1 / E2 pilot)
=======================================================================================

Stage 0 answers the questions the X1 toy cannot, because they are VLM-specific:

  Q1 (R1 / information loss): does compression KEEP the concept? If accuracy and a
     refit probe both survive, any later "auditability" effect is not just a dumber
     model. We log per-image behaviour margin + correctness and probe AUROC.

  Q2 (read artifact): is the prune d_eff / leverage collapse we saw in RUN 1-8 a real
     property, or an artifact of pooling a few top-norm tokens? The READOUT-ONLY
     control runs the model UNCOMPRESSED but pools the read over the top-norm
     positions only. If d_eff collapses there too, it is a read artifact.

  Q3 (staleness): FROZEN monitor (fit once at keep=1.0) vs REFIT monitor (fit at each
     keep). A frozen-only drop is staleness, not intrinsic evadability.

  Q4 (universality / G1 null): the RANDOM-PRUNE arm. If random pruning shows the same
     sign, the effect is about how many tokens survive, not which.

It is forward-passes only -- NO attack, NO gradients. Cheap (<~1 h/model). The evasion
metric itself (Jacobian Gamma_r + OC) is Stage 1; this stage decides whether Stage 1
is worth running and gives the matched-accuracy covariate it needs.

Reads at BOTH positions: the LAST TOKEN (what a real latent monitor reads) and the
VISUAL-SPAN MEAN (what p2 used). Metrics per (method, keep): accuracy, median margin,
probe AUROC (frozen & refit), class separation d', and d_eff -- for each read.

Design: the VLM collection REUSES the existing p2 HFVLMBackend (model loading, hook
machinery) without modifying it; torch is imported lazily so this module (and the mock
self-check) import with numpy alone. v0.4 untouched; new p2-only file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["BehaviourSpec", "BEHAVIOURS", "auroc", "fit_linear_probe", "probe_scores",
           "dprime", "d_eff", "keep_metrics", "run_stage0",
           "Stage0VLM", "Stage0Mock", "DEFAULT_KEEPS", "METHODS"]

EPS = 1e-12
DEFAULT_KEEPS: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10, 0.06)
METHODS: Tuple[str, ...] = ("merge", "prune", "random_prune")  # + readout_only control


# --------------------------------------------------------------------------- #
# behaviour specs (the task that gives the model a real answer to be right/wrong on)
# --------------------------------------------------------------------------- #
@dataclass
class BehaviourSpec:
    name: str
    prompt: str          # user text; the model answers at the next token
    pos_word: str        # answer that maps to label == 1
    neg_word: str        # answer that maps to label == 0


BEHAVIOURS: Dict[str, BehaviourSpec] = {
    "cats_vs_dogs": BehaviourSpec(
        "cats_vs_dogs", "Is this a cat or a dog? Answer with one word.", "cat", "dog"),
    "coco_person": BehaviourSpec(
        "coco_person", "Is there a person in this image? Answer yes or no.", "yes", "no"),
}


# --------------------------------------------------------------------------- #
# pure-numpy metrics (the part that must be correct; smoke-tested in --self-check)
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney). No sklearn dependency."""
    scores = np.asarray(scores, float); y = np.asarray(y)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_linear_probe(H: np.ndarray, y: np.ndarray, lam: float = 1.0
                     ) -> Tuple[np.ndarray, float]:
    """Ridge linear probe to +/-1 targets (HiddenDetect-style linear monitor).
    Closed form, stable when n < d via the lam ridge term. Returns (w, b)."""
    X = np.hstack([H, np.ones((H.shape[0], 1))])
    t = 2.0 * np.asarray(y, float) - 1.0
    A = X.T @ X + lam * np.eye(X.shape[1])
    sol = np.linalg.solve(A, X.T @ t)
    return sol[:-1], float(sol[-1])


def probe_scores(H: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return H @ w + b


def dprime(H: np.ndarray, y: np.ndarray, direction: np.ndarray) -> float:
    """Class separation (Cohen's d) along a direction: |mu1-mu0| / pooled sd."""
    s = H @ direction
    s1, s0 = s[y == 1], s[y == 0]
    if len(s1) < 2 or len(s0) < 2:
        return 0.0
    pooled = np.sqrt(0.5 * (s1.var(ddof=1) + s0.var(ddof=1))) + EPS
    return float(abs(s1.mean() - s0.mean()) / pooled)


def d_eff(H: np.ndarray) -> float:
    """Participation ratio of the centred read spectrum (same def as geometry.py)."""
    Hc = H - H.mean(0, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    ev = s ** 2
    tot = ev.sum()
    return float((tot ** 2) / (ev ** 2).sum()) if tot > 0 else 1.0


# --------------------------------------------------------------------------- #
# per-(method, keep) metric computation from collected packs
# --------------------------------------------------------------------------- #
def _one_read_metrics(eval_H, eval_y, mon_H, mon_y, mon_H_full, mon_y_full,
                      lam: float) -> Dict[str, float]:
    """AUROC (frozen & refit), d', d_eff for ONE read position."""
    out = {}
    # refit: probe fit on monitor reads at THIS keep
    w_r, b_r = fit_linear_probe(mon_H, mon_y, lam)
    out["auroc_refit"] = auroc(probe_scores(eval_H, w_r, b_r), eval_y)
    # frozen: probe fit on monitor reads at keep=1.0
    w_f, b_f = fit_linear_probe(mon_H_full, mon_y_full, lam)
    out["auroc_frozen"] = auroc(probe_scores(eval_H, w_f, b_f), eval_y)
    # class separation along the (refit) probe direction, and d_eff of the eval read
    out["dprime"] = dprime(eval_H, eval_y, w_r / (np.linalg.norm(w_r) + EPS))
    out["d_eff"] = d_eff(eval_H)
    return out


def keep_metrics(eval_pack: Dict, mon_pack: Dict, mon_pack_full: Dict,
                 lam: float = 1.0) -> Dict[str, float]:
    """All metrics for one (method, keep): behaviour + both reads."""
    m = {
        "accuracy": float(np.mean(eval_pack["correct"])),
        "margin_median": float(np.median(eval_pack["margin"])),
        "margin_mean": float(np.mean(eval_pack["margin"])),
    }
    for read in ("last", "span"):
        r = _one_read_metrics(eval_pack[read], eval_pack["y"],
                              mon_pack[read], mon_pack["y"],
                              mon_pack_full[read], mon_pack_full["y"], lam)
        for k, v in r.items():
            m[f"{read}_{k}"] = v
    return m


# --------------------------------------------------------------------------- #
# orchestration (collector-agnostic: works for VLM or mock)
# --------------------------------------------------------------------------- #
def run_stage0(collector, keeps=DEFAULT_KEEPS, methods=METHODS,
               include_readout_only=True, lam: float = 1.0) -> Dict:
    """Collector must implement collect(keep, method) -> pack with keys
    last[n,d], span[n,d], margin[n], correct[n], y[n] for the EVAL set, and
    collect_monitor(keep, method) -> same for the MONITOR-train set."""
    results: Dict[str, Dict] = {"keeps": list(keeps), "methods": [], "config": collector.describe()}

    # frozen monitor is fit on UNCOMPRESSED monitor reads (method-agnostic at keep=1.0)
    mon_full = collector.collect_monitor(1.0, "merge")

    all_methods = list(methods) + (["readout_only"] if include_readout_only else [])
    for method in all_methods:
        rows = []
        for keep in keeps:
            ev = collector.collect(keep, method)
            mon = collector.collect_monitor(keep, method)
            row = {"keep": keep}
            row.update(keep_metrics(ev, mon, mon_full, lam))
            rows.append(row)
        results[method] = rows
        results["methods"].append(method)
    return results


# --------------------------------------------------------------------------- #
# VLM collector -- reuses the existing HFVLMBackend (lazy torch import)
# --------------------------------------------------------------------------- #
@dataclass
class Stage0Config:
    model: str = "llava"
    model_id: str = ""
    dataset: str = "cats_vs_dogs"      # key into BEHAVIOURS
    read_layer_frac: float = 0.70
    device: str = "cuda"
    load_4bit: bool = False
    n_eval: int = 256
    n_monitor: int = 144
    seed: int = 0


class Stage0VLM:
    """Forward-only collector over a real VLM. Builds one HFVLMBackend (model +
    processor + hook helpers) and reuses its _find_embeds / _candidate_hook_modules
    / _visual_span. Its own pre-hook supports merge / prune / random_prune /
    readout_only and records the read positions, so NO change to backends.py."""

    def __init__(self, cfg: Stage0Config, images: List, labels: np.ndarray):
        import torch                                    # lazy
        from .backends import HFVLMBackend, VLMConfig, tome_merge_torch, tome_prune_mask
        self.torch = torch
        self._merge = tome_merge_torch
        self._prune_mask = tome_prune_mask
        self.cfg = cfg
        self.behaviour = BEHAVIOURS[cfg.dataset]
        self.images = images
        self.labels = np.asarray(labels)
        bcfg = VLMConfig(model=cfg.model, model_id=cfg.model_id,
                         read_layer_frac=cfg.read_layer_frac, device=cfg.device,
                         load_4bit=cfg.load_4bit)
        self.backend = HFVLMBackend(bcfg, images, labels)   # loads the model
        self.model = self.backend.model
        self.processor = self.backend.processor
        self.read_layer = self.backend.read_layer
        self._pos_id, self._neg_id = self._resolve_answer_ids()
        # fixed, disjoint eval / monitor index splits (paired across keeps)
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(images))
        self._eval_idx = perm[: cfg.n_eval]
        self._mon_idx = perm[cfg.n_eval: cfg.n_eval + cfg.n_monitor]
        # per-forward state read by the hook
        self._keep = 1.0; self._method = "merge"; self._span = None
        self._read_positions = None; self._fired = False; self._call_id = 0

    # -- describe (for results header) --
    def describe(self) -> Dict:
        return {"kind": "vlm", "model": self.cfg.model, "dataset": self.cfg.dataset,
                "read_layer": int(self.read_layer), "n_eval": len(self._eval_idx),
                "n_monitor": len(self._mon_idx),
                "pos_word": self.behaviour.pos_word, "neg_word": self.behaviour.neg_word,
                "pos_id": int(self._pos_id), "neg_id": int(self._neg_id)}

    # -- answer-token resolution (VERIFY on first real run via --self-check-ids) --
    def _resolve_answer_ids(self) -> Tuple[int, int]:
        tok = getattr(self.processor, "tokenizer", self.processor)

        def first_id(word: str) -> int:
            for variant in (" " + word, word):
                try:
                    ids = tok.encode(variant, add_special_tokens=False)
                except TypeError:
                    ids = tok.encode(variant)
                if ids:
                    return int(ids[0])
            return int(tok.encode(word)[0])

        return first_id(self.behaviour.pos_word), first_id(self.behaviour.neg_word)

    # -- prompt build (mirrors backend._prepare but with the behaviour question) --
    def _prepare(self, image):
        txt = self.behaviour.prompt
        try:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": txt}]}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inp = self.processor(images=image, text=text, return_tensors="pt")
        except Exception:
            inp = self.processor(images=image,
                                 text=f"USER: <image>\n{txt} ASSISTANT:",
                                 return_tensors="pt")
        dev = next(self.model.parameters()).device
        return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inp.items()}

    # -- the compression hook (merge / prune / random_prune / readout_only) --
    def _hook(self, module, args, kwargs):
        if self._fired or self._keep >= 1.0 or self._span is None:
            return None
        emb, loc = self.backend._find_embeds(args, kwargs)
        if emb is None:
            return None
        sp = self._span
        if len(sp) <= 1:
            return None
        torch = self.torch
        k = max(1, int(round(self._keep * len(sp))))

        if self._method == "merge":
            merged = emb.clone()
            merged[:, sp, :] = self._merge(emb[:, sp, :], self._keep, torch)
            self._read_positions = list(sp); self._fired = True
            if loc == "kwargs":
                kwargs["inputs_embeds"] = merged; return (args, kwargs)
            na = list(args); na[loc] = merged; return (tuple(na), kwargs)

        # subset-selection methods
        if self._method == "random_prune":
            r = np.random.default_rng(self.cfg.seed + 101 + int(self._call_id))
            keep_local = sorted(r.permutation(len(sp))[:k].tolist())
        else:                                   # prune, readout_only -> top-L2-norm
            mask = self._prune_mask(len(sp), self._keep, emb[:, sp, :], torch)
            km = mask.detach().cpu().numpy().astype(bool)
            keep_local = [i for i in range(len(sp)) if km[i]]
        kept = [sp[i] for i in keep_local] or list(sp)

        if self._method in ("prune", "random_prune"):
            # genuine removal: stop surviving tokens attending to the dropped ones
            attn = kwargs.get("attention_mask")
            if attn is not None:
                keptset = set(kept)
                pruned = [p for p in sp if p not in keptset]
                if pruned:
                    attn = attn.clone()
                    pr = torch.tensor(pruned, device=attn.device, dtype=torch.long)
                    attn[:, pr] = 0
                    kwargs["attention_mask"] = attn
        # readout_only: model runs UNCOMPRESSED; only the span read is restricted.
        self._read_positions = kept
        self._fired = True
        return (args, kwargs)

    def _collect_idx(self, keep: float, method: str, idx: np.ndarray) -> Dict:
        torch = self.torch
        handles = [m.register_forward_pre_hook(self._hook, with_kwargs=True)
                   for m in self.backend._candidate_hook_modules()]
        self._keep = float(keep); self._method = method
        last, span, marg, corr, ys = [], [], [], [], []
        try:
            for call_id, i in enumerate(idx):
                self._call_id = call_id
                inp = self._prepare(self.images[int(i)])
                self._span = self.backend._visual_span(inp)
                self._read_positions = list(self._span)
                self._fired = False
                with torch.no_grad():
                    out = self.model(**inp, output_hidden_states=True)
                hs = out.hidden_states[self.read_layer][0]          # [seq, d]
                last.append(hs[-1].float().cpu().numpy())
                rp = self._read_positions or self._span
                span.append(hs[rp].mean(0).float().cpu().numpy())
                logit = out.logits[0, -1, :].float()
                m = float(logit[self._pos_id] - logit[self._neg_id])
                y = int(self.labels[int(i)])
                marg.append(m); ys.append(y); corr.append(int((m > 0) == (y == 1)))
        finally:
            for h in handles:
                h.remove()
        return {"last": np.asarray(last), "span": np.asarray(span),
                "margin": np.asarray(marg), "correct": np.asarray(corr),
                "y": np.asarray(ys)}

    def collect(self, keep, method):
        return self._collect_idx(keep, method, self._eval_idx)

    def collect_monitor(self, keep, method):
        return self._collect_idx(keep, method, self._mon_idx)


# --------------------------------------------------------------------------- #
# mock collector -- NO torch; fabricates reads/margins with KNOWN effects so the
# metrics, controls, orchestration, figure and summary are validated end-to-end.
# --------------------------------------------------------------------------- #
class Stage0Mock:
    """Synthetic forward-only collector. Encodes the hypotheses we expect to SEE so
    the harness logic can be checked: merge keeps accuracy + last-token AUROC, prune
    loses information (accuracy down, AUROC down), readout_only collapses the SPAN
    read's d_eff (the read artifact), random_prune sits between. Frozen AUROC lags
    refit (staleness) wherever the read moves. This is a logic fixture, not science."""

    def __init__(self, d=48, n_eval=200, n_monitor=120, seed=0):
        self.d, self.n_eval, self.n_monitor, self.seed = d, n_eval, n_monitor, seed

    def describe(self):
        return {"kind": "mock", "d": self.d, "n_eval": self.n_eval,
                "n_monitor": self.n_monitor}

    def _gen(self, keep, method, n, seed):
        rng = np.random.default_rng(seed)
        d = self.d
        y = (rng.random(n) < 0.5).astype(int)
        # signal strength of the concept at this keep
        if method == "merge":
            sig_last, sig_span, acc_sig = 1.0, 0.9 * keep + 0.1, 1.0
        elif method == "prune":
            sig_last, sig_span, acc_sig = 0.4 + 0.6 * keep, 0.5 * keep, 0.5 + 0.5 * keep
        elif method == "random_prune":
            sig_last, sig_span, acc_sig = 0.3 + 0.5 * keep, 0.4 * keep, 0.4 + 0.5 * keep
        else:                                    # readout_only (full forward)
            sig_last, sig_span, acc_sig = 1.0, 0.9, 1.0
        wsig = np.zeros(d); wsig[0] = 1.0
        def read(sig, rank):
            noise = rng.standard_normal((n, d))
            if rank < d:                         # collapse to a low-rank blob
                B = rng.standard_normal((d, rank))
                noise = (rng.standard_normal((n, rank)) @ B.T)
            return sig * (2 * y - 1)[:, None] * wsig[None, :] + 0.6 * noise
        # readout_only collapses the SPAN read rank (the artifact); last is fine
        span_rank = 3 if method in ("readout_only", "prune") and keep < 0.5 else d
        last = read(sig_last, d)
        span = read(sig_span, span_rank)
        margin = acc_sig * (2 * y - 1) + 0.7 * rng.standard_normal(n)
        correct = ((margin > 0) == (y == 1)).astype(int)
        return {"last": last, "span": span, "margin": margin, "correct": correct, "y": y}

    def collect(self, keep, method):
        return self._gen(keep, method, self.n_eval, self.seed + 1)

    def collect_monitor(self, keep, method):
        return self._gen(keep, method, self.n_monitor, self.seed + 2)