"""
csc_vlm.compression  --  visual-token compression operators (pipeline v2)
=========================================================================

Why this file exists (first-principles review, 2026-11-07)
----------------------------------------------------------
The p2 prototypes did not test what their names said:

  * the old "merge" (``backends.tome_merge_torch``) replaced each token by its group
    mean but KEPT all N0 sequence positions. After layer 1 the duplicates diverge
    (different RoPE phase, different causal prefix), so the LM still has N0
    carriers: it is a fixed-N *information* bottleneck, not carrier compression.
  * the old "prune" removed carriers via attention-mask zeroing, but its
    equivalence to a real drop was never verified.

This module builds every operator as a *plan*: an explicit description of the new
input sequence. A plan says which rows survive, what the visual carriers are, and
which original position each row keeps. The same plan drives the forward-only
pass (Stage 0) and the differentiable carrier leaf (Stage 1). The leaf is always
the k DISTINCT carriers, so the attacker perturbs the real compressed
representation and never duplicates or masked rows.

Operators (``METHODS``)
-----------------------
  none          identity (keep=1.0 reference)
  pool          REAL shortening. Average-pool the visual tokens: 2-D on each
                image grid when the grid is known (LLaVA 24x24, Qwen grid_thw,
                SmolVLM 8x8 tiles), else 1-D contiguous groups. Information is
                largely preserved and carriers fall. This is the merge-family arm
                of the merge-vs-prune contrast (H2b).
  drop_norm     REAL shortening. Keep the top-k tokens by L2 norm (the old prune
                criterion), drop the rest from the sequence.
  drop_random   REAL shortening, random subset. Seeded per image, so the kept sets
                are nested across keeps. This is the G1 / H5 null.
  drop_query    REAL shortening. Keep the top-k tokens by cosine similarity to the
                last text token at an early layer (default 2), taken from an
                uncompressed pre-pass. A text-conditioned (query-based) selector:
                the ICML'26 token-pruning paper reports that this family can flip
                the sign of the safety effect, so it is our R2 contrast.
  merge_dup     LEGACY length-preserving group-mean (the old "merge"). Kept as the
                fixed-N information-bottleneck control (X3-like). The leaf is the
                k group means, expanded to N0 rows by gather.
  mask_norm     LEGACY: the drop_norm selection done by attention-mask zeroing
                with no shortening. Used only to verify mask == drop.

Positions: shortened plans KEEP each surviving row's ORIGINAL position id. A
pooled carrier takes its first member's position. With positions preserved,
drop_norm and mask_norm are mathematically identical: both remove the same keys
from every softmax, and the surviving queries keep their RoPE phases. The
preflight script verifies this numerically on the real model.

Pitfall handled: in transformers >= 5.x, ``create_causal_mask`` treats any jump
in ``position_ids`` as a packed-sequence boundary WHEN ``attention_mask is None``.
That would silently block attention across every dropped gap. We therefore always
pass an explicit 2-D attention mask with shortened plans.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["METHODS", "SHORTENING", "LEGACY", "Plan", "Run", "build_plan",
           "materialize", "update_lm_kwargs", "pool_groups_1d", "pool_groups_2d"]

METHODS = ("none", "pool", "drop_norm", "drop_random", "drop_query",
           "merge_dup", "mask_norm")
SHORTENING = ("pool", "drop_norm", "drop_random", "drop_query")
LEGACY = ("merge_dup", "mask_norm")


@dataclass
class Run:
    """A contiguous run of visual-token sequence positions, with its 2-D layout if
    known. ``grid`` is (frames, H, W) with frames*H*W == len(positions) in
    row-major order, or None for an unknown layout (1-D pooling fallback)."""
    positions: List[int]
    grid: Optional[Tuple[int, int, int]] = None


@dataclass
class Plan:
    """Explicit description of the compressed input sequence.

    Rows of the new sequence are gathered from S = cat(emb[0] (T rows), carriers
    (k rows)): ``src_index[j] < T`` copies an original row (text, or a masked
    visual row for mask_norm), and ``src_index[j] >= T`` takes carrier
    ``src_index[j] - T``. Carriers are ``Wc @ emb[0, span]``.
    """
    method: str
    keep: float
    T: int                              # original sequence length
    span: List[int]                     # original visual positions (sorted)
    Wc: torch.Tensor                    # [k, n_vis] carrier weights (rows sum to 1)
    src_index: torch.Tensor             # [T'] long
    pos_index: torch.Tensor             # [T'] long: original position kept by each row
    carrier_rows: torch.Tensor          # [*] long: new-row indices holding visual carriers (reads)
    mask_zero_rows: torch.Tensor        # [*] long: new-row indices to zero in the attention mask
    shortened: bool
    read_restrict: Optional[torch.Tensor] = None   # readout_only control: restrict span read
    info: dict = field(default_factory=dict)

    @property
    def n_carriers(self) -> int:
        return int(self.Wc.shape[0])

    @property
    def n_visual(self) -> int:
        return len(self.span)

    @property
    def new_len(self) -> int:
        return int(self.src_index.numel())

    @property
    def actual_keep(self) -> float:
        return self.n_carriers / max(1, self.n_visual)


# --------------------------------------------------------------------------- #
# grouping helpers
# --------------------------------------------------------------------------- #
def pool_groups_1d(n: int, keep: float) -> List[List[int]]:
    """k = round(keep*n) contiguous groups over n items (same partition as the
    legacy tome_merge_torch: idx = (arange(n)*k)//n)."""
    k = int(max(1, min(n, round(keep * n))))
    idx = (np.arange(n) * k) // n
    return [np.nonzero(idx == g)[0].tolist() for g in range(k)]


def pool_groups_2d(H: int, W: int, keep: float) -> List[List[int]]:
    """Adaptive average-pool groups on an HxW row-major grid. The output grid is
    Ho x Wo with Ho = round(H*sqrt(keep)) and Wo = round(W*sqrt(keep)), so the
    carrier count is about keep*H*W. Cell (i,j) goes to group
    (floor(i*Ho/H), floor(j*Wo/W)). Groups come back in row-major order of the
    output grid; each group's first member is its top-left cell."""
    s = float(np.sqrt(max(keep, 0.0)))
    Ho = int(max(1, min(H, round(H * s))))
    Wo = int(max(1, min(W, round(W * s))))
    gi = (np.arange(H) * Ho) // H
    gj = (np.arange(W) * Wo) // W
    groups: List[List[int]] = [[] for _ in range(Ho * Wo)]
    for i in range(H):
        for j in range(W):
            groups[gi[i] * Wo + gj[j]].append(i * W + j)
    return [g for g in groups if g]


def _onehot_rows(sel: Sequence[int], n: int, device, dtype) -> torch.Tensor:
    W = torch.zeros(len(sel), n, device=device, dtype=dtype)
    if len(sel):
        W[torch.arange(len(sel), device=device), torch.as_tensor(list(sel), device=device)] = 1.0
    return W


def _group_rows(groups: Sequence[Sequence[int]], n: int, device, dtype) -> torch.Tensor:
    W = torch.zeros(len(groups), n, device=device, dtype=dtype)
    for g, members in enumerate(groups):
        W[g, list(members)] = 1.0 / len(members)
    return W


def _k_of(n: int, keep: float) -> int:
    return int(max(1, min(n, round(keep * n))))


# --------------------------------------------------------------------------- #
# plan construction
# --------------------------------------------------------------------------- #
def build_plan(method: str, keep: float, T: int, runs: Sequence[Run],
               emb_span: torch.Tensor, *, scores: Optional[np.ndarray] = None,
               image_key: int = 0, seed: int = 0) -> Plan:
    """Build the plan for one image.

    Args:
      method, keep  operator name and keep fraction in (0, 1].
      T             original sequence length.
      runs          visual runs (positions + optional grid), in sequence order.
      emb_span      [n_vis, D] input embeddings of the visual positions (norms, dtype).
      scores        [n_vis] relevance scores (drop_query only; higher = keep).
      image_key     stable per-image integer (seeds drop_random; nested across keeps).
      seed          global seed.
    """
    if method not in METHODS and method != "readout_only":
        raise ValueError(f"unknown compression method {method!r}; choose from {METHODS}")
    span = [p for r in runs for p in r.positions]
    if sorted(span) != span or len(set(span)) != len(span):
        raise ValueError("visual runs must be sorted, disjoint sequence positions")
    n = len(span)
    if n == 0:
        raise ValueError("empty visual span")
    if span[-1] >= T - 1:
        raise ValueError("the last sequence position is visual; the read position must be a text token")
    dev, dt = emb_span.device, emb_span.dtype
    keep = float(keep)
    vis_pos = torch.as_tensor(span, device=dev, dtype=torch.long)
    span_set = set(span)
    nonvis = [t for t in range(T) if t not in span_set]
    empty = torch.zeros(0, dtype=torch.long, device=dev)

    def identity_plan(m: str) -> Plan:
        Wc = torch.eye(n, device=dev, dtype=dt)
        src = torch.arange(T, device=dev)
        src[vis_pos] = T + torch.arange(n, device=dev)
        return Plan(m, keep, T, span, Wc, src, torch.arange(T, device=dev),
                    vis_pos.clone(), empty, shortened=False)

    if method == "none" or keep >= 1.0:
        p = identity_plan("none" if method == "none" else method)
        p.keep = 1.0
        return p

    k = _k_of(n, keep)

    # ---- readout-only control: full forward, span read restricted to top-norm ----
    if method == "readout_only":
        p = identity_plan("readout_only")
        p.keep = keep
        top = torch.topk(emb_span.float().norm(dim=-1), k).indices.sort().values
        p.read_restrict = vis_pos[top]
        return p

    # ---- legacy length-preserving duplicate-merge ----
    if method == "merge_dup":
        groups = pool_groups_1d(n, keep)
        Wc = _group_rows(groups, n, dev, dt)
        gid = torch.empty(n, dtype=torch.long, device=dev)
        for g, mem in enumerate(groups):
            gid[mem] = g
        src = torch.arange(T, device=dev)
        src[vis_pos] = T + gid
        return Plan("merge_dup", keep, T, span, Wc, src, torch.arange(T, device=dev),
                    vis_pos.clone(), empty, shortened=False,
                    info={"groups": len(groups)})

    # ---- selection-based methods (drop_* and mask_norm) ----
    if method in ("drop_norm", "drop_random", "drop_query", "mask_norm"):
        if method in ("drop_norm", "mask_norm"):
            sel = torch.topk(emb_span.float().norm(dim=-1), k).indices.tolist()
        elif method == "drop_random":
            rng = np.random.default_rng([int(seed), int(image_key), 7919])
            sel = rng.permutation(n)[:k].tolist()     # nested across keeps for a fixed image
        else:
            if scores is None or len(scores) != n:
                raise ValueError("drop_query needs per-visual-token scores of length n_vis")
            sel = np.argsort(-np.asarray(scores, float), kind="stable")[:k].tolist()
        sel = sorted(int(s) for s in sel)
        Wc = _onehot_rows(sel, n, dev, dt)
        if method == "mask_norm":
            src = torch.arange(T, device=dev)
            src[vis_pos[sel]] = T + torch.arange(k, device=dev)
            dropped = sorted(set(range(n)) - set(sel))
            return Plan("mask_norm", keep, T, span, Wc, src, torch.arange(T, device=dev),
                        vis_pos[sel].clone(),
                        vis_pos[dropped].clone() if dropped else empty, shortened=False)
        # real drop: new sequence = non-visual rows + kept visual rows, original order
        rows = [(t, t) for t in nonvis] + [(span[s], T + j) for j, s in enumerate(sel)]
        rows.sort(key=lambda r: r[0])
        pos = torch.as_tensor([r[0] for r in rows], device=dev, dtype=torch.long)
        src = torch.as_tensor([r[1] for r in rows], device=dev, dtype=torch.long)
        carrier_rows = torch.nonzero(src >= T).flatten()
        return Plan(method, keep, T, span, Wc, src, pos, carrier_rows, empty, shortened=True)

    # ---- pool (real shortening, 2-D where the layout is known) ----
    assert method == "pool"
    groups_global: List[List[int]] = []
    offset = 0
    layout = []
    for r in runs:
        L = len(r.positions)
        if r.grid is not None:
            f, H, W = r.grid
            assert f * H * W == L, "grid does not match run length"
            for fr in range(f):
                for g in pool_groups_2d(H, W, keep):
                    groups_global.append([offset + fr * H * W + m for m in g])
            layout.append(f"{f}x{H}x{W}")
        else:
            for g in pool_groups_1d(L, keep):
                groups_global.append([offset + m for m in g])
            layout.append(f"1d{L}")
        offset += L
    Wc = _group_rows(groups_global, n, dev, dt)
    anchors = [span[min(g)] for g in groups_global]
    rows = [(t, t) for t in nonvis] + [(a, T + j) for j, a in enumerate(anchors)]
    rows.sort(key=lambda r: r[0])
    pos = torch.as_tensor([r[0] for r in rows], device=dev, dtype=torch.long)
    src = torch.as_tensor([r[1] for r in rows], device=dev, dtype=torch.long)
    carrier_rows = torch.nonzero(src >= T).flatten()
    return Plan("pool", keep, T, span, Wc, src, pos, carrier_rows, empty, shortened=True,
                info={"layout": layout})


# --------------------------------------------------------------------------- #
# applying a plan
# --------------------------------------------------------------------------- #
def materialize(plan: Plan, emb: torch.Tensor, carriers: torch.Tensor) -> torch.Tensor:
    """New inputs_embeds [1, T', D] = cat(emb[0].detach(), carriers)[src_index].
    ``carriers`` [k, D] may be a leaf with requires_grad (Stage 1); gradients flow
    only to it. Text rows are detached, so the attacker cannot touch them."""
    if emb.shape[0] != 1:
        raise ValueError("compression plans are per-image (batch size 1)")
    S = torch.cat([emb[0].detach(), carriers.to(emb.dtype)], dim=0)
    return S.index_select(0, plan.src_index)[None]


def update_lm_kwargs(plan: Plan, kwargs: dict) -> dict:
    """Rewrite the text-backbone call's kwargs for the new sequence: attention mask
    (always explicit for shortened plans), position ids (original positions kept),
    cache_position, and any token-type ids. Returns the updated dict."""
    T = plan.T
    am = kwargs.get("attention_mask", None)
    dev = plan.src_index.device
    if isinstance(am, dict):
        raise RuntimeError("attention_mask arrived as a prepared mask mapping; per-row "
                           "editing is not supported. Call the model without precomputed masks.")
    if am is not None and am.dim() != 2:
        raise RuntimeError(f"expected a 2-D attention mask at the text backbone, got {tuple(am.shape)}")
    need_mask = plan.shortened or plan.mask_zero_rows.numel() > 0
    if need_mask:
        base = am if am is not None else torch.ones(1, T, dtype=torch.long, device=dev)
        if base.shape[-1] != T:
            raise RuntimeError(f"attention mask length {base.shape[-1]} != sequence length {T}")
        new_am = base.index_select(1, plan.pos_index.to(base.device)).clone()
        if plan.mask_zero_rows.numel():
            new_am[:, plan.mask_zero_rows.to(base.device)] = 0
        kwargs["attention_mask"] = new_am
    if plan.shortened:
        pid = kwargs.get("position_ids", None)
        if pid is None:
            kwargs["position_ids"] = plan.pos_index[None].to(dev)
        else:
            if pid.shape[-1] != T:
                raise RuntimeError(f"position_ids length {pid.shape[-1]} != {T}")
            kwargs["position_ids"] = pid.index_select(pid.dim() - 1, plan.pos_index.to(pid.device))
        cp = kwargs.get("cache_position", None)
        if cp is not None:
            kwargs["cache_position"] = torch.arange(plan.new_len, device=cp.device, dtype=cp.dtype)
        for key in ("token_type_ids", "mm_token_type_ids"):
            v = kwargs.get(key, None)
            if torch.is_tensor(v) and v.shape[-1] == T:
                kwargs[key] = v.index_select(v.dim() - 1, plan.pos_index.to(v.device))
    return kwargs
