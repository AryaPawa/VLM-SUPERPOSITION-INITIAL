#!/usr/bin/env python3
"""
preflight.py  --  run this FIRST on every new model or environment (about 2-5 min).

It checks, on a handful of real images, the things that would silently invalidate
Stage 0/1:

  [1] the model loads; text backbone, read layer and dtype are reported;
  [2] the visual span and image grid are detected (no 576-position fallback);
  [3] answer tokens: the pos/neg id sets are decoded, the top-5 next tokens are
      printed, accuracy at keep=1, and the polarity is checked;
  [4] each compression method changes the sequence and the read (hard gate),
      and the logits stay finite;
  [5] mask_norm == drop_norm: masking keys with original positions kept must
      equal really dropping them. If this fails, every prune result is invalid;
  [6] (--grad) carrier-leaf gradients are finite, nonzero, and match a finite
      difference;
  [7] timing: seconds per forward and per Stage-1 image, and an ETA for the
      default Stage 0 / Stage 1 configs.

Exit code 0 = all hard checks pass. Writes <out>/preflight.json.

  python scripts/preflight.py --model qwen-3b --dataset hf:microsoft/cats_vs_dogs --grad
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csc_vlm._io import utf8_stdio  # noqa: E402
utf8_stdio()


def main():
    ap = argparse.ArgumentParser(description="pipeline-v2 preflight checks")
    ap.add_argument("--model", default="qwen-3b")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="hf:microsoft/cats_vs_dogs")
    ap.add_argument("--n", type=int, default=12, help="images to use")
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grad", action="store_true", help="also check Stage-1 gradients")
    ap.add_argument("--out", default="runs_preflight/default")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from csc_vlm.data import load_images
    from csc_vlm.stage0 import Stage0Config, Stage0VLM, behaviour_for
    from csc_vlm.vlm import VLMHandle

    report = {"checks": {}, "info": {}}
    hard_fail = []

    def check(name, ok, detail, hard=True):
        report["checks"][name] = {"ok": bool(ok), "detail": detail, "hard": hard}
        tag = "PASS" if ok else ("FAIL" if hard else "WARN")
        print(f"  [{tag}] {name}: {detail}")
        if hard and not ok:
            hard_fail.append(name)

    print("[1] loading model ...")
    t0 = time.time()
    h = VLMHandle.from_pretrained(args.model, model_id=args.model_id, dtype=args.dtype,
                                  device=args.device, single_device=args.grad)
    report["info"]["model"] = h.describe()
    report["info"]["load_seconds"] = time.time() - t0
    print("   ", h.describe())

    per_class = max(2, (args.n + 1) // 2)
    images, labels = load_images(args.dataset, per_class=per_class, seed=args.seed)
    beh = behaviour_for(args.dataset)
    cfg = Stage0Config(model=args.model, dataset=args.dataset, n_eval=len(images) - 2, n_monitor=2)
    s0 = Stage0VLM(cfg, images, labels, handle=h)
    tok = h.tokenizer

    print("[2] visual span / grid ...")
    inp0 = s0.inputs(0)
    runs = h.visual_runs(inp0)
    report["info"]["runs"] = [{"len": len(r.positions), "grid": r.grid} for r in runs]
    nvis = sum(len(r.positions) for r in runs)
    check("visual_span", nvis > 1, f"{len(runs)} run(s), {nvis} visual tokens, grids {[r.grid for r in runs]}, "
          f"seq len {inp0['input_ids'].shape[1]}")

    print("[3] answer tokens ...")
    dec = lambda ids: [tok.decode([i]) for i in ids]
    report["info"]["pos_ids"] = {int(i): s for i, s in zip(s0.pos_ids, dec(s0.pos_ids))}
    report["info"]["neg_ids"] = {int(i): s for i, s in zip(s0.neg_ids, dec(s0.neg_ids))}
    print(f"    prompt: {beh.prompt!r}")
    print(f"    pos {beh.pos_words} -> {report['info']['pos_ids']}")
    print(f"    neg {beh.neg_words} -> {report['info']['neg_ids']}")
    in_top, accs, times = 0, [], []
    for i in range(min(len(images), args.n)):
        inp = s0.inputs(i)
        t = time.time()
        o = h.compressed_run(inp, "none", 1.0)
        times.append(time.time() - t)
        lg = o["logits"]
        top = torch.topk(lg, 5).indices.tolist()
        in_top += int(any(t_ in s0.pos_ids + s0.neg_ids for t_ in top))
        mg = s0.margin_of(lg)
        accs.append(int((mg > 0) == (int(labels[i]) == 1)))
        if i < 4:
            print(f"    img{i} label={int(labels[i])} margin={mg:+.2f} top5={dec(top)}")
    acc = float(np.mean(accs))
    check("logits_finite", bool(torch.isfinite(lg).all()), "uncompressed logits finite")
    check("answer_in_top5", in_top >= 0.75 * len(accs),
          f"an answer token is in the top-5 next tokens on {in_top}/{len(accs)} images "
          "(if low: the model answers with other words -- edit BEHAVIOURS prompt/words)", hard=False)
    check("accuracy_keep1", acc >= 0.6,
          f"accuracy={acc:.2f} on {len(accs)} images "
          f"({'POLARITY REVERSED: fix BEHAVIOURS' if acc <= 0.4 else 'ok' if acc >= 0.6 else 'near chance'})")
    report["info"]["sec_per_forward"] = float(np.median(times))

    print(f"[4] compression methods at keep={args.keep} ...")
    rl = h.read_layer
    base = h.compressed_run(inp0, "none", 1.0, capture=(rl,))
    bl = base["hs"][rl][-1].float()
    for m in ("pool", "drop_norm", "drop_random", "drop_query", "merge_dup", "mask_norm"):
        try:
            o = h.compressed_run(inp0, m, args.keep, image_key=0, capture=(rl,))
            p = o["plan"]
            drel = float((o["hs"][rl][-1].float() - bl).norm() / (bl.norm() + 1e-9))
            ok = bool(torch.isfinite(o["logits"]).all()) and drel > 1e-4 and p.n_carriers < p.n_visual
            check(f"compress_{m}", ok, f"carriers {p.n_carriers}/{p.n_visual}, seq {p.T}->{o['new_len']}, "
                  f"read change {drel:.3e}")
        except Exception as e:  # noqa: BLE001
            check(f"compress_{m}", False, f"{type(e).__name__}: {str(e)[:200]}")

    print("[5] mask_norm == drop_norm (positions preserved) ...")
    worst = 0.0
    for i in range(min(3, len(images))):
        inp = s0.inputs(i)
        a = h.compressed_run(inp, "mask_norm", args.keep)["logits"]
        b = h.compressed_run(inp, "drop_norm", args.keep)["logits"]
        worst = max(worst, float((a - b).abs().max() / (a.abs().max() + 1e-9)))
    tol = 1e-3 if h.dtype == torch.float32 else 3e-2
    check("mask_equals_drop", worst < tol, f"max relative |logit diff| = {worst:.2e} (tol {tol:g}, {h.dtype})")

    if args.grad:
        print("[6] Stage-1 gradients ...")
        t = time.time()
        o = h.compressed_run(inp0, "pool", args.keep, make_leaf=True, capture=(rl,), grad=True)
        v = o["leaf"]
        bb = torch.logsumexp(o["logits"][s0.pos_ids], 0) - torch.logsumexp(o["logits"][s0.neg_ids], 0)
        g = torch.autograd.grad(bb, v, retain_graph=True)[0]
        for _ in range(19):        # time a full Stage-1 image (1 + 8 + 8 + 4 backward passes)
            torch.autograd.grad((o["hs_raw"][rl][0, -1].float() * torch.randn(h.hidden, device=v.device)).sum(),
                                v, retain_graph=True)
        report["info"]["sec_per_stage1_image"] = time.time() - t
        gn = float(g.float().norm())
        check("grad_finite_nonzero", bool(torch.isfinite(g).all()) and gn > 0, f"||g_b|| = {gn:.3e}")
        u = g / (g.norm() + 1e-30)
        eps = 1e-2 * float(v.detach().float().norm()) / math.sqrt(v.numel())
        def bval(vv):
            r = h.compressed_run(inp0, "pool", args.keep, carriers_override=vv)
            return float(torch.logsumexp(r["logits"][s0.pos_ids], 0) - torch.logsumexp(r["logits"][s0.neg_ids], 0))
        fd = (bval(v.detach() + eps * u) - bval(v.detach() - eps * u)) / (2 * eps)
        rel = abs(fd - gn) / (gn + 1e-12)
        check("grad_matches_fd", rel < (0.05 if h.dtype == torch.float32 else 0.3),
              f"directional FD {fd:.4e} vs ||g|| {gn:.4e} (rel err {rel:.2%})",
              hard=h.dtype == torch.float32)

    print("[7] timing / ETA ...")
    spf = report["info"]["sec_per_forward"]
    n_s0 = 5 * 4 * (256 + 144) + 400        # 5 methods x 4 compressed keeps x images + keep1 + query pre-pass
    print(f"    ~{spf:.2f}s/forward -> default Stage 0 ~ {n_s0 * spf / 3600:.1f} h")
    if "sec_per_stage1_image" in report["info"]:
        sp1 = report["info"]["sec_per_stage1_image"]
        print(f"    ~{sp1:.2f}s/Stage-1 image -> default Stage 1 ~ {4 * 5 * 64 * sp1 / 3600 + 4 * 5 * 144 * spf / 3600:.1f} h")
    report["hard_failures"] = hard_fail
    with open(os.path.join(args.out, "preflight.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(("\nPREFLIGHT PASS" if not hard_fail else f"\nPREFLIGHT FAIL: {hard_fail}") +
          f"  (saved {args.out}/preflight.json)")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
