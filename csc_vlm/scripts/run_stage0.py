#!/usr/bin/env python3
"""
run_stage0.py  --  Stage 0 v2: forward-only VLM sanity (pipeline v2)

Run scripts/preflight.py FIRST: about 2 minutes, and it catches answer-token,
mask and compression bugs before any long run.

  # CPU logic self-check (no torch):
  python scripts/run_stage0.py --self-check

  # real run (Kaggle T4 / H100):
  python scripts/run_stage0.py --model qwen-3b --dataset hf:microsoft/cats_vs_dogs \
         --out runs_stage0/qwen3b_cats
  python scripts/run_stage0.py --model llava --dataset coco_person --out runs_stage0/llava_person

Default methods: pool, drop_norm, drop_random, drop_query, merge_dup (+ the
readout_only control). Default keeps: 1, 1/2, 1/4, 1/8, 1/16.
Outputs: stage0_results.json (incl. per-image margins), stage0.png, printed summary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csc_vlm._io import utf8_stdio  # noqa: E402
utf8_stdio()
from csc_vlm.stage0 import (DEFAULT_KEEPS, METHODS, Stage0Config, Stage0Mock,  # noqa: E402
                            Stage0VLM, run_stage0)
from csc_vlm import stats as st  # noqa: E402


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _row(R, m, keep):
    for r in R[m]:
        if abs(r["keep"] - keep) < 1e-9:
            return r
    return None


def print_summary(R):
    keeps = R["keeps"]
    print("\n" + "=" * 100)
    print(" STAGE 0 v2  --  forward-only sanity   ", {k: R["config"].get(k) for k in
                                                     ("model", "dataset", "behaviour", "read_layer",
                                                      "n_eval", "n_monitor", "dtype")})
    print("=" * 100)
    for m in R["methods"]:
        print(f"\n[{m}]")
        print("   keep   acc [95% CI]        lastAUROC refit [CI]     frozen   carriers  seq  "
              "blkPR  r90  top8")
        for r in R[m]:
            print(f"  {r['keep']:.4f}  {r['accuracy']:.3f} [{r['accuracy_lo']:.3f},{r['accuracy_hi']:.3f}]  "
                  f"{r['last_auroc_refit']:.3f} [{r['last_auroc_refit_lo']:.3f},{r['last_auroc_refit_hi']:.3f}]   "
                  f"{r['last_auroc_frozen']:.3f}   {r['n_carriers']:7.0f}  {r['seq_len']:5.0f}  "
                  f"{r['blk_pr']:5.1f}  {r['cov_r90']:3d}  {r['cov_top8']:.2f}")
    S = R["summary"]
    print("\n" + "-" * 100)
    print(" R1 -- information loss (paired slope of log|margin| vs log kept fraction; beta>0 = margin falls)")
    for m, d in S["logmargin_slope"].items():
        print("   " + st.describe_slope(m, d, "margin falls with compression", "margin rises"))
    kmin = min(k for k in keeps if all(_row(R, m, k) for m in R["methods"]))
    print(f"\n Contrasts at keep={kmin:.4f} (accuracy / last-token refit AUROC):")
    def pair(a, b, label):
        ra, rb = _row(R, a, kmin), _row(R, b, kmin)
        if ra and rb:
            print(f"   {label:52s} {a}: {ra['accuracy']:.3f}/{ra['last_auroc_refit']:.3f}   "
                  f"{b}: {rb['accuracy']:.3f}/{rb['last_auroc_refit']:.3f}")
    ms = set(R["methods"])
    if {"pool", "drop_norm"} <= ms:
        pair("pool", "drop_norm", "H2b precursor: info-preserving vs deleting")
    if {"drop_random", "drop_norm"} <= ms:
        pair("drop_random", "drop_norm", "G1: how-many (random) vs which (norm)")
    if {"drop_query", "drop_norm"} <= ms:
        pair("drop_query", "drop_norm", "R2: query-conditioned vs norm selection")
    if {"merge_dup", "pool"} <= ms:
        pair("merge_dup", "pool", "X3-like: fixed-N info bottleneck vs real pooling")
    if "readout_only" in ms:
        r1, rk = _row(R, "readout_only", 1.0), _row(R, "readout_only", kmin)
        print(f"   read artifact: readout_only span d_eff {r1['span_d_eff']:.1f} -> {rk['span_d_eff']:.1f} "
              f"(a collapse here is caused by the READ, not the model)")
    for m in R["methods"]:
        rk = _row(R, m, kmin)
        if rk:
            print(f"   staleness [{m:12s}] refit-frozen last AUROC = "
                  f"{rk['last_auroc_refit'] - rk['last_auroc_frozen']:+.3f}")
    for n in S["notes"]:
        print("   NOTE:", n)
    print("=" * 100)
    print(" Next: Stage 1 (scripts/run_stage1.py). Use the cov_r90/top8 columns to choose --r-max;\n"
          " if accuracy collapses for every method, the effect is R1 and the matched-accuracy control leads.")
    print("=" * 100 + "\n")


def make_figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    panels = [("accuracy", "Behaviour accuracy (R1)"), ("last_auroc_refit", "Last-token AUROC (refit)"),
              ("abs_margin_median", "|margin| median"), ("blk_pr", "Block participation ratio (carriers)"),
              ("cov_r90", "Read PCs for 90% variance (coverage)"), ("seq_len", "Sequence length")]
    for a, (key, title) in zip(ax.ravel(), panels):
        for m in R["methods"]:
            ks = [r["keep"] for r in R[m]]
            ys = [r[key] for r in R[m]]
            a.plot(ks, ys, "o-", label=m)
            lo, hi = key + "_lo", key + "_hi"
            if lo in R[m][0]:
                a.fill_between(ks, [r[lo] for r in R[m]], [r[hi] for r in R[m]], alpha=0.12)
        a.set_xscale("log"); a.invert_xaxis(); a.set_title(title); a.set_xlabel("keep (log; <- more compression)")
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Stage 0 v2: forward-only VLM sanity")
    ap.add_argument("--model", default="qwen-3b")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="hf:microsoft/cats_vs_dogs")
    ap.add_argument("--out", default="runs_stage0/default")
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--n-monitor", type=int, default=144)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--read-layer-frac", type=float, default=0.70)
    ap.add_argument("--query-layer", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--keeps", default="", help="comma list; default 1,0.5,0.25,0.125,0.0625")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--no-readout-only", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    keeps = [float(x) for x in args.keeps.split(",")] if args.keeps else list(DEFAULT_KEEPS)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    os.makedirs(args.out, exist_ok=True)

    if args.self_check:
        print("[self-check] Stage 0 v2 mock collector (no torch) ...")
        R = run_stage0(Stage0Mock(), keeps=keeps, methods=("pool", "drop_norm", "drop_random", "merge_dup"),
                       B=100, verbose=False)
        acc = lambda m: [r["accuracy"] for r in R[m]]
        assert acc("pool")[-1] > acc("drop_random")[-1], "pool should keep more accuracy than random drop"
        ro = [r["span_d_eff"] for r in R["readout_only"]]
        assert ro[-1] < 0.5 * ro[0], "readout-only should collapse span d_eff (read artifact)"
        sl = R["summary"]["logmargin_slope"]
        assert sl["drop_random"]["lo"] > 0, "random drop should show a margin-loss slope"
        print("[self-check] assertions PASS -- metrics, CIs, controls, orchestration OK")
    else:
        from csc_vlm.data import load_images
        from csc_vlm.stage0 import behaviour_for
        print(f"[stage0] dataset {args.dataset} (behaviour={behaviour_for(args.dataset).name}) ...")
        images, labels = load_images(args.dataset, per_class=args.per_class, seed=args.seed)
        cfg = Stage0Config(model=args.model, model_id=args.model_id, dataset=args.dataset,
                           read_layer_frac=args.read_layer_frac, query_layer=args.query_layer,
                           device=args.device, dtype=args.dtype, load_4bit=args.load_4bit,
                           n_eval=args.n_eval, n_monitor=args.n_monitor, seed=args.seed)
        coll = Stage0VLM(cfg, images, labels)
        acc0 = coll.calibrate_polarity()
        print(f"[stage0] keep=1 accuracy (16 imgs) = {acc0:.2f}; pos ids {coll.pos_ids} neg ids {coll.neg_ids}")
        R = run_stage0(coll, keeps=keeps, methods=methods,
                       include_readout_only=not args.no_readout_only, B=args.bootstrap)
    R["git"] = _git_rev()
    R["args"] = vars(args)
    with open(os.path.join(args.out, "stage0_results.json"), "w") as f:
        json.dump(R, f, indent=1, default=_jsonable)
    make_figure(R, os.path.join(args.out, "stage0.png"))
    print_summary(R)
    print(f"saved: {args.out}/stage0_results.json, {args.out}/stage0.png")


if __name__ == "__main__":
    main()
