#!/usr/bin/env python3
"""
run_stage1.py  --  Stage 1 v2: carrier-space evasion geometry (pipeline v2)

  # CPU logic self-check (no torch):
  python scripts/run_stage1.py --self-check

  # real run: ONE GPU, float32 (or bfloat16 on Ampere+/H100):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1.py --model qwen-3b \
      --dataset hf:microsoft/cats_vs_dogs --n-images 64 --out runs_stage1/qwen3b_cats

Cost per image per cell: 1 + r_max + r_frozen + 2 backward passes, plus
n_check extra forwards per cell. Defaults: r_max=8, r_frozen=8, 64 images,
5 keeps, 4 methods. Use --no-staleness to drop the frozen-monitor passes.

Reading the output: everything is a paired slope beta = d log X / d log keep with
a cluster-bootstrap CI (beta>0 means X FALLS under compression). The headline
terms are log T (the monitor's obfuscation tax) and A - A_placebo
(behaviour-specific loss of alignment). The decomposition
log OC = log m - log||g|| + log T is exact per image, so the signed shares add
up to 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csc_vlm.evasion import (DEFAULT_KEEPS, METHODS, READ_RANKS, Stage1Mock,  # noqa: E402
                             format_stage1, run_stage1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_stage0 import _git_rev, _jsonable  # noqa: E402


def make_figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = R["summary"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for method, M in S["methods"].items():
        ks = [r["keep"] for r in M["rows"]]
        ax[0, 0].plot(ks, [r["T"] for r in M["rows"]], "o-", label=method)
        ax[0, 1].plot(ks, [r["cap_in_curve"][-1] for r in M["rows"]], "o-", label=f"{method} real")
        ax[0, 1].plot(ks, [r["cap_pl_out"] for r in M["rows"]], "x:", label=f"{method} placebo_out")
        ax[0, 2].plot(ks, [r["OC"] for r in M["rows"]], "o-", label=method)
        if "cap_read_curve" in M["rows"][0]:
            for r in M["rows"]:
                ax[1, 0].plot(R["read_ranks"], r["cap_read_curve"], "-",
                              alpha=0.3 + 0.7 * r["keep"], label=f"{method} k={r['keep']:.3g}")
        ax[1, 1].plot(ks, [r["m"] for r in M["rows"]], "o-", label=method)
    names = list(S["methods"])
    x = np.arange(len(names))
    for j, key in enumerate(("log_m", "neg_log_g", "log_T")):
        vals = [S["methods"][n]["decomposition"][key]["slope"] for n in names]
        ax[1, 2].bar(x + (j - 1) * 0.27, vals, 0.27, label=key)
    ax[1, 2].set_xticks(x); ax[1, 2].set_xticklabels(names, rotation=20)
    ax[1, 2].axhline(0, color="k", lw=0.5); ax[1, 2].legend(fontsize=7)
    ax[1, 2].set_title("signed slopes of log-OC terms (beta>0: falls under compression)")
    titles = ["obfuscation tax T = 1/Gamma (median)", "captured fraction cap(r_max): real vs placebo",
              "OC (median, carrier space)", "read-space cap(r) curves (rank-to-audit)",
              "|margin| (median)"]
    for a, t in zip([ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1]], titles):
        a.set_title(t); a.legend(fontsize=6)
        if a is not ax[1, 0]:
            a.set_xscale("log"); a.invert_xaxis(); a.set_xlabel("keep (log)")
        else:
            a.set_xscale("log"); a.set_xlabel("monitor rank r (read space)")
    ax[0, 2].set_yscale("log")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Stage 1 v2: carrier-space evasion geometry")
    ap.add_argument("--model", default="qwen-3b")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="hf:microsoft/cats_vs_dogs")
    ap.add_argument("--out", default="runs_stage1/default")
    ap.add_argument("--n-images", type=int, default=64)
    ap.add_argument("--n-monitor", type=int, default=144)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--r-max", type=int, default=8)
    ap.add_argument("--r-frozen", type=int, default=8)
    ap.add_argument("--read-ranks", default=",".join(map(str, READ_RANKS)))
    ap.add_argument("--n-check", type=int, default=4, help="first-order validity checks per cell")
    ap.add_argument("--read-layer-frac", type=float, default=0.70)
    ap.add_argument("--query-layer", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--keeps", default="")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--no-staleness", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    keeps = [float(x) for x in args.keeps.split(",")] if args.keeps else list(DEFAULT_KEEPS)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    read_ranks = [int(x) for x in args.read_ranks.split(",") if x.strip()]
    os.makedirs(args.out, exist_ok=True)

    if args.self_check:
        print("[self-check] Stage 1 v2 mock (no torch) ...")
        R = run_stage1(Stage1Mock(n=48), keeps=(1.0, 0.5, 0.25, 0.125), methods=("evasive", "infoloss"),
                       r_max=8, n_images=48, read_ranks=read_ranks, verbose=False)
        Ev, Il = R["summary"]["methods"]["evasive"], R["summary"]["methods"]["infoloss"]
        assert abs(Ev["decomposition"]["recon_error"]) < 1e-9, "decomposition must be exact"
        assert Ev["decomposition"]["log_T"]["lo"] > 0, "evasive: tax should fall with compression"
        assert Ev["slopes"]["A"]["hi"] < 0, "evasive: alignment should fall with compression"
        assert Ev["placebo_diffs"]["A_minus_pl_out"]["hi"] < 0, "evasive: should beat the placebo"
        assert Il["decomposition"]["log_m"]["lo"] > 0, "infoloss: margin slope should be positive"
        assert abs(Il["decomposition"]["log_T"]["slope"]) < 0.02, "infoloss: tax should be flat"
        print("[self-check] assertions PASS -- per-image decomposition, CIs, placebo contrasts OK")
    else:
        from csc_vlm.data import load_images
        from csc_vlm.evasion import Stage1VLM
        from csc_vlm.stage0 import Stage0Config
        from csc_vlm.vlm import VLMHandle
        images, labels = load_images(args.dataset, per_class=args.per_class, seed=args.seed)
        cfg = Stage0Config(model=args.model, model_id=args.model_id, dataset=args.dataset,
                           read_layer_frac=args.read_layer_frac, query_layer=args.query_layer,
                           device=args.device, dtype=args.dtype, n_eval=args.n_images,
                           n_monitor=args.n_monitor, seed=args.seed)
        handle = VLMHandle.from_pretrained(args.model, model_id=args.model_id, dtype=args.dtype,
                                           device=args.device, single_device=True,
                                           read_layer_frac=args.read_layer_frac,
                                           query_layer=args.query_layer)
        coll = Stage1VLM(cfg, images, labels, handle=handle, seed=args.seed)
        acc0 = coll.calibrate_polarity()
        print(f"[stage1] keep=1 accuracy (16 imgs) = {acc0:.2f}")
        R = run_stage1(coll, keeps=keeps, methods=methods, r_max=args.r_max, r_frozen=args.r_frozen,
                       read_ranks=read_ranks, n_images=args.n_images, n_check=args.n_check,
                       staleness=not args.no_staleness)
    R["git"] = _git_rev()
    R["args"] = vars(args)
    with open(os.path.join(args.out, "stage1_results.json"), "w") as f:
        json.dump(R, f, indent=1, default=_jsonable)
    make_figure(R, os.path.join(args.out, "stage1.png"))
    print(format_stage1(R["summary"]))
    print(f"\nsaved: {args.out}/stage1_results.json, {args.out}/stage1.png")
    print("Re-analyse at another rank without re-running:  python scripts/analyze_stage1.py "
          f"{args.out}/stage1_results.json --r 4")


if __name__ == "__main__":
    main()
