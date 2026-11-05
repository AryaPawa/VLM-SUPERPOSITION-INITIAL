#!/usr/bin/env python3
"""
run_stage1.py  --  Stage 1: the per-example evasion metric on real VLMs (plan Phase 2.2/2.3)

Measures, per image and keep, the carrier-space Jacobian evasion metric:
  Gamma_r (escape fraction), OC (min evading perturbation), and the decomposition
  log OC = log margin - log||g_b|| - log Gamma_r, with a random-direction placebo null
  and a frozen-vs-refit monitor staleness check.

Quick reference
---------------
  # logic self-check on CPU, no torch (seconds):
  python scripts/run_stage1.py --self-check

  # real run (GPU). Use fp32/bf16 (NOT fp16) for gradients. Prefer small models on 16 GB:
  python scripts/run_stage1.py --model qwen-3b --dataset hf:microsoft/cats_vs_dogs \
         --dtype float32 --n-images 96 --rank 8 --out runs_stage1/qwen3b_cats

Verdicts (plan): beta_OC >= 0.25 per log-unit (H2); Gamma term >= 50% of the OC slope
for merge (H3, evasion not info-loss); the real Gamma trend beats the placebo (null).
Stage 2 (not here) is the PGD validity gate that promotes OC to "safety".

v0.4 untouched; p2-only. Reuses Stage0VLM (backends.py unmodified).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from csc_vlm.evasion import run_stage1, Stage1Mock, DEFAULT_KEEPS, METHODS  # noqa: E402


def _resolve_behaviour(dataset_spec: str) -> str:
    s = dataset_spec.lower()
    return "coco_person" if ("person" in s) else "cats_vs_dogs"


def _rows(R, m, key):
    return np.array([r[key] for r in R[m]["rows"]], float)


def print_summary(R):
    def mark(ok): return "PASS" if ok else "FAIL"
    keeps = R["keeps"]
    print("\n" + "=" * 84)
    print(" STAGE 1  --  per-example evasion metric   config:", R["config"])
    print("=" * 84)
    for m in R["methods"]:
        g = _rows(R, m, "gamma"); gp = _rows(R, m, "gamma_placebo")
        oc = _rows(R, m, "oc"); mar = _rows(R, m, "margin")
        d = R[m]["decomp"]
        print(f"\n[{m}]  keep: " + " ".join(f"{k:>5.2f}" for k in keeps))
        print("  Gamma_r (real)   " + " ".join(f"{v:>5.2f}" for v in g))
        print("  Gamma_r (placebo)" + " ".join(f"{v:>5.2f}" for v in gp))
        print("  OC (median)      " + " ".join(f"{v:>5.2f}" for v in oc))
        print("  margin (median)  " + " ".join(f"{v:>5.2f}" for v in mar))
        print(f"  -> beta_OC/log-unit = {R[m]['beta_oc_per_logunit']:+.2f} (H2>=0.25) "
              f"[{mark(R[m]['beta_oc_per_logunit'] >= 0.25)}];  "
              f"Gamma-share = {d['share_gamma']:.0%} (H3>=50%) "
              f"[{mark(d['share_gamma'] >= 0.5)}];  "
              f"beats placebo [{mark(R[m]['beats_placebo'])}]")
    print("\n" + "-" * 84)
    print(" Reading: a real effect = OC falls with compression (beta_OC>=0.25), driven")
    print(" mostly by Gamma_r (not margin), and the Gamma trend exceeds the placebo. If")
    print(" margin carries the OC slope instead, the effect is information loss (R1).")
    print(" Next: Stage 2 PGD validity gate (Spearman>=0.3 of OC vs true minimal-eps).")
    print("=" * 84 + "\n")


def make_figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keeps = R["keeps"]; methods = R["methods"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    for m in methods:
        ax[0, 0].plot(keeps, _rows(R, m, "gamma"), "o-", label=m)
    ax[0, 0].plot(keeps, _rows(R, methods[0], "gamma_placebo"), "k:", label="placebo")
    ax[0, 0].set_title("Γ_r (escape) vs keep"); ax[0, 0].set_ylabel("Γ_r")

    for m in methods:
        ax[0, 1].plot(keeps, _rows(R, m, "oc"), "o-", label=m)
    ax[0, 1].set_yscale("log"); ax[0, 1].set_title("OC (median) vs keep — the safety metric")
    ax[0, 1].set_ylabel("OC (log)")

    for m in methods:
        ax[0, 2].plot(keeps, _rows(R, m, "margin"), "o-", label=m)
    ax[0, 2].set_title("Behaviour margin vs keep (R1 confound)"); ax[0, 2].set_ylabel("margin")

    x = np.arange(len(methods))
    gsh = [R[m]["decomp"]["share_gamma"] for m in methods]
    msh = [R[m]["decomp"]["share_margin"] for m in methods]
    ax[1, 0].bar(x - 0.2, gsh, 0.4, label="Γ (evasion)")
    ax[1, 0].bar(x + 0.2, msh, 0.4, label="margin (info-loss)")
    ax[1, 0].axhline(0.5, color="k", ls=":")
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(methods)
    ax[1, 0].set_title("What drives OC (H3)"); ax[1, 0].set_ylabel("share of OC slope")

    for m in methods:
        ax[1, 1].plot(keeps, _rows(R, m, "gamma"), "o-", label=f"{m} refit")
        ax[1, 1].plot(keeps, _rows(R, m, "gamma_frozen"), "s--", label=f"{m} frozen")
    ax[1, 1].set_title("Γ_r refit vs frozen (staleness)"); ax[1, 1].set_ylabel("Γ_r")

    beta = [R[m]["beta_oc_per_logunit"] for m in methods]
    ax[1, 2].bar(x, beta); ax[1, 2].axhline(0.25, color="r", ls=":", label="H2=0.25")
    ax[1, 2].set_xticks(x); ax[1, 2].set_xticklabels(methods)
    ax[1, 2].set_title("β_OC per log-unit (H2)"); ax[1, 2].set_ylabel("β_OC")

    for a in ax.ravel():
        a.set_xlabel("keep (←more compression)"); a.invert_xaxis(); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Stage 1: per-example evasion metric")
    ap.add_argument("--model", default="qwen-3b")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="hf:microsoft/cats_vs_dogs")
    ap.add_argument("--out", default="runs_stage1/default")
    ap.add_argument("--n-images", type=int, default=96)
    ap.add_argument("--n-monitor", type=int, default=144)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--read-layer-frac", type=float, default=0.70)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--keeps", default="")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-staleness", action="store_true",
                    help="skip the frozen-monitor gradients (halves backward passes; "
                         "no refit-vs-frozen staleness panel). Use for large models.")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    staleness = not args.no_staleness

    keeps = ([float(x) for x in args.keeps.split(",")] if args.keeps else list(DEFAULT_KEEPS))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    os.makedirs(args.out, exist_ok=True)

    if args.self_check:
        print("[self-check] mock Jacobians (no torch); validating Γ_r / OC / decomp / placebo ...")
        coll = Stage1Mock(dim=240, rank=args.rank, n=args.n_images, seed=0)
        R = run_stage1(coll, keeps=keeps, methods=methods, rank=args.rank, n_images=args.n_images, staleness=staleness)
        assert R["merge"]["beta_oc_per_logunit"] > 0, "merge OC should fall with compression"
        assert R["merge"]["decomp"]["share_gamma"] > R["prune"]["decomp"]["share_gamma"], \
            "merge should be more evasion-driven than prune"
        assert R["merge"]["beats_placebo"], "real Γ trend should beat the placebo"
        print("[self-check] assertions PASS — metric math + nulls + orchestration OK")
    else:
        if args.dtype == "float16":
            print("[warn] fp16 gradients can underflow (rival R3). Prefer float32 or bfloat16.")
        from csc_vlm.data import load_images
        from csc_vlm.stage0 import Stage0Config
        from csc_vlm.evasion import Stage1VLM
        behav = _resolve_behaviour(args.dataset)
        print(f"[stage1] loading dataset '{args.dataset}' (behaviour='{behav}') ...")
        images, labels = load_images(args.dataset, per_class=args.per_class, seed=args.seed)
        cfg = Stage0Config(model=args.model, model_id=args.model_id, dataset=behav,
                           read_layer_frac=args.read_layer_frac, device=args.device,
                           n_eval=min(args.n_images, len(images)), n_monitor=args.n_monitor,
                           seed=args.seed)
        coll = Stage1VLM(cfg, images, labels, dtype=args.dtype)
        R = run_stage1(coll, keeps=keeps, methods=methods, rank=args.rank, n_images=args.n_images, staleness=staleness)

    out_json = os.path.join(args.out, "stage1_results.json")
    out_png = os.path.join(args.out, "stage1.png")
    with open(out_json, "w") as f:
        json.dump(R, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    make_figure(R, out_png)
    print_summary(R)
    print(f"saved: {out_json}\nsaved: {out_png}")


if __name__ == "__main__":
    main()