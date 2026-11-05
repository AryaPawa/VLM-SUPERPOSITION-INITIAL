#!/usr/bin/env python3
"""
run_stage0.py  --  Stage 0: forward-passes-only VLM sanity (plan Phase 5.1 / E2 pilot)

Forward passes ONLY (no attack, no gradients). Per (method, keep) it logs behaviour
accuracy + margin, probe AUROC (frozen & refit), class separation d', and d_eff, at
BOTH the last-token read and the visual-span read, plus the readout-only and
random-prune controls.

Quick reference
---------------
  # logic self-check on CPU, no torch/model/download (seconds):
  python scripts/run_stage0.py --self-check

  # real run (H100), cats vs dogs, LLaVA-1.5-7B (~<1 h):
  python scripts/run_stage0.py --model llava --dataset hf:microsoft/cats_vs_dogs \
         --n-eval 256 --n-monitor 144 --out runs_stage0/llava_cats

  # safety-facing concept:
  python scripts/build_person_dataset.py --download --output data/concept
  python scripts/run_stage0.py --model llava --dataset coco_person --out runs_stage0/llava_person

Reads what Stage 1 needs: whether the concept survives compression (R1), whether the
prune d_eff collapse is a read artifact (readout-only control), the frozen-vs-refit
staleness gap, and the random-pruning null. v0.4 untouched; p2-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from csc_vlm.stage0 import (Stage0Mock, Stage0VLM, Stage0Config, run_stage0,
                            DEFAULT_KEEPS, METHODS)  # noqa: E402

ALL_METHODS = list(METHODS) + ["readout_only"]


def resolve_behaviour(dataset_spec: str) -> str:
    s = dataset_spec.lower()
    if "person" in s or "coco_person" in s:
        return "coco_person"
    return "cats_vs_dogs"


def _series(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], float)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def print_summary(R):
    print("\n" + "=" * 86)
    print(" STAGE 0  --  forward-only VLM sanity   config:", R["config"])
    print("=" * 86)
    keeps = R["keeps"]
    for method in R["methods"]:
        rows = R[method]
        acc = _series(rows, "accuracy")
        la_r = _series(rows, "last_auroc_refit"); la_f = _series(rows, "last_auroc_frozen")
        sa_r = _series(rows, "span_auroc_refit")
        ld = _series(rows, "last_d_eff"); sd = _series(rows, "span_d_eff")
        print(f"\n[{method}]  keep: " + " ".join(f"{k:>5.2f}" for k in keeps))
        print("  accuracy        " + " ".join(f"{v:>5.2f}" for v in acc))
        print("  AUROC last refit" + " ".join(f"{v:>5.2f}" for v in la_r))
        print("  AUROC last froz " + " ".join(f"{v:>5.2f}" for v in la_f))
        print("  AUROC span refit" + " ".join(f"{v:>5.2f}" for v in sa_r))
        print("  d_eff last      " + " ".join(f"{v:>5.1f}" for v in ld))
        print("  d_eff span      " + " ".join(f"{v:>5.1f}" for v in sd))

    def ret(method, key):
        rows = R[method]; s = _series(rows, key)
        return s[-1] / (s[0] + 1e-9)

    print("\n" + "-" * 86)
    print(" READOUTS")
    if "merge" in R:
        print(f"  R1 (concept survival, merge): accuracy {ret('merge','accuracy'):.2f}x, "
              f"last-AUROC {ret('merge','last_auroc_refit'):.2f}x of full "
              f"(≈1 ⇒ concept preserved ⇒ a later effect is not just a worse model)")
    if "prune" in R and "readout_only" in R:
        sd_prune = _series(R["prune"], "span_d_eff")
        sd_ro = _series(R["readout_only"], "span_d_eff")
        ld_prune = _series(R["prune"], "last_d_eff")
        print(f"  Read artifact: span d_eff  prune {sd_prune[0]:.1f}->{sd_prune[-1]:.1f}, "
              f"readout-only {sd_ro[0]:.1f}->{sd_ro[-1]:.1f}; last d_eff prune "
              f"{ld_prune[0]:.1f}->{ld_prune[-1]:.1f}")
        print("     (if span d_eff collapses under readout-only too ⇒ the collapse is a")
        print("      READ artifact, not evasion; the last-token read is the fix.)")
    if "prune" in R:
        la_r = _series(R["prune"], "last_auroc_refit"); la_f = _series(R["prune"], "last_auroc_frozen")
        gap = float(np.nanmax(la_r - la_f))
        print(f"  Staleness (prune, last): max(refit-frozen AUROC) = {gap:+.2f} "
              f"(large ⇒ frozen monitor is just stale, not truly evaded)")
    if "random_prune" in R and "prune" in R:
        print(f"  G1 null: random-prune accuracy {ret('random_prune','accuracy'):.2f}x vs "
              f"prune {ret('prune','accuracy'):.2f}x "
              f"(similar ⇒ effect is about how many tokens, not which)")
    print("=" * 86)
    print(" Next: if the concept survives (R1≈1) and the last-token read stays full-rank")
    print(" while refit AUROC still moves, build Stage 1 (evasion.py: Jacobian OC) on the")
    print(" last-token read. If accuracy collapses with compression, the effect is R1 and")
    print(" the matched-accuracy control must lead.")
    print("=" * 86 + "\n")


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def make_figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keeps = R["keeps"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    def plot(a, key, title, ylab, methods=None):
        for m in (methods or R["methods"]):
            if m in R:
                a.plot(keeps, _series(R[m], key), "o-", label=m)
        a.set_xlabel("keep (←more compression)"); a.set_ylabel(ylab)
        a.invert_xaxis(); a.set_title(title); a.legend(fontsize=7)

    plot(ax[0, 0], "accuracy", "Accuracy (R1: concept survival)", "acc")
    plot(ax[0, 1], "last_auroc_refit", "Last-token AUROC (refit monitor)", "AUROC")
    plot(ax[0, 2], "span_auroc_refit", "Span AUROC (refit monitor)", "AUROC")
    plot(ax[1, 0], "last_d_eff", "d_eff (last-token read)", "d_eff")
    plot(ax[1, 1], "span_d_eff", "d_eff (span read) — readout-only = artifact test", "d_eff")

    # staleness panel: frozen vs refit (last) for prune
    a = ax[1, 2]
    if "prune" in R:
        a.plot(keeps, _series(R["prune"], "last_auroc_refit"), "o-", label="prune refit")
        a.plot(keeps, _series(R["prune"], "last_auroc_frozen"), "s--", label="prune frozen")
    a.set_xlabel("keep (←more compression)"); a.set_ylabel("AUROC")
    a.invert_xaxis(); a.set_title("Staleness: frozen vs refit (last)"); a.legend(fontsize=7)

    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Stage 0: forward-only VLM sanity")
    ap.add_argument("--model", default="llava")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="hf:microsoft/cats_vs_dogs")
    ap.add_argument("--out", default="runs_stage0/default")
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--n-monitor", type=int, default=144)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--read-layer-frac", type=float, default=0.70)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--keeps", default="", help="comma list; default = 8 log-spaced")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="subset of merge,prune,random_prune (readout_only always added)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true",
                    help="run the mock collector end-to-end on CPU (no torch), then exit")
    args = ap.parse_args()

    keeps = ([float(x) for x in args.keeps.split(",")] if args.keeps else list(DEFAULT_KEEPS))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    os.makedirs(args.out, exist_ok=True)

    if args.self_check:
        print("[self-check] mock collector (no torch); validating metrics + controls ...")
        coll = Stage0Mock(d=48, n_eval=200, n_monitor=120, seed=0)
        R = run_stage0(coll, keeps=keeps, methods=methods)
        # sanity assertions on the fixture's KNOWN effects
        acc_merge = _series(R["merge"], "accuracy")
        acc_prune = _series(R["prune"], "accuracy")
        sd_ro = _series(R["readout_only"], "span_d_eff")
        assert acc_merge[0] >= acc_merge[-1] - 0.05, "merge accuracy should not rise w/ compression"
        assert acc_prune[-1] < acc_merge[-1] + 0.05, "prune should lose more accuracy than merge"
        assert sd_ro[-1] < sd_ro[0] * 0.5, "readout-only should collapse span d_eff (artifact)"
        print("[self-check] assertions PASS — metrics, controls, orchestration OK")
    else:
        from csc_vlm.data import load_images
        behav = resolve_behaviour(args.dataset)
        print(f"[stage0] loading dataset '{args.dataset}' (behaviour='{behav}') ...")
        images, labels = load_images(args.dataset, per_class=args.per_class, seed=args.seed)
        cfg = Stage0Config(model=args.model, model_id=args.model_id, dataset=behav,
                           read_layer_frac=args.read_layer_frac, device=args.device,
                           load_4bit=args.load_4bit, n_eval=min(args.n_eval, len(images)),
                           n_monitor=args.n_monitor, seed=args.seed)
        coll = Stage0VLM(cfg, images, labels)
        acc0 = _calibrate_polarity(coll)
        print(f"[stage0] polarity-calibrated accuracy at keep=1.0: {acc0:.2f} "
              f"(pos='{coll.behaviour.pos_word}' id={coll._pos_id}, "
              f"neg='{coll.behaviour.neg_word}' id={coll._neg_id})")
        if acc0 < 0.6:
            print("  [warn] uncompressed accuracy is low — verify the answer tokens / prompt "
                  "for this model family before trusting the run.")
        R = run_stage0(coll, keeps=keeps, methods=methods)

    out_json = os.path.join(args.out, "stage0_results.json")
    out_png = os.path.join(args.out, "stage0.png")
    with open(out_json, "w") as f:
        json.dump(R, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    make_figure(R, out_png)
    print_summary(R)
    print(f"saved: {out_json}\nsaved: {out_png}")


def _calibrate_polarity(coll, n=16):
    """One-time check at keep=1.0: if the answer-token polarity is flipped (accuracy
    < 0.5), swap pos/neg ids so 'correct' and later margins have the right sign."""
    idx = coll._eval_idx[:n]
    pack = coll._collect_idx(1.0, "merge", idx)
    acc = float(np.mean(pack["correct"]))
    if acc < 0.5:
        coll._pos_id, coll._neg_id = coll._neg_id, coll._pos_id
        pack = coll._collect_idx(1.0, "merge", idx)
        acc = float(np.mean(pack["correct"]))
    return acc


if __name__ == "__main__":
    main()