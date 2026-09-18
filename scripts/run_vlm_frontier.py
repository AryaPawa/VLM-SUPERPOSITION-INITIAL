#!/usr/bin/env python3
"""
run_vlm_frontier.py -- Phase 3: compression-safety frontier on a real VLM.

Modes:
  --mock                 synthetic Variant-B geometry, CPU, seconds. Validates the
                         whole control pipeline with no torch/transformers.
  --model {llava,qwen}   real model via transformers. Needs a GPU (H100 for the
                         full sweep). Point --image-dir at a labelled image folder
                         (subfolders = classes, or a labels.csv).
  --smoke                with --model: load the model and do ONE collect + ONE
                         attack to validate the plumbing on a minimal GPU (use
                         --load-4bit and the 3B model id) before the cluster run.

Outputs (results/): vlm_frontier.csv, figure_vlm.png

Examples:
  python scripts/run_vlm_frontier.py --mock
  python scripts/run_vlm_frontier.py --model qwen --model-id Qwen/Qwen2.5-VL-3B-Instruct \
         --image-dir data/concept --load-4bit --smoke
  python scripts/run_vlm_frontier.py --model llava --image-dir data/concept   # H100
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from csc.vlm import (VLMConfig, MockVLMBackend, HFVLMBackend,   # noqa: E402
                     vlm_frontier, evaluate_vlm_frontier, fit_monitor)


def _load_labeled_images(image_dir):
    """Minimal loader: image_dir/<class>/*.jpg -> (images, binary labels).
    Class-name sorting: first class = 0, rest = 1 (binary concept present)."""
    from PIL import Image
    classes = sorted(d for d in os.listdir(image_dir)
                     if os.path.isdir(os.path.join(image_dir, d)))
    images, labels = [], []
    for ci, cls in enumerate(classes):
        for fn in os.listdir(os.path.join(image_dir, cls)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                images.append(Image.open(os.path.join(image_dir, cls, fn)).convert("RGB"))
                labels.append(0 if ci == 0 else 1)
    return images, np.asarray(labels)


def _make_synthetic_images(n=24, size=448, seed=0):
    """Two visually-distinct classes on a noisy gray background (red block
    top-left = class 0, green block bottom-right = class 1). Enough of a real
    concept to fit the monitor, so the FULL pipeline is exercised. For PLUMBING
    validation only — swap in real labelled images (COCO/POPE/etc.) for science."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    s = size // 2
    for i in range(n):
        arr = rng.integers(90, 130, (size, size, 3)).astype("uint8")   # gray noise
        lab = i % 2
        if lab == 0:
            arr[10:10 + s, 10:10 + s] = [220, 40, 40]                  # red, top-left
        else:
            arr[size - s - 10:size - 10, size - s - 10:size - 10] = [40, 200, 60]  # green, bottom-right
        imgs.append(Image.fromarray(arr, "RGB"))
        labels.append(lab)
    return imgs, np.asarray(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", choices=["llava", "qwen"])
    ap.add_argument("--model-id", default="")
    ap.add_argument("--image-dir", default="")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="use auto-generated images instead of --image-dir (plumbing test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.mock or not args.model:
        cfg = VLMConfig(model="mock", n_pairs=64, monitor_rank=8, mock_d=64, mock_F_base=48)
        backend = MockVLMBackend(cfg)
        deltas = None   # auto-calibrated from the geometry
        steps = 40
        print("[vlm] MOCK backend (synthetic Variant-B geometry, CPU)")
    else:
        cfg = VLMConfig(model=args.model, model_id=args.model_id, device=args.device,
                        load_4bit=args.load_4bit, image_dir=args.image_dir,
                        n_pairs=8 if args.smoke else 16, monitor_rank=8)
        have_dir = bool(args.image_dir) and os.path.isdir(args.image_dir)
        if args.synthetic or not have_dir:
            if not have_dir and not args.synthetic:
                print(f"[vlm] no valid --image-dir ('{args.image_dir}') -> "
                      f"using SYNTHETIC images (plumbing validation only)")
            images, labels = _make_synthetic_images(n=24 if args.smoke else 64)
        else:
            images, labels = _load_labeled_images(args.image_dir)
        print(f"[vlm] {args.model} backend | {len(images)} images | "
              f"{'4bit' if args.load_4bit else 'fp16'} | read_layer_frac={cfg.read_layer_frac}")
        backend = HFVLMBackend(cfg, images, labels)
        if args.smoke:
            from csc.vlm import measured_geometry
            H, y = backend.collect(1.0, cfg.n_pairs, seed=0)
            w_b, U = fit_monitor(H, y, cfg.monitor_rank)
            geo = measured_geometry(H, w_b, U, cfg.beta)
            print(f"[smoke] read_dim={backend.read_dim()} H={H.shape} "
                  f"read_layer={backend.read_layer} labels={y.tolist()}")
            print(f"[smoke] S(obf_cost)={geo['obf_cost']:.3f} Γ={geo['gamma_r']:.3f} "
                  f"‖a‖={geo['leverage']:.3f} d_eff={geo['d_eff']:.1f}")
            print("[smoke] OK — model loads, hook fires, activations + geometry OK.")
            print("[smoke] Next: build data/concept with real labelled images, run the")
            print("[smoke]       full frontier on the H100s.")
            return 0
        deltas = None   # auto-calibrated from the geometry
        steps = 40

    rows, oracle, (w_b, U) = vlm_frontier(backend, cfg, deltas, steps=steps, seed=0)
    df = pd.DataFrame([{k: r[k] for k in
                        ("delta", "keep", "compression", "S", "S_geom", "lam",
                         "gamma_r", "leverage", "d_eff", "at_bound")} for r in rows])
    df.to_csv(os.path.join(args.outdir, "vlm_frontier.csv"), index=False)

    ev = evaluate_vlm_frontier(rows)
    print("\n=== G-VLM (frontier) ===")
    for k, v in ev["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    st = ev["stats"]
    print(f"  spearman(δ, compression)      = {st['spearman(delta, compression)']:+.3f}")
    print(f"  spearman(compression, ‖a‖)    = {st['spearman(compression, leverage)']:+.3f}  (H3 mechanism)")
    print(f"  median |S − δ| (tracking)     = {st['median|S-delta|']:.3f}")
    print(f"  Prop-1 median S/S_geom        = {st['prop1_median_ratio_S/S_geom']:.3f}  "
          f"(≈1 ⇒ S = β/(Γ·‖a‖) holds on real activations)")
    print(f"  VERDICT: {'PASSED' if ev['passed'] else 'FAILED'}   (model evals: {oracle.evals})")

    # figure: frontier, shadow price, mediators, Prop-1 check
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Phase 3 — compression–safety frontier ({cfg.model})",
                 fontsize=12, fontweight="bold")
    d = np.array([r["delta"] for r in rows]); comp = np.array([r["compression"] for r in rows])
    S = np.array([r["S"] for r in rows]); Sg = np.array([r["S_geom"] for r in rows])
    lam = np.array([r["lam"] for r in rows]); lev = np.array([r["leverage"] for r in rows])
    gam = np.array([r["gamma_r"] for r in rows])

    a = ax[0, 0]; o = np.argsort(S)
    a.plot(S[o], comp[o], "o-", color="#1f77b4")
    a.set_xlabel("achieved safety S"); a.set_ylabel("compression 1/keep")
    a.set_title("(a) frontier: safety costs compression"); a.grid(alpha=0.3)
    b = ax[0, 1]; b.plot(d, lam, "s-", color="#2ca02c")
    b.set_xlabel("safety floor δ"); b.set_ylabel("shadow price λ")
    b.set_title("(b) price of safety"); b.grid(alpha=0.3)
    c = ax[1, 0]; c.plot(comp, lev, "o-", label="‖a‖ (leverage)", color="#d62728")
    c2 = c.twinx(); c2.plot(comp, gam, "^--", label="Γ_r", color="#7f7f7f")
    c.set_xlabel("compression 1/keep"); c.set_ylabel("‖a‖", color="#d62728")
    c2.set_ylabel("Γ_r", color="#7f7f7f")
    c.set_title("(c) mediators vs compression (H3)"); c.grid(alpha=0.3)
    e = ax[1, 1]; deff = np.array([r["d_eff"] for r in rows])
    e.plot(comp, deff, "o-", color="#9467bd")
    e.set_xlabel("compression 1/keep"); e.set_ylabel("read dim d_eff")
    e.set_title("(d) read dimensionality vs compression"); e.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = os.path.join(args.outdir, "figure_vlm.png")
    fig.savefig(figpath, dpi=130); plt.close(fig)
    print(f"\n[vlm] wrote {os.path.join(args.outdir, 'vlm_frontier.csv')}")
    print(f"[vlm] wrote {figpath}")
    return 0 if ev["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())