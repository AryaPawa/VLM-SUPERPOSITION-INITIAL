#!/usr/bin/env python3
"""
run.py -- Compression-safety frontier on real VLMs (standalone entry point).

Quick reference
---------------
  # 0) sanity on CPU, no model, no download (seconds):
  python run.py --model mock

  # 1) plumbing self-check on a small model (validates load + hook + merge):
  python run.py --model qwen --model-id Qwen/Qwen2.5-VL-3B-Instruct --load-4bit --self-check

  # 2) full frontier on the cluster (auto-downloads the dataset, checkpoints):
  python run.py --model qwen  --dataset hf:microsoft/cats_vs_dogs --out runs/qwen
  python run.py --model llava --dataset hf:microsoft/cats_vs_dogs --out runs/llava

  # safety-facing "person" concept (build once, then run):
  python scripts/build_person_dataset.py --download --output data/concept
  python run.py --model qwen --dataset coco_person --out runs/qwen_person

  # 3) resume after a pre-emption (skips completed deltas):
  python run.py --model qwen --dataset hf:microsoft/cats_vs_dogs --out runs/qwen --resume

Dataset specs: synthetic | folder:/path (classA/ classB/ subdirs) |
               hf:<name>[:label=..:pos=..]   (see csc_vlm/data.py)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csc_vlm.backends import VLMConfig                          # noqa: E402
from csc_vlm.runner import run_frontier, self_check            # noqa: E402
from csc_vlm.data import DEFAULT_DATASET                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="CSC VLM compression-safety frontier")
    ap.add_argument("--model", choices=["mock", "llava", "qwen"], default="mock")
    ap.add_argument("--model-id", default="", help="HF model id (defaults per lineage)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="synthetic | folder:/path | coco_person[:/path] | hf:<name>")
    ap.add_argument("--out", default="runs/default", help="output/checkpoint dir")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="cheap plumbing validation (load+hook+merge), then exit")
    ap.add_argument("--synthetic", action="store_true",
                    help="force synthetic images (overrides --dataset)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--read-layer-frac", type=float, default=0.70)
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--monitor-rank", type=int, default=8)
    ap.add_argument("--n-deltas", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = VLMConfig(model=args.model, model_id=args.model_id, device=args.device,
                    load_4bit=args.load_4bit, read_layer_frac=args.read_layer_frac,
                    n_pairs=args.n_pairs, monitor_rank=args.monitor_rank)
    # the mock is free on CPU -> use more samples for a clean, low-noise demo
    if args.model == "mock" and args.n_pairs == 24:
        cfg.n_pairs = 64
    dataset = "synthetic" if args.synthetic else args.dataset

    if args.self_check:
        ok = self_check(cfg, dataset="synthetic", seed=args.seed)
        return 0 if ok else 1

    ev = run_frontier(cfg, dataset, args.out, resume=args.resume,
                      n_deltas=args.n_deltas, steps=args.steps,
                      per_class=args.per_class, seed=args.seed)
    return 0 if ev["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
