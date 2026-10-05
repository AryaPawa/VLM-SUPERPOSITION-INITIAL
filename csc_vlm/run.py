#!/usr/bin/env python3
"""
run.py -- Compression-safety frontier on real VLMs (standalone entry point).

Quick reference
---------------
  # 0) sanity on CPU, no model, no download (seconds):
  python run.py --model mock

  # 1) plumbing self-check (validates load + hook + compression):
  python run.py --model qwen --model-id Qwen/Qwen2.5-VL-3B-Instruct --load-4bit --self-check

  # 2) full grid-search frontier (token merging, default):
  python run.py --model llava --dataset hf:microsoft/cats_vs_dogs --out runs/llava_merge
  python run.py --model qwen  --dataset hf:microsoft/cats_vs_dogs --out runs/qwen_merge

  # 3) full grid-search frontier with token pruning:
  python run.py --model llava --dataset hf:microsoft/cats_vs_dogs --compression-method prune --out runs/llava_prune
  python run.py --model qwen  --dataset hf:microsoft/cats_vs_dogs --compression-method prune --out runs/qwen_prune

  # 4) safety-facing concept (build once, then run):
  python scripts/build_person_dataset.py --download --output data/concept
  python run.py --model llava --dataset coco_person --compression-method merge --out runs/llava_merge_person
  python run.py --model llava --dataset coco_person --compression-method prune --out runs/llava_prune_person

  # 5) resume after a pre-emption (skips completed grid points):
  python run.py --model llava --dataset hf:microsoft/cats_vs_dogs --out runs/llava_merge --resume

Dataset specs: synthetic | folder:/path (classA/ classB/ subdirs) |
               coco_person[:/path] | hf:<name>[:label=..:pos=..]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csc_vlm.backends import VLMConfig                          # noqa: E402
from csc_vlm.runner import run_frontier, self_check            # noqa: E402
from csc_vlm.data import DEFAULT_DATASET                        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="CSC VLM compression-safety frontier (grid search)")
    ap.add_argument("--model",
                    default="mock",
                    help="Short model name: 'mock' (CPU), 'llava', 'qwen', 'llava-next', "
                         "'internvl', 'smolvlm', 'paligemma', or any free-form name "
                         "when paired with --model-id <hf_id>")
    ap.add_argument("--model-id", default="", help="HF model id (defaults per lineage)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="synthetic | folder:/path | coco_person[:/path] | hf:<name>")
    ap.add_argument("--out", default="runs/default", help="output/checkpoint dir")
    ap.add_argument("--resume", action="store_true",
                    help="continue from --out checkpoint, skipping completed grid points")
    ap.add_argument("--self-check", action="store_true",
                    help="cheap plumbing validation (load+hook+compression), then exit")
    ap.add_argument("--synthetic", action="store_true",
                    help="force synthetic images (overrides --dataset)")
    ap.add_argument("--compression-method", choices=["merge", "prune"], default="merge",
                    help="token merging (default) or activation-magnitude pruning")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--read-layer-frac", type=float, default=0.70)
    ap.add_argument("--n-pairs", type=int, default=24,
                    help="activations collected per grid point evaluation")
    ap.add_argument("--monitor-rank", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = VLMConfig(
        model=args.model,
        model_id=args.model_id,
        device=args.device,
        load_4bit=args.load_4bit,
        read_layer_frac=args.read_layer_frac,
        n_pairs=args.n_pairs,
        monitor_rank=args.monitor_rank,
        compression_method=args.compression_method,
    )
    # the mock is free on CPU -> use more samples for a clean, low-noise demo
    if args.model == "mock" and args.n_pairs == 24:
        cfg.n_pairs = 64

    dataset = "synthetic" if args.synthetic else args.dataset

    if args.self_check:
        ok = self_check(cfg, dataset="synthetic", seed=args.seed)
        return 0 if ok else 1

    ev = run_frontier(cfg, dataset, args.out,
                      resume=args.resume,
                      per_class=args.per_class,
                      seed=args.seed)
    return 0 if ev["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
