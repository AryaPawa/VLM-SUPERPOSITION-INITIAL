#!/usr/bin/env python3
"""
prepare_data.py -- download + cache the dataset on a node that HAS internet
(e.g. the login node), so the GPU job can run offline on a compute node.

Usage:
  python scripts/prepare_data.py                          # default dataset
  python scripts/prepare_data.py --dataset hf:microsoft/cats_vs_dogs
  python scripts/prepare_data.py --dataset hf:cifar10 --per-class 300
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csc_vlm import data as datamod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=datamod.DEFAULT_DATASET)
    ap.add_argument("--per-class", type=int, default=200)
    args = ap.parse_args()

    # coco_person needs building (download COCO + sort) rather than a plain load.
    if args.dataset.startswith("coco_person"):
        parts = args.dataset.split(":")
        root = parts[1] if len(parts) > 1 and parts[1] else "data/concept"
        if not os.path.isdir(root):
            print(f"[prepare] coco_person not found at '{root}' -> building from COCO")
            build = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "build_person_dataset.py")
            cmd = f'"{sys.executable}" "{build}" --download --output "{root}" ' \
                  f'--max-per-class {max(args.per_class, 200)}'
            print(f"[prepare] $ {cmd}")
            if os.system(cmd) != 0:
                raise SystemExit("[prepare] build_person_dataset.py failed")
        else:
            print(f"[prepare] coco_person already built at '{root}'")

    print(f"[prepare] resolving {args.dataset} (HF specs download + cache to ~/.cache/huggingface)")
    imgs, y = datamod.load_images(args.dataset, per_class=args.per_class, seed=0)
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    print(f"[prepare] OK -> {len(imgs)} images (class0={n0}, class1={n1})")
    print("[prepare] cache is warm; the GPU job can now run without internet.")


if __name__ == "__main__":
    main()
