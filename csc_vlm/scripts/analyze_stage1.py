#!/usr/bin/env python3
"""
analyze_stage1.py  --  re-analyse a saved stage1_results.json at any monitor rank
(1..r_max) without touching the GPU. Prints the same two-sided summary.

  python scripts/analyze_stage1.py runs_stage1/qwen3b_cats/stage1_results.json --r 4
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csc_vlm.evasion import analyze_stage1, format_stage1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--r", type=int, default=0, help="monitor rank (default: r_max of the run)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    with open(a.results) as f:
        R = json.load(f)
    r = a.r or R["r_max"]
    if not 1 <= r <= R["r_max"]:
        raise SystemExit(f"--r must be in 1..{R['r_max']}")
    print(format_stage1(analyze_stage1(R, r=r, B=a.bootstrap)))


if __name__ == "__main__":
    main()
