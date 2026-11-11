#!/usr/bin/env python3
"""
analyze_attack.py  --  re-analyse one or more saved attack_results.json files (no GPU)

  python scripts/analyze_attack.py runs_attack/*/attack_results.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csc_vlm._io import utf8_stdio  # noqa: E402
utf8_stdio()
from csc_vlm.attack import save_json  # noqa: E402
from csc_vlm.attack_analysis import analyze_attack, format_attack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--json-out", action="store_true", help="also write attack_summary.json next to each file")
    a = ap.parse_args()
    for path in a.results:
        with open(path) as f:
            R = json.load(f)
        S = analyze_attack(R)
        print(f"== {path}  model={R.get('config', {}).get('model')}  dataset={R.get('config', {}).get('dataset')}")
        print(format_attack(S) + "\n")
        if a.json_out:
            save_json(S, os.path.join(os.path.dirname(path), "attack_summary.json"))


if __name__ == "__main__":
    main()
