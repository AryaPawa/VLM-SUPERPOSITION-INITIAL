#!/usr/bin/env python3
"""
run_x1b_toy.py  --  X1b trained toy (CPU, numpy; ~6-8 min full, ~1 min --quick)

Trains small patch-token models end to end at every budget and measures, EXACTLY,
the terms of  log OC = log m - log||g|| + log T  plus the geometry-matched
and matched placebos, the full-capacity (Mahalanobis-optimal) detector cost and
the canonical probe's incidental detection of the plain attack. It answers the
question X1 could not: under EMERGENT geometry, does compression raise or lower
the monitor's obfuscation tax, and does that depend on the regime (dominant vs
subtle behaviour) and the method (pool vs drop)?

Every monitor rank (1, 2, 4) is printed; r = 1 is the canonical probe. Each
slope also shows [seeds a/n]: how many separately trained seeds share the
pooled sign (the image-level CI ignores training variance -- trust a sign only
if the seeds agree).

  python scripts/run_x1b_toy.py --quick          # 1 seed, fewer steps (sign agreement n/a)
  python scripts/run_x1b_toy.py                  # 3 seeds (paper figure)
  python scripts/run_x1b_toy.py --self-check     # gradient + closed-form checks only

Outputs: toy_x1b_results.json (summary + per-keep tables), toy_x1b.png.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csc_vlm._io import utf8_stdio  # noqa: E402
utf8_stdio()
from csc_vlm.toy_trained import (DEFAULT_REGIMES, LEGACY_REGIMES, ToyCfg,  # noqa: E402
                                 format_summary, self_check, sweep)


def make_figure(S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    r1, rmax = min(S), max(S)
    regimes = list(S[r1])
    fig, ax = plt.subplots(3, len(regimes), figsize=(7 * len(regimes), 13), squeeze=False)
    for c, rname in enumerate(regimes):
        for method, M in S[r1][rname].items():
            pk, pkm = M["per_keep"], S[rmax][rname][method]["per_keep"]
            fr = [p["carrier_dim"] / pk[0]["carrier_dim"] for p in pk]
            line, = ax[0, c].plot(fr, [p["cap_in"] for p in pk], "o-", label=f"{method} real r={r1}")
            ax[0, c].plot(fr, [p["cap_match"] for p in pk], "x:", color=line.get_color(), label=f"{method} matched")
            ax[0, c].plot(fr, [p["cap_in"] for p in pkm], "s--", alpha=0.5, color=line.get_color(),
                          label=f"{method} real r={rmax}")
            ax[2, c].plot(fr, [p["full_cost"] for p in pk], "o-", color=line.get_color(), label=f"{method} full-capacity")
            ax[2, c].plot(fr, [p["incidental_det"] for p in pk], "x:", color=line.get_color(),
                          label=f"{method} incidental det.")
        ax[0, c].set_title(f"{rname}: carrier cap (higher = monitor sees more)")
        ax[2, c].set_title(f"{rname}: |b|/sd(b) and P(probe flags plain attack)")
        for a_ in (ax[0, c], ax[2, c]):
            a_.set_xscale("log"); a_.invert_xaxis(); a_.legend(fontsize=7); a_.set_xlabel("carrier fraction")
        names = list(S[r1][rname])
        x = np.arange(len(names))
        for j, key in enumerate(("log_m", "neg_log_g", "log_T")):
            ax[1, c].bar(x + (j - 1) * 0.27, [S[r1][rname][n]["decomposition"][key]["slope"] for n in names],
                         0.27, label=key)
        ax[1, c].set_xticks(x); ax[1, c].set_xticklabels(names)
        ax[1, c].axhline(0, color="k", lw=0.5); ax[1, c].legend(fontsize=7)
        ax[1, c].set_title(f"signed slopes of log-OC terms, r={r1} (beta>0: falls under compression)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--p-behaviour", type=float, default=0.5,
                    help="behaviour frequency; lower = sparser, more superposed behaviour feature")
    ap.add_argument("--q-vis", type=float, default=0.5, help="P(behaviour visible in a patch)")
    ap.add_argument("--legacy", action="store_true", help="also run the read_bottleneck regime (d_r=8)")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.self_check:
        ok = self_check(verbose=True)
        sys.exit(0 if ok else 1)
    seeds = (0,) if a.quick else tuple(int(s) for s in a.seeds.split(","))
    steps = 800 if a.quick else a.steps
    t = time.time()
    regimes = dict(DEFAULT_REGIMES, **(LEGACY_REGIMES if a.legacy else {}))
    R = sweep(regimes=regimes, seeds=seeds, steps=steps, verbose=True,
              base=ToyCfg(p_behaviour=a.p_behaviour, q_vis=a.q_vis))
    for r in R["ranks"]:
        print(format_summary(R["summary"], r) + "\n")
    os.makedirs(a.out, exist_ok=True)
    slim = {"config": R["config"], "keeps": R["keeps"], "ranks": R["ranks"], "summary": R["summary"],
            "seeds": list(seeds), "steps": steps, "regimes": regimes}
    with open(os.path.join(a.out, "toy_x1b_results.json"), "w") as f:
        json.dump(slim, f, indent=1, default=float)
    make_figure(R["summary"], os.path.join(a.out, "toy_x1b.png"))
    print(f"\nsaved {a.out}/toy_x1b_results.json, {a.out}/toy_x1b.png  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()