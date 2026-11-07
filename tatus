#!/usr/bin/env python3
"""
run_x1b_toy.py  --  X1b trained toy (CPU, numpy; ~6-8 min full, ~1 min --quick)

Trains small patch-token models end to end at every budget and measures, EXACTLY,
the terms of  log OC = log m - log||g|| + log T  plus the geometry-matched
placebo. It answers the question X1 could not: under EMERGENT geometry, does
compression raise or lower the monitor's obfuscation tax, and does that depend
on the regime (read-bottleneck vs carrier-bottleneck) and the method (pool vs
drop)?

  python scripts/run_x1b_toy.py --quick          # 1 seed, fewer steps
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
from csc_vlm.toy_trained import ToyCfg, format_summary, self_check, sweep  # noqa: E402


def make_figure(S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regimes = list(S)
    fig, ax = plt.subplots(2, len(regimes), figsize=(7 * len(regimes), 9), squeeze=False)
    for c, rname in enumerate(regimes):
        for method, M in S[rname].items():
            pk = M["per_keep"]
            fr = [p["carrier_dim"] / pk[0]["carrier_dim"] for p in pk]
            ax[0, c].plot(fr, [p["cap_in"] for p in pk], "o-", label=f"{method} behaviour")
            ax[0, c].plot(fr, [p["cap_pl"] for p in pk], "x:", label=f"{method} placebo")
        ax[0, c].set_title(f"{rname}: captured fraction cap(r) (higher = monitor sees more)")
        ax[0, c].set_xscale("log"); ax[0, c].invert_xaxis(); ax[0, c].legend(fontsize=7)
        names = list(S[rname])
        import numpy as np
        x = np.arange(len(names))
        for j, key in enumerate(("log_m", "neg_log_g", "log_T")):
            ax[1, c].bar(x + (j - 1) * 0.27, [S[rname][n]["decomposition"][key]["slope"] for n in names],
                         0.27, label=key)
        ax[1, c].set_xticks(x); ax[1, c].set_xticklabels(names)
        ax[1, c].axhline(0, color="k", lw=0.5); ax[1, c].legend(fontsize=7)
        ax[1, c].set_title("signed slopes of log-OC terms (beta>0: falls under compression)")
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
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    if a.self_check:
        ok = self_check(verbose=True)
        sys.exit(0 if ok else 1)
    seeds = (0,) if a.quick else tuple(int(s) for s in a.seeds.split(","))
    steps = 800 if a.quick else a.steps
    t = time.time()
    R = sweep(seeds=seeds, steps=steps, verbose=True,
              base=ToyCfg(p_behaviour=a.p_behaviour, q_vis=a.q_vis))
    print(format_summary(R["summary"], R["ranks"][-1]))
    os.makedirs(a.out, exist_ok=True)
    slim = {"config": R["config"], "keeps": R["keeps"], "ranks": R["ranks"], "summary": R["summary"],
            "seeds": list(seeds), "steps": steps}
    with open(os.path.join(a.out, "toy_x1b_results.json"), "w") as f:
        json.dump(slim, f, indent=1, default=float)
    make_figure(R["summary"], os.path.join(a.out, "toy_x1b.png"))
    print(f"\nsaved {a.out}/toy_x1b_results.json, {a.out}/toy_x1b.png  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
