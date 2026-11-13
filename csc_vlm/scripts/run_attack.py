#!/usr/bin/env python3
"""
run_attack.py  --  pipeline v3: iterative monitor-constrained minimum-norm attacks

What it measures (see csc_vlm/attack.py for the definitions):
  OC_free  cheapest perturbation that makes the answer WRONG (label-defined target)
  OC_thr   the same while the rank-1 probe tripwire stays silent
  OC_eq    the same while the probe reading stays (nearly) unchanged (Stage-1-comparable)
  T_true = OC_thr / OC_free, all in a FIXED space at every keep:
    --leaf patch   pre-compression visual-token embeddings (the instrument check)
    --leaf pixel   normalised pixels, clipped to valid images (the R2 test with
                   --selections frozen,live)

  # CPU self-check (synthetic targets with known answers + the analysis):
  python scripts/run_attack.py --self-check

  # instrument check (patch space, frozen selection, + live drop_norm):
  python scripts/run_attack.py --model qwen-3b --dataset coco:bottle:data/concept_bottle \
      --leaf patch --modes free,thr,eq --selections frozen,live --out runs_attack/qwen3b_bottle_patch

  # R2 test (pixel space, live vs frozen selection):
  python scripts/run_attack.py --model qwen-3b --dataset coco:bottle:data/concept_bottle \
      --leaf pixel --modes free,thr --methods pool,drop_random,drop_norm,drop_query \
      --selections frozen,live --out runs_attack/qwen3b_bottle_pixel

Results are checkpointed after every image to <out>/attack_results.json; rerunning
the same command resumes. float32 only (16-bit precision rounds small perturbations away).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csc_vlm._io import utf8_stdio  # noqa: E402
utf8_stdio()


def _self_check() -> bool:
    import torch
    from csc_vlm.attack import AttackCfg, LinearTarget, attack_image, min_norm_point
    from csc_vlm.attack_analysis import analyze_attack, format_attack
    ok = True
    rng = np.random.default_rng(0)

    # (a) min_norm_point: KKT-optimal and not beaten by random feasible points
    for trial in range(200):
        D, m = 6, int(rng.integers(1, 4))
        A = torch.tensor(rng.normal(size=(m, D)))
        d = rng.normal(size=m).tolist()
        fl = [bool(v) for v in rng.integers(0, 2, size=m)]
        x, n = min_norm_point(A, d, fl)
        if x is None:
            continue
        Ax = (A.double() @ x).numpy()
        feas = all((abs(Ax[i] - d[i]) < 1e-6) if fl[i] else (Ax[i] >= d[i] - 1e-6) for i in range(m))
        for _ in range(200):
            z = x.numpy() + rng.normal(size=D) * 0.3
            Az = A.numpy() @ z
            if all((abs(Az[i] - d[i]) < 1e-9) if fl[i] else (Az[i] >= d[i]) for i in range(m)):
                if np.linalg.norm(z) < n - 1e-9:
                    ok = False
                    print("[self-check] min_norm_point beaten by a random feasible point")
                    break
        if not feas:
            ok = False
            print(f"[self-check] min_norm_point returned an infeasible point (trial {trial})")
    print(f"[self-check] min_norm_point KKT/feasibility: {'PASS' if ok else 'FAIL'}")

    # (b) linear targets: the attack must reproduce the closed-form minimum (upper bound, +0.3 %)
    worst = 0.0
    for trial in range(30):
        D = 24
        w = rng.normal(size=D); v = rng.normal(size=D) * rng.uniform(0.2, 3) + rng.uniform(-1, 1) * w
        y = int(rng.integers(0, 2)); t, sg = -(2 * y - 1), 2 * y - 1
        tg = LinearTarget(w, v, -t * rng.uniform(0.5, 5), sg * rng.uniform(0.05, 3), y)
        rec = attack_image(tg, ("free", "thr", "eq"), AttackCfg())
        for md in ("free", "thr", "eq"):
            r, lin = rec["modes"][md], rec["lin"][md]
            rel = (r["cost"] / lin - 1) if r["status"] == "ok" else math.inf
            worst = max(worst, abs(rel))
            if not (-1e-6 <= rel < 3e-3):
                ok = False
                print(f"[self-check] linear target {trial}/{md}: attack {r} vs closed form {lin}")
    print(f"[self-check] attack == closed form on linear targets (worst rel. err {worst:.1e}): "
          f"{'PASS' if worst < 3e-3 else 'FAIL'}")

    # (b2) v3.0 regression: gradient concentrated on box-saturated coordinates (the pixel stall)
    worst_box = 0.0
    for trial in range(5):
        D = 300
        w = rng.normal(size=D) * 0.05
        blk = rng.choice(D, 30, replace=False)
        w[blk] = rng.uniform(2, 4, size=30)
        hi = np.full(D, 1.0); hi[blk] = 0.0
        tg = LinearTarget(w, rng.normal(size=D), b0=-2.0, s0=-1.0, y=0, lo=np.full(D, -1.0), hi=hi)
        r = attack_image(tg, ("free",), AttackCfg())["modes"]["free"]
        exact = 2.0 / np.linalg.norm(np.delete(w, blk))
        rel = (r["cost"] / exact - 1) if r["status"] == "ok" else math.inf
        worst_box = max(worst_box, abs(rel))
    good = worst_box < 3e-3
    ok &= good
    print(f"[self-check] box-aware projection on saturated coordinates (worst rel. err {worst_box:.1e}): "
          f"{'PASS' if good else 'FAIL'}")
    # (b3) calibrated tripwire (thrq) reproduces its closed form
    worst_q = 0.0
    for trial in range(10):
        D = 24
        w = rng.normal(size=D); v = rng.normal(size=D) + rng.uniform(-1, 1) * w
        y = int(rng.integers(0, 2)); t, sg = -(2 * y - 1), 2 * y - 1
        tg = LinearTarget(w, v, -t * rng.uniform(0.5, 4), sg * rng.uniform(0.5, 2), y)
        tg.probe.tau_q1, tg.probe.tau_q0 = 0.3, -0.3          # stricter than the midpoint (0)
        rec = attack_image(tg, ("thr", "thrq"), AttackCfg())
        for md in ("thr", "thrq"):
            r = rec["modes"][md]
            if r["status"] == "ok":
                worst_q = max(worst_q, abs(r["cost"] / rec["lin"][md] - 1))
            elif r["status"] != "prefired":
                worst_q = math.inf
        if rec["modes"]["thrq"]["status"] == "ok" and rec["modes"]["thr"]["status"] == "ok":
            if rec["modes"]["thrq"]["cost"] < rec["modes"]["thr"]["cost"] * (1 - 3e-3):
                worst_q = math.inf                               # stricter tripwire cannot be cheaper
    good = worst_q < 3e-3
    ok &= good
    print(f"[self-check] calibrated tripwire == closed form, never cheaper than midpoint: {'PASS' if good else 'FAIL'}")

    # (c) analysis on synthetic records with KNOWN effects
    R = {"leaf": "patch", "methods": ["pool", "drop_norm"], "modes": ["free", "thr"], "records": [],
         "complete": True}
    keeps = (1.0, 0.25, 0.0625)
    for i in range(48):
        y = i % 2
        c_i = math.exp(0.3 * rng.normal())
        for k in keeps:
            cells = [("none", "frozen")] if k >= 1 else [("pool", "frozen"), ("drop_norm", "frozen"),
                                                          ("drop_norm", "live")]
            for meth, sel in cells:
                free = c_i * k ** 0.3 * math.exp(0.02 * rng.normal())
                thr = free * k ** -0.2
                if sel == "live":
                    free, thr = free * 0.95, thr * 0.8
                fires = bool((i % 4 == 0) if k >= 1 else (i % 2 == 0) if k < 0.1 else (i % 3 == 0))
                mk = lambda c: {"status": "ok", "cost": c, "cost_rel": c, "fires": fires,
                                "fires_frozen": fires, "kept_jaccard": 0.9 if sel == "live" else 1.0}
                R["records"].append({"image": i, "y": y, "method": meth, "keep": k, "actual_keep": k,
                                     "selection": sel, "margin0": 1.0,
                                     "lin": {"free": free * math.exp(0.05 * rng.normal()),
                                             "thr": thr * math.exp(0.05 * rng.normal())},
                                     "modes": {"free": mk(free), "thr": mk(thr)}})
    S = analyze_attack(R)
    chk = []
    c = S["costs"]["pool|frozen"]
    chk.append(("instrument passes", S["instrument_pass"].get("free") == "PASS"
                and S["instrument_pass"].get("thr") == "PASS"))
    chk.append(("log OC_free slope ~ +0.30", abs(c["slopes"]["log_free"]["slope"] - 0.3) < 0.02))
    chk.append(("log OC_thr slope ~ +0.10", abs(c["slopes"]["log_thr"]["slope"] - 0.1) < 0.02))
    chk.append(("log T_true slope ~ -0.20", abs(c["decomposition_thr"]["log_T"]["slope"] + 0.2) < 0.02))
    chk.append(("exact split", abs(c["decomposition_thr"]["recon_error"]) < 1e-9))
    st_ = S["steering"]["drop_norm"]["0.0625"]
    chk.append(("steering thr = log(1/0.8)", abs(st_["thr"]["log_ratio"]["mean"] - math.log(1 / 0.8)) < 0.02))
    chk.append(("R2 index ~ 0.172", abs(st_["R2_index_thr"]["mean"] - (math.log(1 / 0.8) - math.log(1 / 0.95))) < 0.02))
    inc = S["incidental"]["pool|frozen"]["paired_diff_refit"]["mean"]
    chk.append(("incidental paired diff = +0.25", abs(inc - 0.25) < 1e-9))
    for name, good in chk:
        if not good:
            ok = False
        print(f"[self-check] analysis: {name}: {'PASS' if good else 'FAIL'}")
    print(format_attack(S)[:600] + " ...")
    return ok


def main():
    from csc_vlm.attack import DEFAULT_KEEPS, DEFAULT_METHODS, LEAVES, AttackCfg
    ap = argparse.ArgumentParser(description="pipeline v3: monitor-constrained minimum-norm attacks")
    ap.add_argument("--model", default="qwen-3b")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="coco:bottle:data/concept_bottle")
    ap.add_argument("--leaf", default="patch", choices=list(LEAVES))
    ap.add_argument("--modes", default="free,thr,thrq,eq",
                    help="free | thr (midpoint tripwire) | thrq (calibrated tripwire, --fpr) | eq")
    ap.add_argument("--selections", default="frozen,live")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--keeps", default=",".join(str(k) for k in DEFAULT_KEEPS))
    ap.add_argument("--n-images", type=int, default=32)
    ap.add_argument("--n-monitor", type=int, default=144)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--iters", type=int, default=200, help="max iterations (early stop on convergence)")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--fpr", type=float, default=0.05, help="calibrated tripwire false-positive rate")
    ap.add_argument("--no-pgd-check", action="store_true")
    ap.add_argument("--bisect", type=int, default=10)
    ap.add_argument("--monitor", default="refit", choices=["refit", "frozen"],
                    help="probe the attacker must evade: refit per (method, keep) or the keep-1 probe")
    ap.add_argument("--read-layer-frac", type=float, default=0.70)
    ap.add_argument("--query-layer", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32"],
                    help="float32 only: 16-bit rounds the small adversarial perturbations away")
    ap.add_argument("--multi-gpu", action="store_true")
    ap.add_argument("--time-limit-hours", type=float, default=0.0,
                    help="stop cleanly (and checkpoint) after this many hours; rerun to resume")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs_attack/default")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        good = _self_check()
        print("[self-check] " + ("ALL PASS" if good else "FAILED"))
        sys.exit(0 if good else 1)

    from csc_vlm.attack import run_attacks, save_json
    from csc_vlm.attack_analysis import analyze_attack, format_attack
    from csc_vlm.data import load_images
    from csc_vlm.stage0 import Stage0Config, Stage0VLM
    from csc_vlm.vlm import VLMHandle
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_stage0 import _git_rev  # noqa: E402

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "attack_results.json")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    sels = [s.strip() for s in args.selections.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    keeps = [float(k) for k in args.keeps.split(",") if k.strip()]
    cfg = AttackCfg(iters=args.iters, bisect=args.bisect, patience=args.patience,
                    tripwire_fpr=args.fpr, pgd_check=not args.no_pgd_check)

    images, labels = load_images(args.dataset, per_class=args.per_class, seed=args.seed)
    s0cfg = Stage0Config(model=args.model, model_id=args.model_id, dataset=args.dataset,
                         read_layer_frac=args.read_layer_frac, query_layer=args.query_layer,
                         device=args.device, dtype=args.dtype, n_eval=max(args.n_images, 16),
                         n_monitor=args.n_monitor, seed=args.seed)
    handle = VLMHandle.from_pretrained(args.model, model_id=args.model_id, dtype=args.dtype,
                                       device=args.device, single_device=not args.multi_gpu,
                                       read_layer_frac=args.read_layer_frac,
                                       query_layer=args.query_layer)
    coll = Stage0VLM(s0cfg, images, labels, handle=handle)
    acc0 = coll.calibrate_polarity()
    print(f"[attack] keep=1 accuracy (16 imgs) = {acc0:.2f}; leaf={args.leaf} modes={modes} "
          f"selections={sels} methods={methods} keeps={keeps} n_images={args.n_images}")
    t0 = time.time()
    R = run_attacks(coll, leaf=args.leaf, keeps=keeps, methods=methods, selections=sels, modes=modes,
                    n_images=args.n_images, cfg=cfg, monitor=args.monitor, out_path=out_json,
                    resume=not args.no_resume,
                    time_limit_s=(args.time_limit_hours * 3600 if args.time_limit_hours > 0 else None))
    R["git"] = _git_rev()
    R["args"] = vars(args)
    R["wall_seconds"] = time.time() - t0
    save_json(R, out_json)
    S = analyze_attack(R)
    txt = format_attack(S)
    print(txt)
    with open(os.path.join(args.out, "attack_summary.txt"), "w") as f:
        f.write(txt + "\n")
    save_json(S, os.path.join(args.out, "attack_summary.json"))
    print(f"\nsaved: {out_json}, {args.out}/attack_summary.{{txt,json}}  "
          f"({'complete' if R.get('complete') else 'PARTIAL: rerun the same command to resume'})")


if __name__ == "__main__":
    main()
