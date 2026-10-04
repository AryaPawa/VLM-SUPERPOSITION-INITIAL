"""
csc_vlm.runner
==============

End-to-end orchestration with crash-safe checkpointing:

  fit monitor ONCE (keep=1.0)  ->  save monitor.npz
  calibrate delta range        ->  save checkpoint.json
  for each delta:  PID solve   ->  append row, save checkpoint.json   [resumable]
  write frontier.csv + figure_frontier.png + print G-VLM verdict

Resume (`--resume`) reloads the monitor and calibration and skips deltas already
in the checkpoint, so a cluster pre-emption costs at most one delta of work.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, List

import numpy as np

from .backends import VLMConfig, MockVLMBackend, HFVLMBackend
from .geometry import fit_monitor, measured_geometry
from .solver import (VLMSafetyOracle, calibrate, restore_calibration,
                     pid_solve, evaluate_frontier)
from . import data as datamod

__all__ = ["run_frontier", "self_check"]


# ----------------------------------------------------------------- helpers #
def _build_backend(cfg: VLMConfig, dataset: str, per_class: int, seed: int,
                   smoke: bool = False):
    if cfg.model == "mock":
        return MockVLMBackend(cfg), None, None
    images, labels = datamod.load_images(dataset, per_class=per_class,
                                         smoke=smoke, seed=seed)
    n1 = int(np.sum(labels == 1)); n0 = int(np.sum(labels == 0))
    print(f"[data] {dataset} -> {len(images)} images (class0={n0}, class1={n1})")
    if n0 == 0 or n1 == 0:
        raise ValueError("dataset resolved to a single class; need both labels present")
    backend = HFVLMBackend(cfg, images, labels)
    return backend, images, labels


def _ckpt_paths(outdir: str):
    return (os.path.join(outdir, "checkpoint.json"),
            os.path.join(outdir, "monitor.npz"))


CKPT_VERSION = "0.4.0"


def _save_checkpoint(path, cfg, dataset, n_deltas, steps, seed, calib, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": CKPT_VERSION, "config": dataclasses.asdict(cfg),
                   "dataset": dataset, "n_deltas": n_deltas, "steps": steps,
                   "seed": seed, "calib": calib, "rows": rows}, f, indent=2)
    os.replace(tmp, path)                      # atomic: never leaves a half-written file


# -------------------------------------------------------------- self-check #
def self_check(cfg: VLMConfig, dataset: str = "synthetic", seed: int = 0) -> bool:
    """Cheap end-to-end validation of the REAL model path (plumbing, incl. the
    merge hook) before committing cluster hours. Loads the model, collects at
    keep=1.0 and keep=0.3, asserts the merge fired and everything is finite."""
    print("=== SELF-CHECK (plumbing incl. merge hook) ===")
    backend, _, _ = _build_backend(cfg, dataset, per_class=12, seed=seed, smoke=True)
    d = backend.read_dim()
    print(f"  read_dim={d}  read_layer={getattr(backend, 'read_layer', 'n/a')}")
    H1, y1 = backend.collect(1.0, cfg.n_pairs, seed=seed)
    ok_base = H1.shape[1] == d and np.isfinite(H1).all() and (set(y1.tolist()) == {0, 1})
    print(f"  keep=1.0: H={H1.shape} finite={np.isfinite(H1).all()} labels={sorted(set(y1.tolist()))}")
    backend._merge_fired = False
    H2, _ = backend.collect(0.3, cfg.n_pairs, seed=seed + 1)
    merged = bool(getattr(backend, "_merge_fired", False)) or cfg.model == "mock"
    print(f"  keep=0.3: H={H2.shape} finite={np.isfinite(H2).all()} merge_hook_fired={merged}")
    w_b, U = fit_monitor(H1, y1, cfg.monitor_rank)
    g1 = measured_geometry(H1, w_b, U, cfg.beta, cfg.shrink)
    g2 = measured_geometry(H2, w_b, U, cfg.beta, cfg.shrink)
    finite = np.isfinite(g1["obf_cost"]) and np.isfinite(g2["obf_cost"])
    print(f"  S(keep=1.0)={g1['obf_cost']:.3f}  S(keep=0.3)={g2['obf_cost']:.3f}  "
          f"‖a‖: {g1['leverage']:.3f}->{g2['leverage']:.3f}")
    passed = bool(ok_base and merged and finite)
    print(f"  SELF-CHECK: {'PASS' if passed else 'FAIL'}")
    if cfg.model != "mock" and not merged:
        print("  [!] merge hook did NOT fire at keep<1 -> visual-token compression is")
        print("      inactive. Check backends.HFVLMBackend._pre_hook / _visual_span for")
        print("      this model before the full run.")
    if passed:
        print("  Note: on synthetic images the geometry is degenerate (near-identical")
        print("        images). Real numbers become meaningful with a real --dataset.")
    return passed


# ------------------------------------------------------------------ figure #
def _figure(rows: List[Dict], sweep: List[Dict], outpath: str, model: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Compression-safety frontier ({model})", fontsize=12, fontweight="bold")

    # operating points (PID delta-frontier)
    d = np.array([r["delta"] for r in rows]); comp = np.array([r["compression"] for r in rows])
    S = np.array([r["S"] for r in rows]); lam = np.array([r["lam"] for r in rows])

    # full-range descriptive sweep (robust curve)
    if sweep:
        sc = np.array([s["compression"] for s in sweep])
        sS = np.array([s["S"] for s in sweep]); sSr = np.array([s["S_refit"] for s in sweep])
        sgam = np.array([s["gamma_r"] for s in sweep]); sgamR = np.array([s["gamma_refit"] for s in sweep])
        slev = np.array([s["leverage"] for s in sweep]); sdeff = np.array([s["d_eff"] for s in sweep])
    else:
        sc = comp; sS = S; sSr = np.array([r.get("S_refit", np.nan) for r in rows])
        sgam = np.array([r["gamma_r"] for r in rows]); sgamR = np.array([r.get("gamma_refit", np.nan) for r in rows])
        slev = np.array([r["leverage"] for r in rows]); sdeff = np.array([r["d_eff"] for r in rows])

    # (a) full-range frontier: safety vs compression, with operating points
    a = ax[0, 0]
    a.plot(sS, sc, "-", color="#1f77b4", alpha=0.6, label="full keep sweep")
    a.plot(S, comp, "o", color="#d62728", label="δ operating points")
    a.set_xlabel("achieved safety S (log)"); a.set_ylabel("compression 1/keep (log)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_title("(a) frontier: safety costs compression")
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)

    # (b) price of safety
    ax[0, 1].plot(d, lam, "s-", color="#2ca02c")
    ax[0, 1].set_xlabel("safety floor δ"); ax[0, 1].set_ylabel("shadow price λ")
    ax[0, 1].set_title("(b) price of safety"); ax[0, 1].grid(alpha=0.3)

    # (c) mediators over the FULL sweep: Gamma_r (primary) + leverage (secondary)
    oc = np.argsort(sc)
    c = ax[1, 0]
    c.plot(sc[oc], sgam[oc], "^-", color="#1f77b4", label="Γ_r (escape, primary)")
    c2 = c.twinx(); c2.plot(sc[oc], slev[oc], "o--", color="#d62728", alpha=0.7, label="‖a‖ (secondary)")
    c.set_xlabel("compression 1/keep (log)"); c.set_xscale("log")
    c.set_ylabel("Γ_r", color="#1f77b4"); c2.set_ylabel("‖a‖", color="#d62728")
    c.set_title("(c) mediator vs compression — Γ_r carries it"); c.grid(alpha=0.3, which="both")

    # (d) staleness control: frozen vs refit safety over the sweep
    dax = ax[1, 1]
    dax.plot(sc[oc], sS[oc], "o-", color="#1f77b4", label="S (frozen monitor)")
    dax.plot(sc[oc], sSr[oc], "s--", color="#ff7f0e", label="S (refit monitor)")
    dax.set_xlabel("compression 1/keep (log)"); dax.set_xscale("log")
    dax.set_yscale("log")
    dax.set_ylabel("safety S (log)"); dax.set_title("(d) staleness control: frozen vs refit")
    dax.grid(alpha=0.3, which="both"); dax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130); plt.close(fig)


# --------------------------------------------------------------- main run #
def run_frontier(cfg: VLMConfig, dataset: str, outdir: str, resume: bool = False,
                 n_deltas: int = 6, steps: int = 60, per_class: int = 200,
                 seed: int = 0) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    ck_path, mon_path = _ckpt_paths(outdir)
    rows: List[Dict] = []
    calib = None
    w_b = U = None

    if resume and os.path.exists(ck_path) and os.path.exists(mon_path):
        with open(ck_path) as f:
            ck = json.load(f)
        ck_ver = str(ck.get("version", "0"))
        if not ck_ver.startswith("0.4"):
            # an older-format checkpoint has a different calibration schema (no
            # full-range sweep / warm starts). Resuming it would mix schemes, so we
            # start fresh rather than silently degrade. Use a new --out to keep old
            # results.
            print(f"[resume] checkpoint is v{ck_ver}, incompatible with v{CKPT_VERSION}; "
                  f"starting fresh (use a new --out to keep the old run).")
        elif ck["config"].get("model") != cfg.model:
            raise ValueError(f"checkpoint model {ck['config'].get('model')} != {cfg.model}; "
                             f"use a fresh --out")
        else:
            rows = ck["rows"]; calib = ck["calib"]
            mon = np.load(mon_path); w_b, U = mon["w_b"], mon["U"]
            print(f"[resume] loaded {len(rows)} completed δ from {ck_path}")

    backend, _, _ = _build_backend(cfg, dataset, per_class=per_class, seed=seed)

    if w_b is None:                                    # fresh: fit monitor once
        print("[monitor] fitting once on uncompressed activations (keep=1.0)")
        H0, y0 = backend.collect(cfg.keep_max, max(64, 4 * cfg.n_pairs), seed=12345)
        w_b, U = fit_monitor(H0, y0, cfg.monitor_rank)
        np.savez(mon_path, w_b=w_b, U=U)
        g0 = measured_geometry(H0, w_b, U, cfg.beta, cfg.shrink)
        print(f"[monitor] d_eff(uncompressed)={g0['d_eff']:.1f} ‖a‖={g0['leverage']:.3f} "
              f"Γ={g0['gamma_r']:.3f}")

    oracle = VLMSafetyOracle(backend, cfg, w_b, U)

    if calib is None:                                  # fresh: calibrate
        print("[calibrate] probing keep grid for δ range + O(1) scales")
        calib = calibrate(oracle, cfg, n_deltas=n_deltas)
        _save_checkpoint(ck_path, cfg, dataset, n_deltas, steps, seed, calib, rows)
    restore_calibration(oracle, calib)

    deltas = calib["deltas"]
    warm = calib.get("warm", [None] * len(deltas))
    done = {round(r["delta"], 6) for r in rows}
    for i, delta in enumerate(deltas):
        if round(float(delta), 6) in done:
            continue
        # keep_lo/keep_hi left to pid_solve's tight local bracket around the warm
        # start (inverse-curve solution) -- see solver.pid_solve.
        r = pid_solve(oracle, float(delta), cfg,
                      keep_start=(warm[i] if i < len(warm) else None),
                      steps=steps, seed=seed + 100 * i)
        rows.append(r)
        _save_checkpoint(ck_path, cfg, dataset, n_deltas, steps, seed, calib, rows)
        flag = " (bound)" if r["at_bound"] else ""
        print(f"  δ={delta:6.3f} -> keep={r['keep']:.3f} comp={r['compression']:5.2f}x "
              f"S={r['S']:.3f} S_refit={r['S_refit']:.3f} λ={r['lam']:.3f} "
              f"Γ={r['gamma_r']:.3f} Γ_refit={r['gamma_refit']:.3f} "
              f"‖a‖={r['leverage']:.3f} d_eff={r['d_eff']:.1f}{flag}  [saved]")

    rows = sorted(rows, key=lambda r: r["delta"])
    import pandas as pd
    keys = ("delta", "keep", "compression", "S", "S_sigma", "S_refit", "S_geom",
            "lam", "gamma_r", "gamma_refit", "A", "leverage", "d_eff", "at_bound")
    pd.DataFrame([{k: r.get(k) for k in keys} for r in rows]).to_csv(
        os.path.join(outdir, "frontier.csv"), index=False)
    # descriptive full-range sweep (robust curve, incl. the near-budget cliff)
    sweep = calib.get("sweep", [])
    if sweep:
        skeys = ("keep", "compression", "S", "S_sigma", "S_refit", "gamma_r",
                 "gamma_refit", "A", "leverage", "d_eff")
        pd.DataFrame([{k: s.get(k) for k in skeys} for s in sweep]).to_csv(
            os.path.join(outdir, "sweep.csv"), index=False)
    _figure(rows, sweep, os.path.join(outdir, "figure_frontier.png"), cfg.model)

    ev = evaluate_frontier(rows, S_ref=float(calib.get("S_ref", 1.0)))
    print("\n=== G-VLM (frontier) ===")
    for k, v in ev["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    for k, v in ev["stats"].items():
        print(f"  {k} = {v}")
    print(f"  VERDICT: {'PASSED' if ev['passed'] else 'FAILED'}  "
          f"(model evals: {oracle.evals})")
    print(f"[out] {os.path.join(outdir, 'frontier.csv')}")
    print(f"[out] {os.path.join(outdir, 'sweep.csv')}")
    print(f"[out] {os.path.join(outdir, 'figure_frontier.png')}")
    with open(os.path.join(outdir, "verdict.json"), "w") as f:
        json.dump(ev, f, indent=2)
    return ev
