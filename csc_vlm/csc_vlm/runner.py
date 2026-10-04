"""
csc_vlm.runner
==============

End-to-end orchestration with crash-safe checkpointing.

Pipeline (replaces PID-based calibrate->solve loop):

  fit monitor ONCE (keep=1.0)        -> save monitor.npz
  grid_solve over KEEP_GRID          -> save checkpoint.json after every point
  write frontier.csv + figure + verdict.json

Resume (`--resume`) reloads the monitor and skips keep_ratio points already
saved in the checkpoint, so a pre-emption costs at most one grid point of work.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, List, Optional

import numpy as np

from .backends import VLMConfig, MockVLMBackend, HFVLMBackend
from .geometry import fit_monitor, measured_geometry
from .solver import VLMSafetyOracle, KEEP_GRID, grid_solve, evaluate_frontier
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


def _save_checkpoint(path, cfg, dataset, seed, rows):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": "1.0.0",
                   "config": dataclasses.asdict(cfg),
                   "dataset": dataset,
                   "seed": seed,
                   "rows": rows}, f, indent=2)
    os.replace(tmp, path)       # atomic write: never leaves a half-written file


# -------------------------------------------------------------- self-check #
def self_check(cfg: VLMConfig, dataset: str = "synthetic", seed: int = 0) -> bool:
    """Cheap end-to-end validation of the real model path (plumbing, incl. the
    compression hook) before committing cluster hours. Loads the model, collects
    at keep=1.0 and keep=0.3, asserts the hook fired and everything is finite."""
    method = cfg.compression_method
    print(f"=== SELF-CHECK (plumbing incl. {method} hook) ===")
    backend, _, _ = _build_backend(cfg, dataset, per_class=12, seed=seed, smoke=True)
    d = backend.read_dim()
    print(f"  read_dim={d}  read_layer={getattr(backend, 'read_layer', 'n/a')}")
    H1, y1 = backend.collect(1.0, cfg.n_pairs, seed=seed)
    ok_base = H1.shape[1] == d and np.isfinite(H1).all() and (set(y1.tolist()) == {0, 1})
    print(f"  keep=1.0: H={H1.shape} finite={np.isfinite(H1).all()} labels={sorted(set(y1.tolist()))}")
    backend._merge_fired = False
    H2, _ = backend.collect(0.3, cfg.n_pairs, seed=seed + 1)
    hook_fired = bool(getattr(backend, "_merge_fired", False)) or cfg.model == "mock"
    print(f"  keep=0.3: H={H2.shape} finite={np.isfinite(H2).all()} hook_fired={hook_fired}")
    w_b, U = fit_monitor(H1, y1, cfg.monitor_rank)
    g1 = measured_geometry(H1, w_b, U, cfg.beta, cfg.shrink)
    g2 = measured_geometry(H2, w_b, U, cfg.beta, cfg.shrink)
    finite = np.isfinite(g1["obf_cost"]) and np.isfinite(g2["obf_cost"])
    print(f"  S(keep=1.0)={g1['obf_cost']:.3f}  S(keep=0.3)={g2['obf_cost']:.3f}  "
          f"‖a‖: {g1['leverage']:.3f}->{g2['leverage']:.3f}")
    passed = bool(ok_base and hook_fired and finite)
    print(f"  SELF-CHECK: {'PASS' if passed else 'FAIL'}")
    if cfg.model != "mock" and not hook_fired:
        print(f"  [!] {method} hook did NOT fire at keep<1 -> visual-token compression is")
        print("      inactive. Check backends.HFVLMBackend._pre_hook / _visual_span for")
        print("      this model before the full run.")
    if passed:
        print("  Note: on synthetic images the geometry is degenerate (near-identical")
        print("        images). Real numbers become meaningful with a real --dataset.")
    return passed


# ------------------------------------------------------------------ figure #
def _figure(rows: List[Dict], outpath: str, model: str, method: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Compression-safety frontier  |  model={model}  method={method}",
                 fontsize=12, fontweight="bold")

    comp = np.array([r["compression"] for r in rows])
    S    = np.array([r["S"]           for r in rows])
    lev  = np.array([r["leverage"]    for r in rows])
    gam  = np.array([r["gamma_r"]     for r in rows])
    deff = np.array([r["d_eff"]       for r in rows])

    o = np.argsort(comp)    # sort by compression for clean lines

    # (a) Main frontier
    ax[0, 0].plot(comp[o], S[o], "o-", color="#1f77b4")
    ax[0, 0].set_xlabel("compression 1/keep")
    ax[0, 0].set_ylabel("safety cost S")
    ax[0, 0].set_title("(a) frontier: more compression -> lower safety")
    ax[0, 0].grid(alpha=0.3)

    # (b) Safety vs keep_ratio (raw view)
    keep = np.array([r["keep"] for r in rows])
    ax[0, 1].plot(keep[o], S[o], "s-", color="#2ca02c")
    ax[0, 1].set_xlabel("keep ratio")
    ax[0, 1].set_ylabel("safety cost S")
    ax[0, 1].set_title("(b) safety vs keep ratio (raw grid)")
    ax[0, 1].grid(alpha=0.3)

    # (c) H3 mechanism: leverage & gamma_r vs compression (inference only)
    c = ax[1, 0]
    c.plot(comp[o], lev[o], "o-", color="#d62728", label="‖a‖ leverage")
    c2 = c.twinx()
    c2.plot(comp[o], gam[o], "^--", color="#7f7f7f", label="Γ_r")
    c.set_xlabel("compression 1/keep")
    c.set_ylabel("‖a‖", color="#d62728")
    c2.set_ylabel("Γ_r", color="#7f7f7f")
    c.set_title("(c) H3 mechanism (inference only — not a gate)")
    c.grid(alpha=0.3)

    # (d) Effective read dimensionality
    ax[1, 1].plot(comp[o], deff[o], "o-", color="#9467bd")
    ax[1, 1].set_xlabel("compression 1/keep")
    ax[1, 1].set_ylabel("read dim d_eff")
    ax[1, 1].set_title("(d) read dimensionality vs compression")
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------- main run #
def run_frontier(cfg: VLMConfig, dataset: str, outdir: str,
                 resume: bool = False,
                 keep_grid: Optional[List[float]] = None,
                 per_class: int = 200,
                 seed: int = 0) -> Dict:
    """Run the full grid-search frontier.

    Args:
        cfg        : VLMConfig (includes compression_method).
        dataset    : dataset identifier string.
        outdir     : output directory for CSV, figure, verdict, checkpoint.
        resume     : if True, reload monitor and skip already-completed grid points.
        keep_grid  : list of keep_ratio values to sweep. Defaults to KEEP_GRID.
        per_class  : images per class to load.
        seed       : base random seed.

    Returns:
        The verdict dict from evaluate_frontier().
    """
    if keep_grid is None:
        keep_grid = KEEP_GRID

    os.makedirs(outdir, exist_ok=True)
    ck_path, mon_path = _ckpt_paths(outdir)
    rows: List[Dict] = []
    w_b = U = None

    # ------ Resume ------
    if resume and os.path.exists(ck_path) and os.path.exists(mon_path):
        with open(ck_path) as f:
            ck = json.load(f)
        if ck["config"].get("model") != cfg.model:
            raise ValueError(f"checkpoint model {ck['config'].get('model')} != {cfg.model}; "
                             f"use a fresh --out dir")
        rows = ck["rows"]
        mon = np.load(mon_path); w_b, U = mon["w_b"], mon["U"]
        print(f"[resume] loaded {len(rows)} completed grid points from {ck_path}")

    backend, _, _ = _build_backend(cfg, dataset, per_class=per_class, seed=seed)

    # ------ Fit monitor once on uncompressed activations ------
    if w_b is None:
        print("[monitor] fitting once on uncompressed activations (keep=1.0)")
        H0, y0 = backend.collect(cfg.keep_max, max(64, 4 * cfg.n_pairs), seed=12345)
        w_b, U = fit_monitor(H0, y0, cfg.monitor_rank)
        np.savez(mon_path, w_b=w_b, U=U)
        g0 = measured_geometry(H0, w_b, U, cfg.beta, cfg.shrink)
        print(f"[monitor] d_eff(uncompressed)={g0['d_eff']:.1f} "
              f"‖a‖={g0['leverage']:.3f} Γ={g0['gamma_r']:.3f}")

    oracle = VLMSafetyOracle(backend, cfg, w_b, U)

    # ------ Grid Search ------
    method = cfg.compression_method
    print(f"\n[grid] sweeping {len(keep_grid)} keep_ratio points  "
          f"(method={method}, dataset={dataset})\n")
    done_keeps = {round(r["keep"], 4) for r in rows}

    for i, keep in enumerate(keep_grid):
        keep_clipped = float(np.clip(keep, cfg.keep_min, cfg.keep_max))
        if round(keep_clipped, 4) in done_keeps:
            print(f"  keep={keep_clipped:.3f}  [skip — already in checkpoint]")
            continue
        g = oracle.measure(keep_clipped, seed=seed + i)
        S = g["obf_cost"]
        S_geom = float(cfg.beta / (g["gamma_r"] * g["leverage"] + 1e-12))
        row = {
            "keep": keep_clipped,
            "compression": float(1.0 / keep_clipped),
            "S": S,
            "S_geom": S_geom,
            "leverage": g["leverage"],
            "gamma_r": g["gamma_r"],
            "d_eff": g["d_eff"],
        }
        rows.append(row)
        _save_checkpoint(ck_path, cfg, dataset, seed, rows)
        print(f"  keep={keep_clipped:.3f}  comp={row['compression']:5.2f}x  "
              f"S={S:.4f}  ‖a‖={g['leverage']:.3f}  "
              f"Γ={g['gamma_r']:.3f}  d_eff={g['d_eff']:.1f}  [saved]")

    # ------ Output ------
    rows_sorted = sorted(rows, key=lambda r: r["keep"], reverse=True)

    import pandas as pd
    keys = ("keep", "compression", "S", "S_geom", "leverage", "gamma_r", "d_eff")
    csv_path = os.path.join(outdir, "frontier.csv")
    pd.DataFrame([{k: r[k] for k in keys} for r in rows_sorted]).to_csv(
        csv_path, index=False)

    fig_path = os.path.join(outdir, "figure_frontier.png")
    _figure(rows_sorted, fig_path, cfg.model, method)

    ev = evaluate_frontier(rows_sorted)

    print(f"\n=== G-VLM (grid frontier | method={method}) ===")
    print("  -- Gates --")
    for k, v in ev["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    print("  -- Inference (H3 mechanism, not a gate) --")
    for k, v in ev["inference"].items():
        if isinstance(v, bool):
            print(f"  [{'yes' if v else 'no '}]  {k}")
        else:
            print(f"  {k} = {v:.4f}")
    print("  -- Stats --")
    for k, v in ev["stats"].items():
        print(f"  {k} = {v}")
    print(f"  VERDICT: {'PASSED' if ev['passed'] else 'FAILED'}  "
          f"(model evals: {oracle.evals})")
    print(f"[out] {csv_path}")
    print(f"[out] {fig_path}")

    verdict_path = os.path.join(outdir, "verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(ev, f, indent=2)
    return ev
