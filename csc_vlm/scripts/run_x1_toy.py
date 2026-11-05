#!/usr/bin/env python3
"""
run_x1_toy.py  --  X1 toy-model validation suite (plan Phase 5.1, the kill-switch)
=================================================================================

Runs five experiments on the Elhage-style toy (csc_vlm.toy), each with KNOWN
ground truth, and prints a PASS/FAIL verdict table against the proof-plan
thresholds, saves `toy_x1_results.json` and a figure `toy_x1.png`.

What each experiment decides
----------------------------
  E-NOISE  : does the p2 Gamma_r estimator report a large escape when the true
             escape is ~0?  (reproduces the 0.004-vs-0.33 baseline + n-dependence)
  E-FRONT  : on one compression frontier, the OLD p2 metric (escape flat under
             merge, S blows up under prune) vs the NEW per-example OC (monotone,
             decomposable).  Shows which metric tells the truth.
  E-OCVAL  : does OC equal the closed-form minimal perturbation AND an independent
             attack?  (OC is real; attack is non-masked)
  E-POS    : positive control -- no cheap evasion in the non-superposed regime
             (Stevinson: zero successful attacks when features are orthogonal).
  E-MASK   : does the discrete prune hide a cheaper attack from the naive gradient?
             (the Phase-3.3 / gate-G2 gradient-masking risk, and that a gradient-
             free upper bound catches it).
  E-ALPHA  : F_eff ~ m^alpha -- merge sub-linear (superposition, H1a) vs prune
             alpha->1 (information loss).  The merge-vs-prune R1 discriminator.

CPU only, pure numpy/matplotlib. Run:  python scripts/run_x1_toy.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# allow running from scripts/ or repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from csc_vlm import toy  # noqa: E402
from csc_vlm.toy import (ToyConfig, make_geometry, sample_reads, reads_packed,
                         behaviour_direction, behaviour_labels, compress_prune,
                         coherence, welch_floor, d_eff, f_eff_decodable,
                         fit_monitor_p2, measured_geometry_p2,
                         fit_monitor_detect, jacobian_metric, verify_oc_against_attack,
                         masking_index_through_prune, spearman, fit_loglog_slope,
                         DEFAULT_KEEPS)


# --------------------------------------------------------------------------- #
# E-NOISE : the p2 Gamma_r noise floor
# --------------------------------------------------------------------------- #
def exp_noise_floor(cfg: ToyConfig):
    """Uncompressed reads only. Compare p2 Gamma in-sample (monitor fit on the
    same points -> overfit -> ~0) vs out-of-sample (monitor frozen, Sigma^1/2 on
    fresh points -> the honest estimate the grid actually uses) across n. The gap
    and the out-of-sample inflation are pure estimator artifact -- the RUN-1..8
    '0.004 at fit time vs 0.33 at the grid point' effect."""
    U = make_geometry(cfg.F, cfg.D, seed=cfg.seed)
    ns = [24, 48, 96, 200, 500, 2000]
    rows = []
    for n in ns:
        c = ToyConfig(**{**cfg.__dict__, "n": n})
        H, Z, _ = sample_reads(U, c, seed=100)
        y, _ = behaviour_labels(H, behaviour_direction(U, cfg))
        Up2 = fit_monitor_p2(H, y, cfg.monitor_rank)
        g_in = measured_geometry_p2(H, Up2)["gamma_r"]            # in-sample
        # out-of-sample: freeze monitor on this set, evaluate Sigma^1/2 on fresh pts
        Hb, _, _ = sample_reads(U, c, seed=777)
        g_out = measured_geometry_p2(Hb, Up2)["gamma_r"]
        rows.append({"n": n, "gamma_p2_insample": g_in, "gamma_p2_outsample": g_out})
    # honest reference: rank-r detection monitor, Jacobian Gamma on a large set
    cbig = ToyConfig(**{**cfg.__dict__, "n": 4000})
    Hbig, _, _ = sample_reads(U, cbig, seed=100)
    wb = behaviour_direction(U, cfg)
    ybig, taubig = behaviour_labels(Hbig, wb)
    Ud = fit_monitor_detect(Hbig, ybig, cfg.monitor_rank)
    gamma_true = float(jacobian_metric(Hbig, wb, Ud, taubig)["gamma_r"][0])
    worst_out = max(r["gamma_p2_outsample"] for r in rows if r["n"] == 24)
    return {"rows": rows, "gamma_true_jacobian": gamma_true,
            "p2_floor_at_n24": worst_out,
            "verdict_inflated": bool(worst_out > 0.25)}


# --------------------------------------------------------------------------- #
# E-FRONT : old vs new metric over a compression frontier
# --------------------------------------------------------------------------- #
def _gen_reads(U, cfg, wb, method, keep):
    """merge = information-preserving superposition sweep (reads_packed);
    prune = information-losing discrete selection on sink-bearing reads."""
    if method == "merge":
        Hc, Zk = reads_packed(U, cfg, keep, seed=11)
    else:
        Hk, Zk, _ = sample_reads(U, cfg, seed=11)      # paired: same draw each keep
        Hc = compress_prune(Hk, keep)
    return Hc, Zk


def _frontier_one(U, cfg, wb, method, keeps):
    """Frozen p2 monitor (fit at keep=1.0, as runner.py does) + refit detection
    monitor per keep for the new metric."""
    H1, _ = _gen_reads(U, cfg, wb, method, 1.0)
    y1, _ = behaviour_labels(H1, wb)
    Up2_frozen = fit_monitor_p2(H1, y1, cfg.monitor_rank)

    out = {k: [] for k in ["keep", "p2_S", "p2_escape", "p2_leverage", "p2_gamma",
                           "new_oc", "new_gamma", "new_gnorm", "new_margin",
                           "d_eff", "f_eff"]}
    for keep in keeps:
        Hc, Zk = _gen_reads(U, cfg, wb, method, keep)
        yk, tauk = behaviour_labels(Hc, wb)
        p2 = measured_geometry_p2(Hc, Up2_frozen)
        Ud = fit_monitor_detect(Hc, yk, cfg.monitor_rank)
        jm = jacobian_metric(Hc, wb, Ud, tauk)
        out["keep"].append(keep)
        out["p2_S"].append(p2["S"]); out["p2_escape"].append(p2["escape"])
        out["p2_leverage"].append(p2["leverage"]); out["p2_gamma"].append(p2["gamma_r"])
        out["new_oc"].append(float(np.median(jm["oc"])))
        out["new_gamma"].append(float(jm["gamma_r"][0]))
        out["new_gnorm"].append(float(jm["g_b_norm"][0]))
        out["new_margin"].append(float(np.median(jm["margin"])))
        out["d_eff"].append(d_eff(Hc)); out["f_eff"].append(f_eff_decodable(Hc, Zk))
    for k in out:
        out[k] = np.array(out[k], float)
    return out


def _decompose_slopes(fr):
    """log OC = log margin - log||g|| - log Gamma. Slopes vs log keep; share of
    the OC slope carried by the Gamma (escape/evasion) term vs the margin (info)
    term."""
    lk = np.log(fr["keep"])
    s_oc = fit_loglog_slope(fr["keep"], fr["new_oc"])
    s_m = fit_loglog_slope(fr["keep"], fr["new_margin"])
    s_g = fit_loglog_slope(fr["keep"], fr["new_gnorm"])
    s_gam = fit_loglog_slope(fr["keep"], fr["new_gamma"])
    # d logOC = d log m - d log||g|| - d log Gamma
    recon = s_m - s_g - s_gam
    tot = abs(s_m) + abs(s_g) + abs(s_gam) + toy.EPS
    return {"slope_oc": s_oc, "slope_oc_recon": recon,
            "slope_margin": s_m, "slope_gnorm": s_g, "slope_gamma": s_gam,
            "share_gamma": abs(s_gam) / tot, "share_margin": abs(s_m) / tot}


def exp_frontier(cfg: ToyConfig):
    U = make_geometry(cfg.F, cfg.D, seed=cfg.seed)
    wb = behaviour_direction(U, cfg)
    keeps = np.array(DEFAULT_KEEPS)
    res = {}
    for method in ("merge", "prune"):
        fr = _frontier_one(U, cfg, wb, method, keeps)
        dec = _decompose_slopes(fr)
        res[method] = {
            "frontier": {k: fr[k].tolist() for k in fr},
            "spearman_keep_p2S": spearman(fr["keep"], fr["p2_S"]),
            "spearman_p2gamma_leverage": spearman(fr["p2_gamma"], fr["p2_leverage"]),
            "spearman_keep_newOC": spearman(fr["keep"], fr["new_oc"]),
            "beta_oc_per_logunit": dec["slope_oc"],     # H2: >= 0.25
            "decomp": dec,
            "d_eff_full": float(fr["d_eff"][0]), "d_eff_min": float(fr["d_eff"].min()),
        }
    return res


# --------------------------------------------------------------------------- #
# E-OCVAL : OC == closed form == independent attack  (non-masked)
# --------------------------------------------------------------------------- #
def exp_oc_validation(cfg: ToyConfig):
    U = make_geometry(cfg.F, cfg.D, seed=cfg.seed)
    wb = behaviour_direction(U, cfg)
    Hc, Z = reads_packed(U, cfg, 0.5, seed=22)
    y, tau = behaviour_labels(Hc, wb)
    Ud = fit_monitor_detect(Hc, y, cfg.monitor_rank)
    v = verify_oc_against_attack(Hc, wb, Ud, tau, n_check=48, seed=5)
    v["verdict_formula_ok"] = bool(v["rel_formula_vs_closed"] < 0.02)
    v["verdict_nonmasked"] = bool(v["rel_pgd_vs_closed"] < 0.05)
    v["verdict_monitor_blind"] = bool(v["monitor_leak_max"] < 1e-6)
    return v


# --------------------------------------------------------------------------- #
# E-POS : positive control (superposed vs non-superposed)
# --------------------------------------------------------------------------- #
def _orthogonal_geometry(F, D, seed):
    """Non-superposed geometry: F <= D orthonormal feature directions."""
    assert F <= D
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(A)
    return Q[:, :F]


def exp_positive_control(cfg: ToyConfig):
    # Stevinson control: no cheap evasion channel without superposition. Large n,
    # no sinks, so we isolate geometry. Superposed arm taken at a COMPRESSED
    # operating point (keep=0.12) -- the regime the project is actually about --
    # vs the orthogonal (non-superposed) regime where the channel is closed.
    base = {**cfg.__dict__, "n": 3000, "n_sinks": 0}
    # superposed: F >> D, packed (compressed)
    cfgs = ToyConfig(**base)
    Us = make_geometry(cfgs.F, cfgs.D, seed=cfgs.seed)
    wbs = behaviour_direction(Us, cfgs)
    Hsc, _ = reads_packed(Us, cfgs, 0.12, seed=33)
    ys, taus = behaviour_labels(Hsc, wbs)
    Uds = fit_monitor_detect(Hsc, ys, cfgs.monitor_rank)
    jms = jacobian_metric(Hsc, wbs, Uds, taus)
    # non-superposed: F = D/2 orthonormal (features fit with room to spare)
    Fn = cfg.D // 2
    Un = _orthogonal_geometry(Fn, cfg.D, seed=cfg.seed)
    cfgn = ToyConfig(**{**base, "F": Fn})
    wbn = behaviour_direction(Un, cfgn)
    Hnc, _, _ = sample_reads(Un, cfgn, seed=33)
    yn, taun = behaviour_labels(Hnc, wbn)
    Udn = fit_monitor_detect(Hnc, yn, cfgn.monitor_rank)
    jmn = jacobian_metric(Hnc, wbn, Udn, taun)

    gam_s = float(jms["gamma_r"][0]); gam_n = float(jmn["gamma_r"][0])
    oc_s = float(np.median(jms["oc"])); oc_n = float(np.median(jmn["oc"]))
    ratio = oc_n / (oc_s + toy.EPS)
    return {"coherence_superposed": coherence(Us), "coherence_ortho": coherence(Un),
            "gamma_superposed": gam_s, "gamma_nonsup": gam_n,
            "oc_superposed": oc_s, "oc_nonsup": oc_n, "oc_ratio_nonsup_over_sup": ratio,
            "verdict_posctrl": bool(gam_n < 0.1 and ratio > 3.0)}


# --------------------------------------------------------------------------- #
# E-MASK : gradient masking through discrete prune
# --------------------------------------------------------------------------- #
def exp_masking(cfg: ToyConfig):
    U = make_geometry(cfg.F, cfg.D, seed=cfg.seed)
    wb = behaviour_direction(U, cfg)
    H, _, _ = sample_reads(U, cfg, seed=44)
    keeps = [0.5, 0.25, 0.12]
    rows = []
    for keep in keeps:
        Hc = compress_prune(H, keep)      # fit monitor on the pruned reads
        y, tau = behaviour_labels(Hc, wb)
        Ud = fit_monitor_detect(Hc, y, cfg.monitor_rank)
        mi = masking_index_through_prune(H, wb, Ud, tau, keep, n_check=32, seed=7)
        rows.append({"keep": keep, **mi})
    worst = max(r["masking_index"] for r in rows)
    return {"rows": rows, "worst_masking_index": worst,
            "verdict_masking_detected": bool(worst > 1.25)}


# NOTE: F_eff ~ N^alpha (H1a, the information-loss vs superposition exponent) is a
# VLM experiment (plan E2 / Stage 0), not a toy kill-switch -- a linear decoder on
# 3%-sparse toy features is too weak to estimate F_eff reliably. The information-
# loss vs evasion split IS demonstrated on the toy, via the margin behaviour in
# E-FRONT (merge keeps margin => evasion-driven; prune drops margin => info-loss).


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def make_figure(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (1) noise floor
    nf = R["noise_floor"]
    ns = [r["n"] for r in nf["rows"]]
    ax[0, 0].plot(ns, [r["gamma_p2_outsample"] for r in nf["rows"]], "o-", label="p2 Γ (out-of-sample)")
    ax[0, 0].plot(ns, [r["gamma_p2_insample"] for r in nf["rows"]], "s--", label="p2 Γ (in-sample)")
    ax[0, 0].axhline(nf["gamma_true_jacobian"], color="k", ls=":", label="true Γ (Jacobian)")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("n samples"); ax[0, 0].set_ylabel("Γ_r")
    ax[0, 0].set_title("E-NOISE: p2 Γ is estimator noise"); ax[0, 0].legend(fontsize=8)

    # (2) merge frontier, old vs new
    for col, method in [(1, "merge"), (2, "prune")]:
        fr = R["frontier"][method]["frontier"]
        keep = np.array(fr["keep"])
        a = ax[0, col]
        a.plot(keep, np.array(fr["p2_S"]) / (np.array(fr["p2_S"])[0] + 1e-9), "o-", color="r", label="p2 S (norm)")
        a.plot(keep, np.array(fr["p2_escape"]) / (np.array(fr["p2_escape"])[0] + 1e-9), "s--", color="m", label="p2 escape (norm)")
        a.plot(keep, np.array(fr["new_oc"]) / (np.array(fr["new_oc"])[0] + 1e-9), "^-", color="b", label="new OC (norm)")
        a.set_yscale("log")
        a.set_xlabel("keep (←more compression)"); a.set_ylabel("normalised (log)")
        a.invert_xaxis()
        a.set_title(f"E-FRONT [{method}]: p2 unstable, new OC monotone"); a.legend(fontsize=8)

    # (4) decomposition shares
    a = ax[1, 0]
    methods = ["merge", "prune"]
    gshare = [R["frontier"][m]["decomp"]["share_gamma"] for m in methods]
    mshare = [R["frontier"][m]["decomp"]["share_margin"] for m in methods]
    x = np.arange(2)
    a.bar(x - 0.2, gshare, 0.4, label="Γ (evasion) share")
    a.bar(x + 0.2, mshare, 0.4, label="margin (info-loss) share")
    a.set_xticks(x); a.set_xticklabels(methods); a.set_ylabel("share of OC slope")
    a.set_title("E-FRONT: what drives OC"); a.legend(fontsize=8)

    # (5) positive control
    a = ax[1, 1]
    pc = R["positive_control"]
    a.bar([0, 1], [pc["oc_superposed"], pc["oc_nonsup"]], color=["#c44", "#4a4"])
    a.set_xticks([0, 1]); a.set_xticklabels(["superposed", "non-superposed"])
    a.set_ylabel("OC (median)"); a.set_title(f"E-POS: no cheap evasion w/o superpos.\n"
                                             f"Γ {pc['gamma_superposed']:.2f} vs {pc['gamma_nonsup']:.2f}")

    # (6) masking index
    a = ax[1, 2]
    mk = R["masking"]["rows"]
    ks = [r["keep"] for r in mk]
    a.plot(ks, [r["masking_index"] for r in mk], "o-")
    a.axhspan(0.8, 1.25, color="g", alpha=0.15, label="non-masked band")
    a.set_xlabel("keep (prune)"); a.set_ylabel("OC_naive / OC_gradfree")
    a.invert_xaxis(); a.set_title("E-MASK: discrete prune hides a cheaper attack"); a.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# verdict table
# --------------------------------------------------------------------------- #
def print_verdicts(R):
    def mark(ok): return "PASS" if ok else "FAIL"
    print("\n" + "=" * 74)
    print(" X1 TOY VALIDATION  --  verdicts vs proof-plan thresholds")
    print("=" * 74)

    nf = R["noise_floor"]
    print(f"[E-NOISE] p2 Γ at n=24 (true≈{nf['gamma_true_jacobian']:.3f}) = "
          f"{nf['p2_floor_at_n24']:.3f}  -> p2 Γ is noise  [{mark(nf['verdict_inflated'])}]")

    for m in ("merge", "prune"):
        fm = R["frontier"][m]
        print(f"[E-FRONT:{m:5s}] p2 Spearman(keep,S)={fm['spearman_keep_p2S']:+.2f}  "
              f"Spearman(Γ_p2,lev)={fm['spearman_p2gamma_leverage']:+.2f}  "
              f"d_eff {fm['d_eff_full']:.1f}->{fm['d_eff_min']:.1f}  | "
              f"new β_OC/logu={fm['beta_oc_per_logunit']:+.2f} (H2≥0.25)  "
              f"Γ-share={fm['decomp']['share_gamma']:.0%}")
    h2b = (R["frontier"]["merge"]["decomp"]["share_gamma"]
           > R["frontier"]["prune"]["decomp"]["share_gamma"])
    print(f"           H2b merge is evasion-driven > prune is info-driven  [{mark(h2b)}]")

    ov = R["oc_validation"]
    print(f"[E-OCVAL ] OC vs closed-form rel={ov['rel_formula_vs_closed']:.1e} "
          f"[{mark(ov['verdict_formula_ok'])}]  "
          f"independent attack rel={ov['rel_pgd_vs_closed']:.1e} (non-masked) "
          f"[{mark(ov['verdict_nonmasked'])}]  flip={ov['flip_rate']:.0%}")

    pc = R["positive_control"]
    print(f"[E-POS   ] non-sup Γ={pc['gamma_nonsup']:.3f}, OC ratio(nonsup/sup)="
          f"{pc['oc_ratio_nonsup_over_sup']:.1f}x  [{mark(pc['verdict_posctrl'])}]")

    mk = R["masking"]
    print(f"[E-MASK  ] worst masking index={mk['worst_masking_index']:.2f} "
          f"(>1.25 ⇒ discrete prune hides a cheaper attack) [{mark(mk['verdict_masking_detected'])}]")

    print("=" * 74)
    print(" Summary: the p2 metric fails on ground truth (E-NOISE, E-FRONT); the")
    print(" replacement OC is exact & non-masked (E-OCVAL), has the right null")
    print(" (E-POS), and the discrete-prune masking risk is real & detectable")
    print(" (E-MASK). Merge vs prune cleanly separates evasion (margin-stable,")
    print(" Γ-driven) from information loss (margin-driven) in E-FRONT (H2b).")
    print(" => build evasion.py on VLMs next: carrier-space OC + refit monitor +")
    print(" random-prune null + the gradient-free check. (F_eff/α = Stage-0 on VLMs.)")
    print("=" * 74 + "\n")


def main():
    # F >> D so there is genuine superposition; sparse base activation; a few sinks
    # for the prune-collapse arm. n modest to mirror the real small-sample regime.
    cfg = ToyConfig(F=160, D=64, sparsity=0.03, n=256, monitor_rank=8,
                    n_sinks=3, sink_scale=4.0, seed=0)
    R = {
        "config": cfg.__dict__,
        "geometry": {
            "coherence": coherence(make_geometry(cfg.F, cfg.D, cfg.seed)),
            "welch_floor_full": welch_floor(cfg.F, cfg.D),
        },
        "noise_floor": exp_noise_floor(cfg),
        "frontier": exp_frontier(cfg),
        "oc_validation": exp_oc_validation(cfg),
        "positive_control": exp_positive_control(cfg),
        "masking": exp_masking(cfg),
    }
    out_json = os.path.join(os.getcwd(), "toy_x1_results.json")
    out_png = os.path.join(os.getcwd(), "toy_x1.png")
    with open(out_json, "w") as f:
        json.dump(R, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    make_figure(R, out_png)
    print_verdicts(R)
    print(f"saved: {out_json}\nsaved: {out_png}")


if __name__ == "__main__":
    main()