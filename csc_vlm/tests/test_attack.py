"""
tests/test_attack.py  --  pipeline v3 attack harness on REAL transformers classes (tiny,
random LLaVA + Qwen2.5-VL; no downloads) and on synthetic targets with known answers.

Checks:
  * the new VLMHandle.run options are exact: a zero span_delta reproduces the
    compressed logits; patch- and pixel-leaf gradients match finite differences;
  * frozen vs live selection: identical for tiny perturbations, different selections
    for large ones (drop_norm), and the attack uses the live plan;
  * min_norm_point is KKT-optimal; the attack reproduces closed-form minima on linear
    targets; and on curved 2-D targets it is within 1.5 % of a brute-force minimum;
  * the full grid (run_attacks) runs end to end on both tiny models, every returned
    cost is a verified success, and the analysis runs on the output.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_models import tiny_llava, tiny_qwen25vl  # noqa: E402

BUILDERS = {"llava": tiny_llava, "qwen25vl": tiny_qwen25vl}
POS, NEG = [13, 14], [11, 12]


@pytest.fixture(scope="module", params=list(BUILDERS))
def handle(request):
    try:
        return BUILDERS[request.param]()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"cannot build tiny {request.param}: {type(e).__name__}: {e}")


def _coll(handle, n_eval=6, n_mon=12):
    from csc_vlm.stage0 import Stage0Config, Stage0VLM
    images = list(range(n_eval + n_mon + 4)); labels = np.array([i % 2 for i in images])
    cfg = Stage0Config(dataset="hf:microsoft/cats_vs_dogs", n_eval=n_eval, n_monitor=n_mon)
    return Stage0VLM(cfg, images, labels, handle=handle, pos_ids=POS, neg_ids=NEG)


def _b(logits):
    return float(torch.logsumexp(logits[POS], 0) - torch.logsumexp(logits[NEG], 0))


# --------------------------------------------------------------------------- #
# VLMHandle.run extensions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["none", "pool", "drop_norm", "drop_random", "merge_dup"])
def test_zero_span_delta_reproduces_compressed_logits(handle, method):
    from csc_vlm.compression import build_plan
    inp = handle.prepare(3, "Is this a cat?")
    keep = 1.0 if method == "none" else 0.5
    ref = handle.compressed_run(inp, method, keep, image_key=3)["logits"]
    T = int(inp["input_ids"].shape[1]); runs = handle.visual_runs(inp)
    cap = {}

    def pf(e, rr):
        cap["e"] = e
        return build_plan(method, keep, T, rr, e, image_key=3)
    n = sum(len(r.positions) for r in runs)
    o = handle.run(inp, pf, runs=runs, span_delta=torch.zeros(n, handle.hidden))
    assert float((o["logits"] - ref).abs().max()) < 1e-5


def test_span_delta_requires_plan(handle):
    inp = handle.prepare(3, "x")
    with pytest.raises(ValueError):
        handle.run(inp, None, span_delta=torch.zeros(4, handle.hidden))


def _fd_check(tgt, eps=1e-3, tol=3e-3):
    x0 = tgt.zeros()
    g = tgt.evaluate(x0, grad=True, need_gs=True)
    gen = torch.Generator().manual_seed(0)
    d = torch.randn(x0.shape, generator=gen).to(x0.device)
    d = d / d.norm()
    for key, gk in (("b", "gb"), ("s", "gs")):
        fp = tgt.evaluate(x0 + eps * d)[key]; fm = tgt.evaluate(x0 - eps * d)[key]
        fd = (fp - fm) / (2 * eps)
        an = float((g[gk].reshape(-1) * d.reshape(-1)).sum())
        assert abs(fd - an) < tol * (1 + abs(an)), (key, fd, an)


@pytest.mark.parametrize("leaf,method", [("patch", "pool"), ("patch", "drop_norm"), ("patch", "none"),
                                         ("pixel", "pool"), ("pixel", "none"), ("pixel", "drop_random")])
def test_leaf_gradients_match_finite_differences(handle, leaf, method):
    from csc_vlm.attack import VLMTarget, fit_probe
    coll = _coll(handle)
    keep = 1.0 if method == "none" else 0.5
    probe = fit_probe(coll, method, keep); frozen = fit_probe(coll, "none", 1.0)
    tgt = VLMTarget(coll, int(coll.eval_idx[0]), method, keep, leaf, "frozen", probe, frozen)
    _fd_check(tgt)


def test_live_selection_tracks_the_perturbed_input(handle):
    from csc_vlm.attack import VLMTarget, fit_probe
    coll = _coll(handle)
    probe = fit_probe(coll, "drop_norm", 0.5); frozen = fit_probe(coll, "none", 1.0)
    i = int(coll.eval_idx[0])
    fr = VLMTarget(coll, i, "drop_norm", 0.5, "patch", "frozen", probe, frozen)
    lv = VLMTarget(coll, i, "drop_norm", 0.5, "patch", "live", probe, frozen)
    x = fr.zeros()
    a, b = fr.evaluate(x), lv.evaluate(x)
    assert abs(a["b"] - b["b"]) < 1e-6 and lv.evaluate(x)["kept"] == fr.kept0
    # a large perturbation that boosts the norm of currently DROPPED tokens changes the live set
    dropped = sorted(set(range(fr.n_vis)) - set(fr.kept0))
    big = fr.zeros()
    big[dropped] = 10.0 * fr.emb0[dropped].float() / (fr.emb0[dropped].float().norm(dim=-1, keepdim=True) + 1e-9) \
        * fr.emb0.float().norm(dim=-1).max()
    kl = lv.evaluate(big)["kept"]
    assert kl != fr.kept0 and fr.evaluate(big)["kept"] == fr.kept0
    assert abs(lv.evaluate(big)["b"] - fr.evaluate(big)["b"]) > 1e-6


# --------------------------------------------------------------------------- #
# optimiser exactness on synthetic targets
# --------------------------------------------------------------------------- #
def test_attack_matches_closed_form_on_linear_targets():
    from csc_vlm.attack import AttackCfg, LinearTarget, attack_image
    rng = np.random.default_rng(1)
    for _ in range(15):
        D = 20
        w = rng.normal(size=D); v = rng.normal(size=D) + rng.uniform(-1, 1) * w
        y = int(rng.integers(0, 2)); t, sg = -(2 * y - 1), 2 * y - 1
        tg = LinearTarget(w, v, -t * rng.uniform(0.5, 4), sg * rng.uniform(0.05, 2), y)
        rec = attack_image(tg, ("free", "thr", "eq"), AttackCfg())
        for m in ("free", "thr", "eq"):
            r = rec["modes"][m]
            assert r["status"] == "ok"
            assert -1e-6 <= r["cost"] / rec["lin"][m] - 1 < 3e-3


def test_box_saturated_gradient_does_not_stall():
    """v3.0 regression (the pixel-run stall): most of the gradient on coordinates already
    at a box bound. The box-aware projection must reach the exact constrained minimum."""
    from csc_vlm.attack import AttackCfg, LinearTarget, attack_image
    rng = np.random.default_rng(3)
    for _ in range(5):
        D = 300
        w = rng.normal(size=D) * 0.05
        blk = rng.choice(D, 30, replace=False)
        w[blk] = rng.uniform(2, 4, size=30)
        hi = np.full(D, 1.0); hi[blk] = 0.0
        tg = LinearTarget(w, rng.normal(size=D), b0=-2.0, s0=-1.0, y=0, lo=np.full(D, -1.0), hi=hi)
        rec = attack_image(tg, ("free", "thr"), AttackCfg())
        r = rec["modes"]["free"]
        assert r["status"] == "ok" and r["converged"]
        exact = 2.0 / np.linalg.norm(np.delete(w, blk))
        assert -1e-6 <= r["cost"] / exact - 1 < 1e-3
        assert rec["blocked_grad_frac"] > 0.9


def test_calibrated_tripwire_closed_form_and_ordering():
    from csc_vlm.attack import AttackCfg, LinearTarget, attack_image
    rng = np.random.default_rng(4)
    for _ in range(10):
        D = 24
        w = rng.normal(size=D); v = rng.normal(size=D) + rng.uniform(-1, 1) * w
        y = int(rng.integers(0, 2)); t, sg = -(2 * y - 1), 2 * y - 1
        tg = LinearTarget(w, v, -t * rng.uniform(0.5, 4), sg * rng.uniform(0.5, 2), y)
        tg.probe.tau_q1, tg.probe.tau_q0 = 0.3, -0.3
        rec = attack_image(tg, ("free", "thr", "thrq"), AttackCfg())
        for m in ("thr", "thrq"):
            r = rec["modes"][m]
            if r["status"] == "ok":
                assert -1e-6 <= r["cost"] / rec["lin"][m] - 1 < 1e-3
        c = {m: rec["modes"][m]["cost"] for m in ("free", "thr", "thrq") if rec["modes"][m]["status"] == "ok"}
        if "thr" in c and "free" in c:
            assert c["thr"] >= c["free"] * (1 - 1e-4)          # a constraint can only add cost
        if "thrq" in c and "thr" in c:
            assert c["thrq"] >= c["thr"] * (1 - 1e-4)          # a stricter tripwire can only add cost


def test_attack_close_to_brute_force_on_curved_targets():
    from csc_vlm.attack import AttackCfg, LinearTarget, attack_image
    rng = np.random.default_rng(2)
    th = np.linspace(0, 2 * np.pi, 2880, endpoint=False)
    rs = np.linspace(1e-4, 15, 6000)
    n_checked = 0
    for _ in range(10):
        w, v, q = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
        b0, s0, curv = rng.uniform(0.5, 2.0), rng.uniform(0.05, 0.6), rng.uniform(-0.5, 0.5)
        tg = LinearTarget(w, v, b0, s0, 1, curv=curv, q=q)            # y=1: target b<0, keep s>0
        rec = attack_image(tg, ("free", "thr"), AttackCfg(iters=60))
        D = np.stack([np.cos(th), np.sin(th)], 1)
        for m in ("free", "thr"):
            best = np.inf
            for d in D:
                X = rs[:, None] * d[None]
                okk = -(X @ w + b0 + curv * (X @ q) ** 2) > 0
                if m == "thr":
                    okk &= (X @ v + s0) > 0
                if okk.any():
                    best = min(best, rs[np.argmax(okk)])
            r = rec["modes"][m]
            if not np.isfinite(best):
                assert r["status"] == "failed"
                continue
            assert r["status"] == "ok"
            assert r["cost"] >= best * (1 - 0.005)          # a verified success cannot beat the minimum
            assert r["cost"] <= best * 1.015
            n_checked += 1
    assert n_checked >= 8


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("leaf", ["patch", "pixel"])
def test_run_attacks_end_to_end(handle, leaf, tmp_path):
    from csc_vlm.attack import AttackCfg, run_attacks
    from csc_vlm.attack_analysis import analyze_attack, format_attack
    coll = _coll(handle, n_eval=4, n_mon=12)
    out = str(tmp_path / "a.json")
    R = run_attacks(coll, leaf=leaf, keeps=(1.0, 0.5), methods=("pool", "drop_norm"),
                    selections=("frozen", "live"), modes=("free", "thr", "thrq", "eq"), n_images=3,
                    cfg=AttackCfg(iters=12, bisect=6, patience=4, pgd_steps=5), out_path=out, verbose=False)
    assert R["complete"]
    recs = [r for r in R["records"] if "modes" in r]
    # cells: none@1 + pool@.5 frozen + drop_norm@.5 frozen/live = 4 cells x 3 images
    assert len(recs) == 12 and not [r for r in R["records"] if "error" in r]
    for r in recs:
        for m, mr in r["modes"].items():
            assert mr["status"] in ("ok", "prebroken", "prefired", "failed")
            if mr["status"] == "ok":
                assert mr["cost"] > 0 and math.isfinite(mr["cost"])
                t = -(2 * r["y"] - 1)
                assert t * mr["b_final"] > 0                      # verified: answer is the target
                if m == "thr":
                    assert not mr["fires"]                         # verified: tripwire silent
                if m == "thrq":
                    assert not mr["fires_q"]                       # verified: calibrated tripwire silent
    # resume: rerunning does no new work
    R2 = run_attacks(coll, leaf=leaf, keeps=(1.0, 0.5), methods=("pool", "drop_norm"),
                     selections=("frozen", "live"), modes=("free", "thr", "thrq", "eq"), n_images=3,
                     cfg=AttackCfg(iters=12, bisect=6, patience=4, pgd_steps=5), out_path=out, verbose=False)
    assert len(R2["records"]) == 12
    S = analyze_attack(R)
    assert isinstance(format_attack(S), str) and S["n_records"] == 12


def test_pixel_box_qwen_layout():
    from csc_vlm.attack import pixel_box

    class IP:
        image_mean = [0.5, 0.4, 0.3]
        image_std = [0.2, 0.25, 0.3]

    class H:
        class processor:
            image_processor = IP
    per = 2 * 4 * 4
    pv = torch.zeros(10, 3 * per)
    for c in range(3):
        pv[:, c * per:(c + 1) * per] = (0.5 - IP.image_mean[c]) / IP.image_std[c]
    lo, hi = pixel_box(H, pv)
    assert torch.allclose(lo[0, per:2 * per], torch.full((per,), (0 - 0.4) / 0.25))
    assert torch.allclose(hi[0, 2 * per:], torch.full((per,), (1 - 0.3) / 0.3))
    assert pixel_box(H, pv + 100.0) is None                 # outside the inferred box -> refuse
