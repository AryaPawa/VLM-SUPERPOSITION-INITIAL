"""
tests/test_pipeline_v2.py  --  end-to-end checks of pipeline v2 on REAL transformers
model classes (tiny random LLaVA and Qwen2.5-VL), CPU only, no downloads.

    python -m pytest tests/test_pipeline_v2.py -v        (about 1-3 min on CPU)

Skipped automatically if torch/transformers are not installed. What is checked:
  * every compression plan runs through the real model code (masks, positions, mrope);
  * mask_norm == drop_norm (positions preserved, keys removed) to float precision;
  * keep=1.0 with a leaf reproduces the uncompressed logits exactly;
  * carrier-leaf gradients match finite differences (b and a monitor coordinate);
  * nested Gram capture == dense QR projection;
  * Stage 0 and Stage 1 collectors + analysis run end to end, and the first-order
    check executes;
  * the x1b trained toy self-check passes (numpy only).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_models import tiny_llava, tiny_qwen25vl  # noqa: E402

BUILDERS = {"llava": tiny_llava, "qwen25vl": tiny_qwen25vl}
METHODS = ["pool", "drop_norm", "drop_random", "drop_query", "merge_dup", "mask_norm"]


@pytest.fixture(scope="module", params=list(BUILDERS))
def handle(request):
    try:
        return BUILDERS[request.param]()
    except Exception as e:  # pragma: no cover - transformers version without the class/config
        pytest.skip(f"cannot build tiny {request.param}: {type(e).__name__}: {e}")


def _b(h, logits):
    return logits[11] - logits[13]


def test_plans_run_and_mask_equals_drop(handle):
    inp = handle.prepare(3, "Is this a cat?")
    base = handle.compressed_run(inp, "none", 1.0)["logits"]
    for m in METHODS:
        o = handle.compressed_run(inp, m, 0.25, image_key=3)
        assert torch.isfinite(o["logits"]).all(), m
        p = o["plan"]
        assert p.n_carriers >= 1 and p.n_carriers <= p.n_visual
        if m in ("pool", "drop_norm", "drop_random", "drop_query"):
            assert o["new_len"] == p.T - p.n_visual + p.n_carriers
        else:
            assert o["new_len"] == p.T
    a = handle.compressed_run(inp, "mask_norm", 0.25)["logits"]
    d = handle.compressed_run(inp, "drop_norm", 0.25)["logits"]
    assert float((a - d).abs().max()) < 1e-4 * (1 + float(a.abs().max()))
    assert float((a - base).abs().max()) > 1e-6        # compression changed the output


def test_identity_leaf_reproduces_base(handle):
    inp = handle.prepare(4, "Is this a cat?")
    base = handle.compressed_run(inp, "none", 1.0)["logits"]
    o = handle.compressed_run(inp, "none", 1.0, make_leaf=True, grad=True)
    assert float((o["logits"].detach() - base).abs().max()) < 1e-5


@pytest.mark.parametrize("method", ["pool", "drop_random", "merge_dup"])
def test_leaf_gradients_match_finite_differences(handle, method):
    inp = handle.prepare(5, "Is this a cat?")
    rl = handle.read_layer
    o = handle.compressed_run(inp, method, 0.25, image_key=5, make_leaf=True, capture=(rl,), grad=True)
    v = o["leaf"]
    b = _b(handle, o["logits"])
    u = torch.randn(handle.hidden)
    mon = (u * o["hs_raw"][rl][0, -1]).sum()
    gb = torch.autograd.grad(b, v, retain_graph=True)[0]
    gm = torch.autograd.grad(mon, v)[0]
    d = torch.randn_like(v)
    d = d / d.norm()
    eps = 1e-3

    def eval_at(vv):
        r = handle.compressed_run(inp, method, 0.25, image_key=5, capture=(rl,), carriers_override=vv)
        return float(_b(handle, r["logits"])), float((u * r["hs"][rl][-1]).sum())
    bp, mp = eval_at(v.detach() + eps * d)
    bm, mm = eval_at(v.detach() - eps * d)
    fd_b, fd_m = (bp - bm) / (2 * eps), (mp - mm) / (2 * eps)
    an_b, an_m = float((gb * d).sum()), float((gm * d).sum())
    assert abs(fd_b - an_b) < 2e-3 * (1 + abs(an_b)), (fd_b, an_b)
    assert abs(fd_m - an_m) < 2e-3 * (1 + abs(an_m)), (fd_m, an_m)


def test_nested_capture_matches_qr():
    from csc_vlm.evasion import nested_capture
    rng = np.random.default_rng(1)
    G = rng.standard_normal((5, 300)); g = rng.standard_normal(300) + 2 * G[0]
    caps = nested_capture(G @ G.T, G @ g, float(g @ g))
    for r in range(1, 6):
        Q, _ = np.linalg.qr(G[:r].T)
        assert abs(caps[r - 1] - np.sum((Q.T @ g) ** 2) / (g @ g)) < 1e-8


def test_stage0_and_stage1_end_to_end(handle):
    from csc_vlm.stage0 import Stage0Config, Stage0VLM, run_stage0
    from csc_vlm.evasion import Stage1VLM, run_stage1, format_stage1
    images = list(range(16)); labels = np.array([i % 2 for i in images])
    cfg = Stage0Config(dataset="hf:microsoft/cats_vs_dogs", n_eval=8, n_monitor=8)
    pos, neg = [13, 14], [11, 12]
    s0 = Stage0VLM(cfg, images, labels, handle=handle, pos_ids=pos, neg_ids=neg)
    R0 = run_stage0(s0, keeps=(1.0, 0.5, 0.25), methods=("pool", "drop_random", "drop_query"),
                    B=50, verbose=False)
    assert "summary" in R0 and len(R0["pool"]) == 3
    s1 = Stage1VLM(cfg, images, labels, handle=handle, pos_ids=pos, neg_ids=neg)
    R1 = run_stage1(s1, keeps=(1.0, 0.25), methods=("pool", "drop_norm"), r_max=4, r_frozen=4,
                    read_ranks=(1, 2, 4), n_images=4, n_check=2, verbose=False)
    ok = [r for r in R1["per_image"] if r.get("ok")]
    assert len(ok) == 16
    assert all(len(r["cap_in"]) == 4 and 0 <= r["cap_in"][-1] <= 1 for r in ok)
    assert any("check" in r for r in ok)
    dec = R1["summary"]["methods"]["pool"]["decomposition"]
    assert abs(dec["recon_error"]) < 1e-8
    assert isinstance(format_stage1(R1["summary"]), str)


def test_trained_toy_selfcheck():
    from csc_vlm.toy_trained import self_check
    assert self_check()
