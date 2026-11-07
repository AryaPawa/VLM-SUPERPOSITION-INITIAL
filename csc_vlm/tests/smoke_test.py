"""
tests/smoke_test.py  --  fast CPU smoke test (no GPU, no torch, no network).

Verifies the legacy grid pipeline on the mock backend and that the analysis
harnesses (X1 toy, X1b trained toy, Stage 0 v2, Stage 1 v2) pass their self-checks.
The real-model-class tests (tiny random LLaVA / Qwen2.5-VL, needs torch +
transformers) live in tests/test_pipeline_v2.py; run everything with
    python -m pytest tests/ -v This is the test the
README points at:

    python -m pytest tests/smoke_test.py -v

Run it from the csc_vlm/ project directory. Each case shells out to the real CLI so
it tests exactly what a user runs. Total runtime is a few seconds.
"""
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PY = os.path.join(PROJECT_ROOT, "run.py")
SCRIPTS = os.path.join(PROJECT_ROOT, "scripts")


def _run(args, cwd):
    """Run a CLI command; return (returncode, stdout+stderr). 180 s cap."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"                     # headless matplotlib
    env["PYTHONIOENCODING"] = "utf-8"             # child writes UTF-8 even when piped on Windows
    proc = subprocess.run([sys.executable, *args], cwd=cwd, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", timeout=180)
    return proc.returncode, proc.stdout


@pytest.mark.parametrize("method", ["merge", "prune"])
def test_mock_grid_pipeline(tmp_path, method):
    """run.py --model mock produces a PASSED verdict and all output files."""
    out = tmp_path / f"mock_{method}"
    rc, log = _run([RUN_PY, "--model", "mock", "--compression-method", method,
                    "--out", str(out)], cwd=str(tmp_path))
    assert rc == 0, f"mock {method} exited {rc}\n{log}"
    for fname in ("frontier.csv", "verdict.json", "figure_frontier.png",
                  "checkpoint.json", "monitor.npz"):
        assert (out / fname).exists(), f"missing {fname} for mock {method}\n{log}"
    assert "VERDICT: PASSED" in log, f"mock {method} did not PASS\n{log}"


def test_x1_toy(tmp_path):
    """X1 toy kill-switch: all six checks PASS."""
    rc, log = _run([os.path.join(SCRIPTS, "run_x1_toy.py")], cwd=str(tmp_path))
    assert rc == 0, f"x1 toy exited {rc}\n{log}"
    assert "FAIL" not in log, f"an X1 check FAILed\n{log}"


def test_x1b_trained_toy_selfcheck(tmp_path):
    """X1b: manual backprop == finite differences; closed-form OC == min-norm solve."""
    rc, log = _run([os.path.join(SCRIPTS, "run_x1b_toy.py"), "--self-check"], cwd=str(tmp_path))
    assert rc == 0, f"x1b self-check exited {rc}\n{log}"
    assert "PASS" in log, f"x1b self-check did not pass\n{log}"


def test_stage0_selfcheck(tmp_path):
    """Stage 0 metric/orchestration self-check (mock, no torch)."""
    rc, log = _run([os.path.join(SCRIPTS, "run_stage0.py"), "--self-check",
                    "--out", str(tmp_path / "s0")], cwd=str(tmp_path))
    assert rc == 0, f"stage0 self-check exited {rc}\n{log}"
    assert "assertions PASS" in log, f"stage0 self-check did not pass\n{log}"


def test_stage1_selfcheck(tmp_path):
    """Stage 1 evasion-metric math + nulls self-check (mock, no torch)."""
    rc, log = _run([os.path.join(SCRIPTS, "run_stage1.py"), "--self-check",
                    "--out", str(tmp_path / "s1")], cwd=str(tmp_path))
    assert rc == 0, f"stage1 self-check exited {rc}\n{log}"
    assert "assertions PASS" in log, f"stage1 self-check did not pass\n{log}"


if __name__ == "__main__":        # allow `python tests/smoke_test.py` too
    sys.exit(pytest.main([__file__, "-v"]))
