"""
csc_vlm -- Compression-Auditability study on real VLMs.

Pipeline v2 (2026-11-07):
  compression.py  real carrier-reducing operators (pool, drop_*) + legacy controls, as plans
  vlm.py          VLMHandle: one model handle (load, prompt, spans/grids, answer tokens, run)
  stage0.py       forward-only sanity with CIs (R1, H2b, G1, R2, coverage, geometry)
  evasion.py      Stage 1: carrier-space Gamma(r), obfuscation tax T, log-alignment A,
                  geometry-matched placebos, exact per-image decomposition, validity check
  stats.py        paired slopes, cluster bootstrap, exact signed decomposition
  toy.py          X1 (estimator validation; hard-coded geometry)
  toy_trained.py  X1b (trained toy; emergent geometry; coverage vs interference)

Legacy (kept for reference, known-flawed metric): geometry.py, solver.py, runner.py,
backends.py (run.py grid pipeline). numpy-only modules import without torch.
"""
__version__ = "0.4.0-p2-pipeline-v2"

from . import geometry, backends, data, solver, runner  # noqa: F401
