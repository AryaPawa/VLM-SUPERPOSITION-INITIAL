"""
csc_vlm -- Compression-Auditability trade-off on real VLMs (Phase 3, standalone).

Self-contained: depends only on numpy/scipy/pandas/matplotlib (+ torch/transformers
for the real-model backends, imported lazily). Nothing from the original `csc`
research package is required.

Public entry point: `csc_vlm.runner.run_frontier`, driven by `run.py`.
"""
__version__ = "0.3.0-vlm-standalone"

from . import geometry, backends, data, solver, runner  # noqa: F401
