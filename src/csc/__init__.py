"""
csc -- Compress-under-a-Safety-Constraint
=========================================

Phase-A package: the exact-geometry toy substrate that proves the core relation
(N -> rho -> Gamma -> S_OC with a Welch feasibility ceiling) before any
cluster-scale work. Owned by workstream R3.

Public surface (stable within Phase A):
    ToyConfig, ToyModel                 -- the substrate (frozen Toy Model API v0.1)
    estimators.*                        -- Appendix-B measurement definitions
    delta_star_analytic, delta_star_pgd -- the safety factor S_OC = ||delta*||
    SweepGrid, run_sweep, aggregate_cells
    evaluate_gate, GateThresholds       -- the G0 gate
    figure0
"""
from . import estimators
from .model import ToyConfig, ToyModel
from .attack import delta_star_analytic, delta_star_pgd, EvasionResult
from .sweep import SweepGrid, run_sweep, aggregate_cells
from .gates import evaluate_gate, GateThresholds, GateResult
from .plotting import figure0

__version__ = "0.1.0-phaseA"

__all__ = [
    "estimators", "ToyConfig", "ToyModel",
    "delta_star_analytic", "delta_star_pgd", "EvasionResult",
    "SweepGrid", "run_sweep", "aggregate_cells",
    "evaluate_gate", "GateThresholds", "GateResult", "figure0",
]
