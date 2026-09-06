"""
csc -- Compress-under-a-Safety-Constraint
=========================================

Phase-A package: the exact-geometry toy substrate that proves the core relation
(N -> rho -> Gamma -> S_OC with a Welch feasibility ceiling) before any
cluster-scale work. Owned by workstream R3.

Public surface (Phase A):
    ToyConfig, ToyModel                 -- the substrate (frozen Toy Model API v0.1)
    TokenParams, build_token_dictionary -- token-aggregation substrate (N -> rho_rd)
    estimators.*                        -- Appendix-B measurement definitions
    delta_star_analytic, delta_star_pgd -- input-space S_OC = ||delta*||
    masking_demo, soft_top_k            -- gradient-masking defense probe (R2)
    SweepGrid, run_sweep, aggregate_cells, token_sweep
    evaluate_gate, GateThresholds       -- the G0 gate (random substrate)
    evaluate_token_gate, evaluate_masking
    figure0, figure_token, figure_masking
"""
from . import estimators
from .tokens import TokenParams, build_token_dictionary, TokenBuild
from .model import ToyConfig, ToyModel
from .attack import (delta_star_analytic, delta_star_pgd, EvasionResult,
                     masking_demo, soft_top_k, MaskingResult)
from .sweep import SweepGrid, run_sweep, aggregate_cells, token_sweep
from .gates import (evaluate_gate, GateThresholds, GateResult,
                    evaluate_token_gate, TokenGateResult, evaluate_masking)
from .plotting import figure0, figure_token, figure_masking

__version__ = "0.2.0-phaseA-hardened"

__all__ = [
    "estimators", "ToyConfig", "ToyModel",
    "TokenParams", "build_token_dictionary", "TokenBuild",
    "delta_star_analytic", "delta_star_pgd", "EvasionResult",
    "masking_demo", "soft_top_k", "MaskingResult",
    "SweepGrid", "run_sweep", "aggregate_cells", "token_sweep",
    "evaluate_gate", "GateThresholds", "GateResult",
    "evaluate_token_gate", "TokenGateResult", "evaluate_masking",
    "figure0", "figure_token", "figure_masking",
]