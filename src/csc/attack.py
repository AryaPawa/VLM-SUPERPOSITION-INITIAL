"""
csc.attack
==========

Safety factor S_OC in the toy substrate = per-example minimal evading
perturbation ||delta*|| (review 5.4). Larger == safer.

CORRECT (input-space) evasion. The attacker perturbs the *input feature
activations* z (Elhage convention), not the read activation directly. Read
activation h = W z. Behaviour readout uses w_b = W[:, b]; monitor reads the
rank-r subspace U. With a = W^T w_b (leverage of each feature move on the
behaviour) and M = P_U W (monitor image of each feature move):

    minimise ||delta_z||   s.t.  a^T delta_z >= beta,  ||M delta_z|| <= tau.

Hard-evasion (tau = 0): delta_z in null(M), and

    ||delta*|| = beta / || P_null(M) a ||  =  beta / (Gamma_r * ||a||),

    Gamma_r = || P_null(M) a || / ||a||      (input-space rank-r evasion gain,
                                              == review Appendix-B Gamma_r with J = W)

This is Proposition 1 exactly, and it carries the mechanism: superposition makes
||a|| = ||W^T w_b|| large (feature b correlated with many features) while the
rank-r monitor removes at most r dimensions of leverage, so ||P_null(M) a||
grows and ||delta*|| falls as pressure rises. In the no-superposition regime
(W orthonormal, monitor watching w_b) P_null(M) a -> 0 and ||delta*|| -> inf:
the "zero successful attacks" control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["EvasionResult", "null_projector", "input_evasion_gain",
           "delta_star_analytic", "delta_star_pgd"]


@dataclass
class EvasionResult:
    delta_input: float      # ||delta*|| in input (feature) metric  == S_OC
    gamma_r: float          # input-space rank-r evasion gain
    leverage: float         # ||a|| = ||W^T w_b||
    feasible: bool


def null_projector(M: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthogonal projector onto null(M) via SVD (M is d x F)."""
    F = M.shape[1]
    if M.size == 0:
        return np.eye(F)
    U_, sv, Vt = np.linalg.svd(M, full_matrices=True)
    tol = (sv.max() * rtol) if sv.size and sv.max() > 0 else 0.0
    rank = int((sv > tol).sum())
    V_null = Vt[rank:].T                  # (F, F-rank)
    if V_null.shape[1] == 0:
        return np.zeros((F, F))
    return V_null @ V_null.T


def input_evasion_gain(W: np.ndarray, w_b: np.ndarray, U: np.ndarray):
    """Return (Gamma_r, ||a||) for the input-space evasion.

    Gamma_r = ||P_null(P_U W) a|| / ||a||,  a = W^T w_b.
    """
    P_U = _subspace_projector(U)
    M = P_U @ W
    a = W.T @ w_b
    Pn = null_projector(M)
    na = np.linalg.norm(a)
    surv = np.linalg.norm(Pn @ a)
    gamma = float(surv / na) if na > 1e-12 else 0.0
    return gamma, float(na)


def _subspace_projector(U: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(U)
    return Q @ Q.T


def delta_star_analytic(W: np.ndarray, w_b: np.ndarray, U: np.ndarray,
                        beta: float = 1.0, tau: float = 0.0) -> EvasionResult:
    """Exact minimal input-space evading perturbation.

    W   : dictionary (d, F)
    w_b : behaviour direction in read space (d,), unit norm
    U   : monitor subspace basis (d, r)
    """
    P_U = _subspace_projector(U)
    M = P_U @ W
    a = W.T @ w_b
    na = float(np.linalg.norm(a))

    if tau <= 1e-15:
        Pn = null_projector(M)
        surv_vec = Pn @ a
        surv = float(np.linalg.norm(surv_vec))
        gamma = surv / na if na > 1e-12 else 0.0
        if surv <= 1e-12:
            return EvasionResult(np.inf, gamma, na, False)   # monitor robust
        delta = beta / surv
        return EvasionResult(float(delta), float(gamma), na, True)

    # tau > 0: SOCP; solve by penalty gradient descent (empirical) and report
    d_emp = delta_star_pgd(W, w_b, U, beta=beta, tau=tau)
    gamma, _ = input_evasion_gain(W, w_b, U)
    return EvasionResult(float(d_emp), float(gamma), na, np.isfinite(d_emp))


def delta_star_pgd(W: np.ndarray, w_b: np.ndarray, U: np.ndarray,
                   beta: float = 1.0, tau: float = 0.0,
                   steps: int = 800, restarts: int = 4, lr: float = 0.02,
                   penalty: float = 100.0, seed: int = 0) -> float:
    """Empirical minimal ||delta_z|| by penalty projected gradient descent.

    Minimises ||dz||^2 + penalty*[hinge(beta - a^T dz) + hinge(||M dz|| - tau)],
    with a hard projection of the monitor component when tau == 0. Positive
    control that the closed form is not an artefact.
    """
    rng = np.random.default_rng(seed)
    P_U = _subspace_projector(U)
    M = P_U @ W
    a = W.T @ w_b
    F = W.shape[1]
    Pn = null_projector(M) if tau <= 1e-15 else None
    best = np.inf
    for _ in range(max(restarts, 1)):
        dz = 0.01 * rng.standard_normal(F)
        if Pn is not None:
            dz = Pn @ dz
        for _ in range(steps):
            g = 2.0 * dz
            if a @ dz < beta:
                g = g - penalty * a
            if tau > 1e-15:
                mv = M @ dz
                nm = np.linalg.norm(mv)
                if nm > tau:
                    g = g + penalty * (M.T @ (mv / max(nm, 1e-12)))
            dz = dz - lr * g
            if Pn is not None:
                dz = Pn @ dz                     # stay monitor-invisible
        feas_behav = (a @ dz) >= beta - 1e-2
        feas_mon = (np.linalg.norm(M @ dz) <= tau + 1e-2)
        if feas_behav and feas_mon:
            best = min(best, float(np.linalg.norm(dz)))
    return best
