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


# =========================================================================== #
# Gradient-masking defense probe (R2 machinery, exercised in the toy).
#
# A "soft top-k" gate at the read position is a defense that LOOKS robust to a
# naive gradient attacker but isn't: the gate zeroes small coordinates, so the
# honest gradient of the behaviour read-out is ~0 exactly where the attacker
# needs to push. A naive PGD therefore fails to raise the behaviour and reports a
# large perturbation ("safe"). BPDA (straight-through backward pass) and a
# gradient-free search push through the gate's cliff and recover the true, small
# perturbation. This is the Athalye et al. obfuscated-gradients check in
# miniature: it validates that our S_OC estimates measure real erosion, not a
# gradient artefact.
#
# The probe is behaviour-only (raise w_b . gate(h) by >= beta) and operates at
# the read position (eta in R^{d_eff}); it is deliberately separate from the
# input-space S_OC above, which already handles monitor evasion. A baseline h0 is
# constructed so the behaviour-carrying coordinates sit just BELOW the gate
# threshold -- the regime where masking bites.
# =========================================================================== #

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


@dataclass
class MaskingResult:
    S_nogate: float     # min ||eta|| with NO gate (linear reference)
    S_naive: float      # masked PGD, honest gradient (inflated -> looks safe)
    S_bpda: float       # straight-through backward pass (recovers true attack)
    S_gradfree: float   # gradient-free search (independent confirmation)
    budget: float       # perturbation cap; an attacker at the cap "failed"

    @property
    def masking_ratio(self) -> float:
        base = min(self.S_bpda, self.S_gradfree)
        return float(self.S_naive / max(base, 1e-12))


def soft_top_k(h: np.ndarray, k: int, temp: float = 0.05) -> np.ndarray:
    """Soft gate keeping the top-k coordinates by magnitude (differentiable)."""
    k = int(np.clip(k, 1, h.shape[-1]))
    thr = np.sort(np.abs(h))[-k]
    return _sigmoid((np.abs(h) - thr) / temp) * h


def _gate_diag_jac(h: np.ndarray, k: int, temp: float) -> np.ndarray:
    """Diagonal of d(soft_top_k(h))/dh (threshold treated as constant)."""
    k = int(np.clip(k, 1, h.shape[-1]))
    thr = np.sort(np.abs(h))[-k]
    g = _sigmoid((np.abs(h) - thr) / temp)
    dg = g * (1.0 - g) * np.sign(h) / temp
    return g + h * dg


def _behaviour(h, w_b, k, temp):
    return float(w_b @ soft_top_k(h, k, temp))


def _pgd_behaviour(h0, w_b, k, temp, beta, straight_through, budget,
                   steps=2000, restarts=6, lr=0.05, penalty=50.0, seed=0):
    """Min ||eta|| s.t. gated behaviour rises by >= beta. Returns the achieved
    ||eta|| if feasible, else ``budget`` (i.e. the attacker could not do better).
    """
    rng = np.random.default_rng(seed)
    d = h0.shape[0]
    beh0 = _behaviour(h0, w_b, k, temp)
    best = budget
    for _ in range(restarts):
        eta = 0.01 * rng.standard_normal(d)
        for _ in range(steps):
            h = h0 + eta
            beh = _behaviour(h, w_b, k, temp)
            jac = np.ones(d) if straight_through else _gate_diag_jac(h, k, temp)
            g = 2.0 * eta
            if (beh - beh0) < beta:
                g = g - penalty * (w_b * jac)
            eta = eta - lr * g
            n = np.linalg.norm(eta)
            if n > budget:                       # stay within the perturbation cap
                eta *= budget / n
        if (_behaviour(h0 + eta, w_b, k, temp) - beh0) >= beta - 1e-2:
            best = min(best, float(np.linalg.norm(eta)))
    return float(best)


def _gradfree_behaviour(h0, w_b, k, temp, beta, budget, samples=3000, seed=0):
    """Gradient-free confirmation: search directions, line-search the magnitude
    that first satisfies the gated behaviour target; keep the smallest ||eta||.
    """
    rng = np.random.default_rng(seed)
    d = h0.shape[0]
    beh0 = _behaviour(h0, w_b, k, temp)
    best = budget
    for i in range(samples):
        v = w_b + 0.5 * rng.standard_normal(d) if i % 2 == 0 else rng.standard_normal(d)
        v = v / (np.linalg.norm(v) + 1e-12)
        for c in np.linspace(0.05, budget, 60):
            if (_behaviour(h0 + c * v, w_b, k, temp) - beh0) >= beta:
                best = min(best, float(c)); break
    return float(best)


def masking_demo(W: np.ndarray, w_b: np.ndarray, U: np.ndarray | None = None,
                 k: int | None = None, temp: float = 0.02, beta: float = 1.0,
                 budget: float = 20.0, V: float = 3.0, seed: int = 0) -> MaskingResult:
    """Run the four behaviour-raising attackers on one planted pair.

    Controlled read-position scenario (validates the detection machinery): the
    behaviour direction is concentrated on the few coordinates where |w_b| is
    largest, and the top-k gate slots are filled by DISTRACTOR coordinates that
    carry ~no behaviour weight. The behaviour coordinates start gated OFF (below
    threshold), so raising the behaviour requires pushing them across the gate's
    cliff. The honest gradient there is ~0 (masked), so the naive attacker
    stalls at the budget cap ("looks safe"); BPDA and the gradient-free search
    push through and recover the true, small perturbation. ``U`` is accepted for
    interface symmetry but unused (behaviour-only probe).
    """
    d = W.shape[0]
    if k is None:
        k = max(2, d // 4)
    m = max(1, d // 8)
    order = np.argsort(np.abs(w_b))          # ascending |w_b|
    beh_coords = order[-m:]                   # concentrate behaviour on largest |w_b|
    dist_coords = order[:k]                   # distractors: smallest |w_b| (~0 weight)
    w_hat = np.zeros(d)
    w_hat[beh_coords] = w_b[beh_coords]
    nrm = np.linalg.norm(w_hat)
    w_hat = w_hat / nrm if nrm > 1e-12 else w_b / np.linalg.norm(w_b)
    h0 = np.zeros(d)
    h0[dist_coords] = V                       # distractors ON -> threshold ~ V
    # behaviour coords remain at 0 -> gated off

    S_nogate = float(beta / max(np.linalg.norm(w_hat), 1e-12))
    S_naive = _pgd_behaviour(h0, w_hat, k, temp, beta, False, budget, seed=seed)
    S_bpda = _pgd_behaviour(h0, w_hat, k, temp, beta, True, budget, seed=seed)
    S_gf = _gradfree_behaviour(h0, w_hat, k, temp, beta, budget, seed=seed)
    return MaskingResult(S_nogate=S_nogate, S_naive=S_naive, S_bpda=S_bpda,
                         S_gradfree=S_gf, budget=budget)


__all__ += ["MaskingResult", "soft_top_k", "masking_demo"]