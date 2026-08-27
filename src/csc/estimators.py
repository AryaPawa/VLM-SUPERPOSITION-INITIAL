"""
csc.estimators
==============

Exact-geometry estimators for the Phase-A toy substrate.

These implement the "estimator definitions in implementable form" from the
adversarial review (Appendix B), specialised to the toy model where every
quantity is available in closed form rather than estimated:

    rho_rd        F_eff / d_eff at the read position   (primary pressure variable)
    d_eff         participation ratio of the activation covariance
    coherence     max_{i!=j} |<w_i, w_j>| over unit feature directions
    welch_floor   sqrt((F - d) / (d (F - 1)))          (Welch 1974 coherence lower bound)
    coherence_gap coherence - welch_floor              (report the gap, not the raw value)
    gamma_r       || P_{U_perp} w_b || / || w_b ||     (rank-r evasion gain)
    log_alignment A = -log(1 - gamma_r^2)              (the mediator that enters regressions)

Everything is numpy; no autograd, no heavy deps. Feature directions are the
columns of W in R^{d_eff x F}. Unless stated otherwise, columns are treated as
the read-position directions of individual latent features.

References (verified in the plan):
  - L. R. Welch, IEEE T-IT 20(3):397-399, 1974 (coherence lower bound).
  - Elhage et al., "Toy Models of Superposition," Transformer Circuits 2022.
  - Review Appendix B, "Estimator definitions in implementable form."
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_columns",
    "gram",
    "coherence",
    "welch_floor",
    "coherence_gap",
    "participation_ratio",
    "d_eff",
    "f_eff",
    "rho_rd",
    "subspace_projector",
    "gamma_r",
    "log_alignment",
]


# --------------------------------------------------------------------------- #
# Basic geometry
# --------------------------------------------------------------------------- #
def normalize_columns(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return W with each column scaled to unit L2 norm."""
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    return W / np.maximum(norms, eps)


def gram(W: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Gram matrix of the feature directions (columns of W).

    If ``normalized`` the columns are unit-normalised first, so ``gram`` holds
    pairwise cosines.
    """
    Wn = normalize_columns(W) if normalized else W
    return Wn.T @ Wn


def coherence(W: np.ndarray) -> float:
    """Worst-case mutual coherence mu = max_{i != j} |<w_i, w_j>| (unit columns)."""
    G = gram(W, normalized=True)
    np.fill_diagonal(G, 0.0)
    return float(np.max(np.abs(G)))


def welch_floor(F: int, d: int) -> float:
    """Welch lower bound on coherence for F unit vectors in R^d.

    mu >= sqrt((F - d) / (d (F - 1)))  for F > d.  For F <= d an orthonormal
    set exists and the floor is 0.
    """
    if F <= d:
        return 0.0
    return float(np.sqrt((F - d) / (d * (F - 1))))


def coherence_gap(W: np.ndarray, F: int | None = None, d: int | None = None) -> float:
    """Empirical coherence minus the Welch floor (>= 0 up to numerical slack).

    Report *the gap*, per the review: a small gap means the frame is near the
    packing limit for its load, so its coherence is forced by geometry rather
    than by a bad construction.
    """
    d = W.shape[0] if d is None else d
    F = W.shape[1] if F is None else F
    return coherence(W) - welch_floor(F, d)


# --------------------------------------------------------------------------- #
# Effective dimension / effective feature count / pressure
# --------------------------------------------------------------------------- #
def participation_ratio(cov_or_acts: np.ndarray, is_cov: bool = False) -> float:
    """Participation ratio d_eff = (sum lambda_i)^2 / sum lambda_i^2.

    Accepts either a covariance matrix (``is_cov=True``) or a stack of
    activations of shape (n_samples, d) whose covariance is formed internally.
    """
    if is_cov:
        cov = cov_or_acts
    else:
        A = cov_or_acts
        A = A - A.mean(axis=0, keepdims=True)
        cov = (A.T @ A) / max(A.shape[0] - 1, 1)
    lam = np.linalg.eigvalsh(cov)
    lam = np.clip(lam, 0.0, None)
    s1 = lam.sum()
    s2 = (lam ** 2).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 ** 2) / s2)


def d_eff(acts: np.ndarray) -> float:
    """Effective dimension of read-position activations (participation ratio)."""
    return participation_ratio(acts, is_cov=False)


def f_eff(acts_features: np.ndarray, floor: float = 1e-4) -> int:
    """Number of features with activation frequency above ``floor``.

    ``acts_features`` is (n_samples, F): the per-feature activation magnitudes.
    A feature counts if it is active (|a| > floor) on a non-negligible fraction
    of samples, matching the review's pre-registered activation-frequency floor.
    """
    active_freq = (np.abs(acts_features) > floor).mean(axis=0)
    return int((active_freq > floor).sum())


def rho_rd(F_eff_val: float, d_eff_val: float, eps: float = 1e-12) -> float:
    """Read-position superposition pressure rho_rd = F_eff / d_eff."""
    return float(F_eff_val / max(d_eff_val, eps))


# --------------------------------------------------------------------------- #
# Rank-r evasion gain and log-alignment
# --------------------------------------------------------------------------- #
def subspace_projector(U: np.ndarray) -> np.ndarray:
    """Orthogonal projector P_U onto the column space of U (d x r).

    Uses an economy QR so U need not have orthonormal columns.
    """
    Q, _ = np.linalg.qr(U)
    return Q @ Q.T


def gamma_r(w_b: np.ndarray, U: np.ndarray, eps: float = 1e-12) -> float:
    """Rank-r evasion gain Gamma_r = || P_{U_perp} w_b || / || w_b ||.

    w_b : behaviour direction in read space (d,)
    U   : monitor subspace basis (d, r)

    Gamma_r -> 1 : behaviour is (almost) orthogonal to the monitor subspace,
                   so it can be induced without moving the monitor -> evasion is cheap.
    Gamma_r -> 0 : behaviour lies inside the monitor subspace -> monitor is robust.
    """
    P_U = subspace_projector(U)
    d = w_b.shape[0]
    P_perp = np.eye(d) - P_U
    num = np.linalg.norm(P_perp @ w_b)
    den = np.linalg.norm(w_b)
    return float(num / max(den, eps))


def log_alignment(gamma: float, eps: float = 1e-12) -> float:
    """Log-alignment mediator A = -log(1 - Gamma_r^2).

    Spreads the informative region of Gamma across the real line so the
    mediator has usable variance across cells (review 3.3, repair ii).
    """
    g2 = min(max(gamma * gamma, 0.0), 1.0 - eps)
    return float(-np.log(1.0 - g2))
