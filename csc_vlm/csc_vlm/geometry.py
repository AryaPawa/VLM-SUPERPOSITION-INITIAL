"""
csc_vlm.geometry
================

Model-agnostic read-space geometry, with the two estimator helpers we need
inlined so the package is standalone.

Safety definition (same as the toy's Proposition 1, computed on real activations):

    a       = Sigma^{1/2} w_b            leverage vector
    ||a||   = leverage                    how much the behaviour spreads
    Gamma_r = ||P_perp_U a|| / ||a||      escape fraction from the monitor U
    S       = beta / ||P_perp_U a||       obfuscation cost  =  beta / (Gamma_r * ||a||)

S falls as compression raises superposition -> leverage up, monitor easier to
evade -> lower obfuscation cost. Closed form on the measured covariance (cheap,
forward-only at the model).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = ["normalize_columns", "d_eff", "cov_half", "fit_monitor",
           "measured_geometry"]


# ------------------------------------------------------------------ estimators #
def normalize_columns(W: np.ndarray) -> np.ndarray:
    """Unit-norm each column."""
    return W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)


def d_eff(H: np.ndarray) -> float:
    """Participation ratio of the (centred) activation energy spectrum:
    d_eff = (sum lambda_i)^2 / sum lambda_i^2. Read dimensionality proxy."""
    Hc = H - H.mean(0, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    ev = s ** 2
    tot = ev.sum()
    if tot <= 0:
        return 1.0
    return float((tot ** 2) / (ev ** 2).sum())


def cov_half(H: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    """Sigma^{1/2} with Ledoit-Wolf-style diagonal loading. Shrinkage is ESSENTIAL
    when n < d (few activation samples in a high-dim read space -- the real-VLM
    regime): it keeps the covariance full-rank so Sigma^{1/2} and the obfuscation
    cost are stable across resamples."""
    Hc = H - H.mean(0, keepdims=True)
    Sig = (Hc.T @ Hc) / max(1, H.shape[0] - 1)
    d = Sig.shape[0]
    mu = np.trace(Sig) / d
    Sig = (1.0 - shrink) * Sig + shrink * mu * np.eye(d)
    w, V = np.linalg.eigh(Sig)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


# --------------------------------------------------------------- monitor + S #
def fit_monitor(H: np.ndarray, y: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """Behaviour direction w_b = normalised class-mean difference; monitor U =
    orthonormal [w_b, top principal directions] of rank r. Fit ONCE on
    uncompressed activations, then held fixed across compression (persistence)."""
    y = np.asarray(y)
    mu1 = H[y == 1].mean(0) if (y == 1).any() else H.mean(0)
    mu0 = H[y == 0].mean(0) if (y == 0).any() else np.zeros(H.shape[1])
    w_b = mu1 - mu0
    w_b = w_b / (np.linalg.norm(w_b) + 1e-12)
    Hc = H - H.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    basis = [w_b]
    for k in range(Vt.shape[0]):
        v = Vt[k]
        for b in basis:                                  # Gram-Schmidt vs current basis
            v = v - (v @ b) * b
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            basis.append(v / nv)
        if len(basis) >= rank:
            break
    U = np.stack(basis[:rank], axis=1)                   # [d, r]
    return w_b, U


def measured_geometry(H: np.ndarray, w_b: np.ndarray, U: np.ndarray,
                      beta: float = 1.0, shrink: float = 0.2) -> Dict[str, float]:
    """Measured mediators + obfuscation cost from the activation covariance."""
    Sh = cov_half(H, shrink=shrink)
    a = Sh @ w_b
    lev = float(np.linalg.norm(a))
    P_U = U @ U.T
    esc = a - P_U @ a
    escn = float(np.linalg.norm(esc))
    gamma = float(escn / (lev + 1e-12))
    obf = float(beta / max(escn, 1e-9))
    return {"leverage": lev, "gamma_r": gamma, "d_eff": float(d_eff(H)),
            "obf_cost": obf}
