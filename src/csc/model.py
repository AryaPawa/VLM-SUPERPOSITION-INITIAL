"""
csc.model
=========

The exact-geometry toy substrate for Phase A (R3's deliverable).

This is the *direct read-position* model: F latent features live as unit
directions (columns of W) in the d_eff-dimensional read-position space that the
safety monitor consumes. Superposition is the geometric fact that F > d_eff, so
the columns cannot be mutually orthogonal; the Welch bound then forces a minimum
coherence, and coherence is what an attacker spends (Stevinson et al.;
Gorton & Lewis).

Frozen Toy Model API (v0.1) -- coordination boundary for R1/R2/R4:

    cfg   = ToyConfig(...)
    model = ToyModel(cfg)
    h     = model.representation(z)          # (d_eff,)  read-position activation
    model.coherence()                        # scalar mutual coherence
    model.welch_floor()                      # scalar Welch coherence floor
    model.plant_pair(rng)  -> (b_idx, w_b, U)   # a (behaviour, monitor-subspace) probe
    model.gamma_r(w_b, U)                    # rank-r evasion gain
    model.log_alignment(w_b, U)              # A = -log(1 - Gamma_r^2)

Geometry modes:
    "random"  : columns i.i.d. Gaussian then unit-normalised (near-Welch scaling,
                fully controllable, no training confound).
    "trained" : Elhage-style ReLU autoencoder trained under feature sparsity so
                superposition *emerges*; demonstrates onset. (numpy Adam, light.)

N coupling (documented, swappable): d_eff is the true packing dimension. The
carrier-token budget N enters through ``d_eff_from_N``; the default ties the
effective read dimension to the compression ratio kappa = N / N0, capped at the
model width d_model. Set coupling="identity" to sweep d_eff directly and hold N
purely descriptive. This is the single Phase-A modelling choice flagged for
review; the richer token-aggregation substrate is the next-iteration upgrade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np

from . import estimators as est

__all__ = ["ToyConfig", "ToyModel"]


@dataclass
class ToyConfig:
    F: int = 128                 # number of ground-truth latent features
    d_eff: int = 32              # read-position (packing) dimension
    N: int = 32                  # carrier-token budget (descriptive unless coupled)
    N0: int = 64                 # uncompressed token budget (for kappa = N / N0)
    d_model: int = 64            # cap on effective read dimension
    s: float = 0.05              # feature sparsity (P[feature active]); Elhage regime s << 1
    geometry: Literal["random", "trained"] = "random"
    coupling: Literal["identity", "kappa"] = "identity"
    monitor_rank: int = 4        # r in the rank-r monitor subspace
    monitor_angle_deg: float = 30.0  # planted principal angle between behaviour and monitor
    seed: int = 0
    # training (geometry="trained") --------------------------------------- #
    train_steps: int = 2000
    train_lr: float = 1e-2
    train_batch: int = 2048
    train_importance_decay: float = 0.9  # Elhage feature-importance geometric decay

    def effective_d(self) -> int:
        """Resolve the read dimension actually used, given the N coupling."""
        if self.coupling == "identity":
            return int(self.d_eff)
        # kappa coupling: shrink the read dimension with the token budget.
        kappa = self.N / max(self.N0, 1)
        return max(1, min(self.d_model, int(round(kappa * self.d_model))))


class ToyModel:
    """Exact-geometry read-position model with a planted behaviour/monitor pair."""

    def __init__(self, cfg: ToyConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.d = cfg.effective_d()
        self.F = cfg.F
        if cfg.geometry == "random":
            self.W = self._random_dictionary()
        elif cfg.geometry == "trained":
            self.W = self._trained_dictionary()
        else:  # pragma: no cover
            raise ValueError(f"unknown geometry {cfg.geometry!r}")

    # ---------------------------------------------------------------- setup #
    def _random_dictionary(self) -> np.ndarray:
        W = self.rng.standard_normal((self.d, self.F))
        return est.normalize_columns(W)

    def _trained_dictionary(self) -> np.ndarray:
        """Train an Elhage ReLU autoencoder: minimise importance-weighted
        reconstruction of sparse features through a d-dim bottleneck.

        x = z (sparse, F-dim);  h = W z (bottleneck, d-dim);
        x_hat = ReLU(W^T h + b);  loss = sum_i imp_i (x_i - x_hat_i)^2.
        Superposition emerges when F > d and s is small.
        """
        cfg = self.cfg
        rng = self.rng
        d, F = self.d, self.F
        imp = cfg.train_importance_decay ** np.arange(F)      # feature importances
        W = 0.1 * rng.standard_normal((d, F))
        b = np.zeros(F)
        # Adam state
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        mb = np.zeros_like(b); vb = np.zeros_like(b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, cfg.train_steps + 1):
            # sample sparse feature batch: active w.p. s, magnitude ~ U[0,1]
            mask = (rng.random((cfg.train_batch, F)) < cfg.s)
            z = mask * rng.random((cfg.train_batch, F))          # (B, F)
            h = z @ W.T                                          # (B, d)
            pre = h @ W + b                                      # (B, F)
            xhat = np.maximum(pre, 0.0)
            relu_mask = (pre > 0.0).astype(z.dtype)
            err = (xhat - z) * imp                               # weighted residual (B, F)
            # grads
            g_pre = 2.0 * err * relu_mask / cfg.train_batch      # (B, F)
            gb = g_pre.sum(axis=0)
            gW = (g_pre.T @ h).T + (z.T @ (g_pre @ W.T)).T       # d(loss)/dW via both paths
            # Adam update
            for (p, g, m, v) in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= b1; m += (1 - b1) * g
                v *= b2; v += (1 - b2) * (g * g)
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                p -= cfg.train_lr * mhat / (np.sqrt(vhat) + eps)
        return est.normalize_columns(W)

    # ------------------------------------------------------------- geometry #
    def representation(self, z: np.ndarray) -> np.ndarray:
        """Read-position activation h = W z."""
        return self.W @ z

    def coherence(self) -> float:
        return est.coherence(self.W)

    def welch_floor(self) -> float:
        return est.welch_floor(self.F, self.d)

    def coherence_gap(self) -> float:
        return est.coherence_gap(self.W, self.F, self.d)

    # ------------------------------------------------ behaviour / monitor #
    def plant_pair(self, rng: np.random.Generator | None = None
                   ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Plant a (behaviour feature, rank-r monitor subspace) probe.

        The behaviour direction is a feature column w_b. The monitor subspace U
        is planted to make a controllable *principal angle* with w_b: one basis
        vector is placed at angle ``monitor_angle_deg`` to w_b (so the monitor
        partially -- but not perfectly -- watches the behaviour), the remaining
        r-1 directions are random. Smaller angle -> monitor more aligned ->
        lower evasion gain -> costlier attack.
        """
        rng = self.rng if rng is None else rng
        cfg = self.cfg
        b_idx = int(rng.integers(self.F))
        w_b = self.W[:, b_idx].copy()
        w_b /= np.linalg.norm(w_b)

        # direction 1 of U at the planted angle to w_b, within a random plane.
        theta = np.deg2rad(cfg.monitor_angle_deg)
        rand = rng.standard_normal(self.d)
        rand -= (rand @ w_b) * w_b                       # component orthogonal to w_b
        n = np.linalg.norm(rand)
        perp = rand / n if n > 1e-12 else self._any_orthogonal(w_b)
        u1 = np.cos(theta) * w_b + np.sin(theta) * perp

        r = min(cfg.monitor_rank, self.d)
        U = np.zeros((self.d, r))
        U[:, 0] = u1
        for k in range(1, r):
            U[:, k] = rng.standard_normal(self.d)
        Q, _ = np.linalg.qr(U)                           # orthonormalise the subspace
        return b_idx, w_b, Q[:, :r]

    @staticmethod
    def _any_orthogonal(v: np.ndarray) -> np.ndarray:
        e = np.zeros_like(v); e[0] = 1.0
        if abs(v[0]) > 0.9:
            e[:] = 0.0; e[1 % len(v)] = 1.0
        w = e - (e @ v) * v
        return w / np.linalg.norm(w)

    def gamma_r(self, w_b: np.ndarray, U: np.ndarray) -> float:
        return est.gamma_r(w_b, U)

    def log_alignment(self, w_b: np.ndarray, U: np.ndarray) -> float:
        return est.log_alignment(self.gamma_r(w_b, U))

    # ---------------------------------------------------------- diagnostics #
    def sample_features(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw n sparse feature vectors (for d_eff / F_eff estimation)."""
        rng = self.rng if rng is None else rng
        mask = rng.random((n, self.F)) < self.cfg.s
        return mask * rng.random((n, self.F))

    def read_activations(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Read-position activations for n sampled inputs: (n, d_eff)."""
        Z = self.sample_features(n, rng)
        return Z @ self.W.T
