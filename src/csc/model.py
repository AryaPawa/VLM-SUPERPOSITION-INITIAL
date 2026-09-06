"""
csc.model
=========

The exact-geometry toy substrate for Phase A (R3's deliverable).

Read-position model: F latent features live as unit directions (columns of W) in
the d_eff-dimensional read-position space the safety monitor consumes.
Superposition is the geometric fact that F > d_eff, so the columns cannot be
mutually orthogonal; the Welch bound then forces a minimum coherence, and
coherence is what an attacker spends (Stevinson et al.; Gorton & Lewis).

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
    "random"  : columns i.i.d. Gaussian then unit-normalised. Fully controllable,
                no training confound. This is the substrate the G0 gate is
                validated on.
    "trained" : Elhage-style ReLU autoencoder trained under feature sparsity so
                superposition *emerges*. Repaired build (near-uniform importance,
                LR decay, dead-column re-seed). Still EXPERIMENTAL -- verify that
                coherence lands below 1 and above the Welch floor before relying
                on it. Used for the onset (alpha) story, not the core G0 pass.
    "token"   : the token-aggregation substrate (csc.tokens). N carrier tokens are
                pooled into a rank-limited read subspace; compression = fewer
                tokens. The honest N -> rho_rd bridge to the VLM harness.

N coupling (only relevant to "random"/"trained"): d_eff is the packing dimension.
coupling="identity" sweeps d_eff directly (N descriptive). coupling="kappa" ties
the read dimension to N / N0. In "token" mode N is *not* descriptive -- it drives
the pooled rank directly, so coupling is ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

from . import estimators as est
from .tokens import TokenParams, build_token_dictionary

__all__ = ["ToyConfig", "ToyModel"]


@dataclass
class ToyConfig:
    F: int = 128                 # number of ground-truth latent features
    d_eff: int = 32              # read-position (packing) dimension
    N: int = 32                  # carrier-token budget
    N0: int = 64                 # uncompressed token budget (for kappa = N / N0)
    d_model: int = 64            # cap on effective read dimension (kappa coupling)
    s: float = 0.05              # feature sparsity (P[feature active]); Elhage regime s << 1
    geometry: Literal["random", "trained", "token"] = "random"
    coupling: Literal["identity", "kappa"] = "identity"
    monitor_rank: int = 4        # r in the rank-r monitor subspace
    monitor_angle_deg: float = 30.0  # planted principal angle between behaviour and monitor
    seed: int = 0
    # token geometry ------------------------------------------------------ #
    k_per_token: int = 2         # read directions each carrier token contributes
    # training (geometry="trained") --------------------------------------- #
    train_steps: int = 4000
    train_lr: float = 1e-2
    train_batch: int = 2048
    train_importance_decay: float = 0.995   # ~uniform: keep ALL features represented
    train_dead_floor: float = 0.05          # re-seed columns whose norm < floor * median

    def effective_d(self) -> int:
        """Resolve the read dimension used, given the N coupling (non-token modes)."""
        if self.coupling == "identity":
            return int(self.d_eff)
        kappa = self.N / max(self.N0, 1)
        return max(1, min(self.d_model, int(round(kappa * self.d_model))))


class ToyModel:
    """Exact-geometry read-position model with a planted behaviour/monitor pair."""

    def __init__(self, cfg: ToyConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.F = cfg.F
        # token-substrate diagnostics (populated only in token mode)
        self.r_pool: int | None = None
        self.rho_blk: float | None = None

        if cfg.geometry == "random":
            self.d = cfg.effective_d()
            self.W = self._random_dictionary()
        elif cfg.geometry == "trained":
            self.d = cfg.effective_d()
            self.W = self._trained_dictionary()
        elif cfg.geometry == "token":
            self.d = int(cfg.d_eff)          # read width is FIXED (Variant B)
            build = build_token_dictionary(TokenParams(
                F_ref=cfg.F, N=cfg.N, N_ref=cfg.N0, d_eff=cfg.d_eff,
                k_per_token=cfg.k_per_token, seed=cfg.seed,
            ))
            self.W = build.W
            self.F = build.F_rd              # effective read features grow as N falls
            self.r_pool = build.d_eff        # full-width read; packing dim == d_eff
            self.rho_blk = build.rho_blk
        else:  # pragma: no cover
            raise ValueError(f"unknown geometry {cfg.geometry!r}")

    # ---------------------------------------------------------------- setup #
    def _random_dictionary(self) -> np.ndarray:
        W = self.rng.standard_normal((self.d, self.F))
        return est.normalize_columns(W)

    def _trained_dictionary(self) -> np.ndarray:
        """Elhage ReLU autoencoder: reconstruct sparse features through a d-dim
        bottleneck so superposition emerges when F > d.

        Repairs vs the first pass (which collapsed to coherence ~= 1):
          * near-uniform importance (decay ~ 0.995) so every feature is
            represented, not just the top few;
          * cosine LR decay for a cleaner minimum;
          * post-hoc re-seed of any dead columns (norm below a floor) as fresh
            random unit vectors, so a handful of unused features cannot spike the
            worst-case coherence to 1.
        """
        cfg = self.cfg
        rng = self.rng
        d, F = self.d, self.F
        imp = cfg.train_importance_decay ** np.arange(F)
        imp = imp / imp.max()                                 # normalise importances
        W = 0.1 * rng.standard_normal((d, F))
        b = np.zeros(F)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        mb = np.zeros_like(b); vb = np.zeros_like(b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, cfg.train_steps + 1):
            lr = cfg.train_lr * 0.5 * (1 + np.cos(np.pi * (t - 1) / cfg.train_steps))
            mask = (rng.random((cfg.train_batch, F)) < cfg.s)
            z = mask * rng.random((cfg.train_batch, F))       # (B, F) sparse
            h = z @ W.T                                       # (B, d) bottleneck
            pre = h @ W + b                                   # (B, F)
            xhat = np.maximum(pre, 0.0)
            relu_mask = (pre > 0.0).astype(z.dtype)
            err = (xhat - z) * imp
            g_pre = 2.0 * err * relu_mask / cfg.train_batch
            gb = g_pre.sum(axis=0)
            gW = (g_pre.T @ h).T + (z.T @ (g_pre @ W.T)).T    # both W-paths
            for (p, g, m, v) in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= b1; m += (1 - b1) * g
                v *= b2; v += (1 - b2) * (g * g)
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)
        # re-seed dead columns so unused features do not fake high coherence
        norms = np.linalg.norm(W, axis=0)
        med = np.median(norms[norms > 0]) if np.any(norms > 0) else 1.0
        dead = norms < (cfg.train_dead_floor * med)
        if dead.any():
            W[:, dead] = rng.standard_normal((d, int(dead.sum())))
        return est.normalize_columns(W)

    # ------------------------------------------------------------- geometry #
    def representation(self, z: np.ndarray) -> np.ndarray:
        """Read-position activation h = W z."""
        return self.W @ z

    def coherence(self) -> float:
        return est.coherence(self.W)

    def welch_floor(self) -> float:
        # In token mode the true packing dimension is the pooled rank r_pool.
        d_pack = self.r_pool if (self.r_pool is not None) else self.d
        return est.welch_floor(self.F, d_pack)

    def coherence_gap(self) -> float:
        d_pack = self.r_pool if (self.r_pool is not None) else self.d
        return est.coherence(self.W) - est.welch_floor(self.F, d_pack)

    # ------------------------------------------------ behaviour / monitor #
    def plant_pair(self, rng: np.random.Generator | None = None
                   ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Plant a (behaviour feature, rank-r monitor subspace) probe.

        The behaviour direction is a feature column w_b. The monitor subspace U
        is planted at a controllable principal angle to w_b: one basis vector at
        ``monitor_angle_deg`` to w_b, the remaining r-1 random. Smaller angle ->
        monitor more aligned -> lower evasion gain -> costlier attack.
        """
        rng = self.rng if rng is None else rng
        cfg = self.cfg
        b_idx = int(rng.integers(self.F))
        w_b = self.W[:, b_idx].copy()
        w_b /= np.linalg.norm(w_b)

        theta = np.deg2rad(cfg.monitor_angle_deg)
        rand = rng.standard_normal(self.d)
        rand -= (rand @ w_b) * w_b
        n = np.linalg.norm(rand)
        perp = rand / n if n > 1e-12 else self._any_orthogonal(w_b)
        u1 = np.cos(theta) * w_b + np.sin(theta) * perp

        r = min(cfg.monitor_rank, self.d)
        U = np.zeros((self.d, r))
        U[:, 0] = u1
        for k in range(1, r):
            U[:, k] = rng.standard_normal(self.d)
        Q, _ = np.linalg.qr(U)
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
        rng = self.rng if rng is None else rng
        mask = rng.random((n, self.F)) < self.cfg.s
        return mask * rng.random((n, self.F))

    def read_activations(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        Z = self.sample_features(n, rng)
        return Z @ self.W.T