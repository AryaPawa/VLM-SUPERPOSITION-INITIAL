"""
csc.tokens
==========

The token-aggregation substrate (R3 -> R4 bridge). This is the honest version of
the compression axis: instead of tying N to d_eff by fiat, we build *N carrier
tokens*, pool them into the read position the monitor consumes, and then MEASURE
the read-position pressure rho_rd that results. The claim "compressing tokens
raises read-position superposition" (N -> rho_rd) becomes an empirical output of
the construction rather than an assumption.

Mechanism (faithful to a pooled VLM read-out)
---------------------------------------------
Each of the N tokens contributes ``k_per_token`` independent read directions in
the d_eff-dimensional read space. The pooled read subspace is the span of all of
them, so its rank is

    r_pool = min(N * k_per_token, d_eff).

The F latent features are written into this pooled subspace (coordinates C in
R^{r_pool}), so the read dictionary is

    W_rd = Q @ C          Q in R^{d_eff x r_pool} orthonormal (the pooled basis)

Fewer tokens -> smaller r_pool -> the same F features are forced into fewer
effective read dimensions -> coherence rises toward the Welch floor for load
F / r_pool, the measured d_eff falls, and rho_rd = F_eff / d_eff rises. The
behaviour leverage ||a|| = ||W_rd^T w_b|| then grows and the minimal evading
perturbation S falls -- exactly the chain, now driven by the token budget N.

Block vs read pressure (review split)
-------------------------------------
    rho_blk = F / (N * k_per_token)     # pressure across the whole token block
    rho_rd  = F_eff / d_eff (measured)  # pressure at the read position

The N -> rho_rd arrow is the empirical aggregation claim; this module is where we
get to watch it happen in a setting where everything is still exact.

The output W_rd is an ordinary (d_eff x F) unit-column dictionary, so every
downstream tool (plant_pair, delta_star_analytic, the estimators, the gate)
consumes it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import estimators as est

__all__ = ["TokenParams", "build_token_dictionary", "TokenBuild"]


@dataclass
class TokenParams:
    F: int = 128            # number of latent features to encode
    N: int = 32             # number of carrier tokens (the compression axis)
    d_eff: int = 32         # read-position dimension the monitor consumes (fixed)
    k_per_token: int = 2    # independent read directions each token contributes
    seed: int = 0

    def r_pool(self) -> int:
        """Rank of the pooled read subspace = min(N * k_per_token, d_eff)."""
        return int(min(self.N * self.k_per_token, self.d_eff))

    def rho_blk(self) -> float:
        """Block-level pressure across the whole token budget."""
        return float(self.F / max(self.N * self.k_per_token, 1))


@dataclass
class TokenBuild:
    W: np.ndarray           # (d_eff, F) unit-column read dictionary
    r_pool: int             # rank of the pooled read subspace
    rho_blk: float          # F / (N * k_per_token)
    Q: np.ndarray           # (d_eff, r_pool) orthonormal pooled basis


def build_token_dictionary(p: TokenParams) -> TokenBuild:
    """Construct the pooled read dictionary for a token budget N.

    Steps
    -----
    1. Each token n gets a random block B_n in R^{d_eff x k_per_token}.
    2. Stack the N blocks and orthonormalise -> pooled basis Q of rank r_pool
       (this is the "mean/attention pool collapses N tokens into a rank-limited
       read" step; rank is capped at d_eff).
    3. Draw feature coordinates C in R^{r_pool x F} and map to read space via Q.
    4. Unit-normalise the columns -> W_rd.
    """
    rng = np.random.default_rng(p.seed)
    k = max(1, p.k_per_token)

    # (1) per-token read blocks, then (2) pool -> orthonormal basis of rank r_pool
    B = rng.standard_normal((p.d_eff, p.N * k))
    # economy QR gives an orthonormal basis for the column span; its rank is
    # min(d_eff, N*k) = r_pool. Take the first r_pool columns.
    Q_full, _ = np.linalg.qr(B)
    r_pool = p.r_pool()
    Q = Q_full[:, :r_pool]

    # (3) features live in the pooled subspace; (4) normalise columns
    C = rng.standard_normal((r_pool, p.F))
    W = Q @ C
    W = est.normalize_columns(W)
    return TokenBuild(W=W, r_pool=r_pool, rho_blk=p.rho_blk(), Q=Q)
