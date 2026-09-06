"""
csc.tokens
==========

The token-aggregation substrate (R3 -> R4 bridge), **Variant B**.

Goal: an honest compression axis where reducing the token budget N raises
read-position pressure rho_rd *and* lowers safety S, with the correct sign.

Why not shrink the read subspace (Variant A -- rejected)
--------------------------------------------------------
The tempting design is "fewer tokens -> lower-rank pooled read subspace"
(r_pool = min(N*k, d_eff)). It has a fatal confound: the safety monitor is a
FIXED rank-r subspace. Once r_pool falls to <= r, the monitor covers the entire
feature subspace, the evasion gain Gamma_r collapses to 0, and S -> infinity --
so heavy compression looks *infinitely safe*. That inverts the relationship for
a substrate reason, not a real one. (Observed: at small N the monitor over-covers
the tiny subspace and every attack is infeasible.)

Variant B (this module)
-----------------------
Keep the read width d_eff FIXED (so the rank-r monitor always covers the same
fraction r/d_eff -> Gamma_r stays steady). Model compression by MERGING MORE
FEATURES into that fixed-width read as tokens are removed:

    F_rd(N) = round(F_ref * N_ref / N)         (capped for tractability)

Fewer tokens -> more features share the same d_eff directions -> coherence up,
behaviour leverage ||a|| = ||W^T w_b|| up, while Gamma_r is held ~constant by the
fixed monitor coverage -> S = beta / (Gamma_r * ||a||) falls monotonically. This
is the faithful "pooling merges features into a fixed read" picture, and it
reuses the validated exact-geometry frame (unit columns in R^{d_eff}); the only
token-specific ingredient is the compression->load map F_rd(N).

    rho_blk = F_rd / (N * k_per_token)     # block-level pressure
    rho_rd  = F_eff / d_eff (measured)     # read pressure ~ F_rd / d_eff

The output W is an ordinary (d_eff x F_rd) unit-column dictionary, consumed
unchanged by plant_pair, delta_star_analytic, the estimators, and the gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import estimators as est

__all__ = ["TokenParams", "build_token_dictionary", "TokenBuild"]


@dataclass
class TokenParams:
    F_ref: int = 32         # features merged into the read at the reference budget
    N: int = 32             # carrier-token budget (the compression axis)
    N_ref: int = 32         # reference (uncompressed) token budget
    d_eff: int = 32         # FIXED read-position width the monitor consumes
    k_per_token: int = 1    # features-worth each token carries (for rho_blk only)
    max_load: float = 24.0  # cap F_rd / d_eff so the frame stays tractable
    seed: int = 0

    def F_rd(self) -> int:
        """Effective read features = F_ref * N_ref / N (fewer tokens -> more)."""
        raw = int(round(self.F_ref * self.N_ref / max(self.N, 1)))
        cap = int(self.max_load * self.d_eff)
        return int(max(1, min(raw, cap)))

    def rho_blk(self) -> float:
        return float(self.F_rd() / max(self.N * self.k_per_token, 1))


@dataclass
class TokenBuild:
    W: np.ndarray           # (d_eff, F_rd) unit-column read dictionary
    F_rd: int               # effective number of read features at this N
    rho_blk: float          # block-level pressure
    d_eff: int              # fixed read width (== packing dimension)


def build_token_dictionary(p: TokenParams) -> TokenBuild:
    """Fixed-width read frame with F_rd(N) features merged into it.

    W is a generic exact-geometry frame (unit Gaussian columns) in the FIXED
    d_eff read space; the token/compression content is entirely in how many
    features F_rd(N) are packed in. This keeps the monitor coverage fraction
    fixed (steady Gamma_r) so the leverage channel drives S with the right sign.
    """
    rng = np.random.default_rng(p.seed)
    F = p.F_rd()
    W = est.normalize_columns(rng.standard_normal((p.d_eff, F)))
    return TokenBuild(W=W, F_rd=F, rho_blk=p.rho_blk(), d_eff=p.d_eff)