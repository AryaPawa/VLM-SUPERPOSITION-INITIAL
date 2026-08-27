# Phase 0 — Calibration (frozen assumptions & what changed vs v1)

This is the scoping contract the rest of the project builds on. It records (a)
the five operating assumptions v2 runs under, (b) the v1→v2 delta the professor's
directions forced, and (c) the adversarial-review corrections that are **binding**
(they override the original proposal wherever they conflict).

## Operating assumptions (frozen)
1. **ADMM is the primary solver.** We solve `min_θ J_comp(θ) s.t. S(θ) ≥ δ, Acc(θ) ≥ Acc_min`
   by variable splitting + augmented Lagrangian (θ/z/u loop). Primal–dual with a
   PID-Lagrangian controller and plain dual ascent become **baselines**, not the method.
2. **Prove the relation first.** The chain `N → ρ_blk → ρ_rd → Γ_r → OC`, the mechanism
   `dS/dρ_rd < 0`, and the Welch feasibility ceiling are established (theory + cheap
   experiment) in a **gating Phase A** before any cluster spend. Fail the gate ⇒ pivot.
3. **CAID (arXiv:2505.19387) supplies the guarantee template** — the √ν parametrization-gap
   transport for the LoRA-parameterized safety subproblem (Phase 3).
4. **"Holds forever" = a persistence property (T-PERSIST)** at the ADMM fixed point, with
   tests under held-out attacks, secret monitor subspaces, and mild shift.
5. **v1 notation reused verbatim** (`f = g∘π∘E`, `C_κ`, `ρ_blk/ρ_rd`, `Γ_r`, `OC`,
   `S/S_OC via AOTC`, surrogate `S̃ = A`). Compute/storage figures (≈3,850 A100-h, ≈1.1 TB)
   are internal v1 projections, flagged for re-estimation under the projection-heavy z-step.

## v1 → v2 delta
| v1 | professor's directive | v2 |
|---|---|---|
| primal–dual + PID-Lagrangian is the solver | "formulate with ADMM; smoother optimization" | ADMM primary; PID/dual are Phase-4 baselines |
| L1–L4 stated; primal–dual convergence (L3) central | "guarantees on the balance between competing objectives" | **T-BALANCE** is the centerpiece; L1/L4 are Phase-A spine; L2/L3 fold into T-ADMM-1 |
| gates escalate straight to scale | "prove the relation first, then continue" | new gating **Phase A**; G1 "promise" gate moves downstream |
| "never sacrifice safety" informal | "constraint satisfaction should hold forever — formalize" | **T-PERSIST** + durability suite |
| training assumed tractable | "training far more tedious; expect trial-and-error" | explicit tuning ladder (warm-start, ρ-schedule, multi-timescale, damping) |
| CAID absent | "build on CAID; transport the parametrization gap" | CAID in literature + Phase 3 √ν transport |

## Binding review corrections (override v1 on conflict)
- **Split the pressure.** `ρ_blk = F_eff^blk / (N·d_eff^blk)` (block) vs `ρ_rd = F_eff^rd / d_eff^rd`
  (read position). The `N → ρ_rd` arrow is an **empirical** aggregation claim, not a Welch identity.
- **F_eff is endogenous.** Measure the exponent **α** in `F_eff ∝ N^α`; **α<1 ⇒ superposition**,
  **α≈1 ⇒ mere information loss**. Delete the invariance assumption.
- **Rank-r Γ.** Use `Γ_r = ‖P_{U⊥} Jᵀw_b‖/‖Jᵀw_b‖` (projection in activation space) and report on the
  **log-alignment scale A = −log(1−Γ_r²)**; the rank-1 Γ sits at 0.9998 and mediates nothing.
- **Redefine S_OC.** Primary = **per-example minimal evading perturbation ‖δ*‖** + **AOTC** (area over the
  ASR/AUROC curve); *not* the AUROC-threshold definition.
- **Welch at the read position.** Apply the coherence floor in `R^{d_eff}` at the read position (where the
  monitor reads), not to a flat `R^{N·d}` block.
- **Gradient-masking controls mandatory.** BPDA backward-pass, soft-top-k, gradient-free min-over-attacks,
  per-cell attack certificates — measure true erosion, not obfuscated gradients.
- **Circularity guards.** Confirmatory superposition measure = SAE-free geometric family (b); probe-capacity
  concept set disjoint from monitor behaviours; report partial-corr controlling for clean probe accuracy.

## Gate map
- **G0 (this stage, toy):** does OC collapse onto ρ in the exact-geometry substrate? (positive control)
- **G-A1..G-A4 (VLM):** α<1; ρ_rd/Γ_r trends; dS/dρ_rd<0 partial-corr; empirical δ_max(N) ~ Welch.
- **G1 (downstream):** δ-sweep frontier on two lineages → professor's cluster handoff.
