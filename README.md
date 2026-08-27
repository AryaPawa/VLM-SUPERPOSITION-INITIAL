# Compress-under-a-Safety-Constraint (`csc`)

Codebase for the safety-constrained visual-token-compression project
(ICML/ICLR/CVPR 2027 target). This is the **Phase-A** slice: the exact-geometry
toy substrate (workstream **R3**) that proves the core relation —

> compressing → superposition (ρ↑) → evasion gain (Γ_r↑) → the minimal
> perturbation that fools a safety monitor (S = ‖δ*‖↓), with a hard Welch
> feasibility ceiling —

**before** any cluster-scale work. If the chain does not appear here, where every
quantity is exact, it will not appear at 13B. See `CALIBRATION.md` for the Phase-0
scoping contract (frozen assumptions, v1→v2 delta, binding review corrections).

## The mechanism, exactly
Read space `R^{d_eff}`, feature dictionary `W ∈ R^{d_eff×F}` (unit columns).
Behaviour direction `w_b = W[:,b]`; rank-r monitor subspace `U`. The attacker
perturbs input feature activations `z` (`h = W z`). With `a = Wᵀw_b` (behaviour
leverage) and `M = P_U W`, the hard-evasion minimal perturbation is closed-form:

    S = ‖δ*‖ = β / ‖P_null(M) a‖ = β / (Γ_r · ‖a‖),   Γ_r = ‖P_null(M)a‖/‖a‖

This is Proposition 1 exactly. Superposition raises `‖a‖` (feature b correlates
with more features) while a rank-r monitor removes at most r dimensions of
leverage ⇒ S falls as pressure rises. The no-superposition regime (W orthonormal,
monitor on w_b) gives S → ∞ ("zero successful attacks").

## Layout
```
src/csc/estimators.py   Appendix-B measurement defs (coherence, Welch, d_eff, ρ_rd, Γ_r, A)
src/csc/tokens.py       token-aggregation substrate: N tokens → pooled read (N→ρ_rd bridge)
src/csc/model.py        ToyConfig / ToyModel  (geometry = random | trained | token)
src/csc/attack.py       S_OC = ‖δ*‖ (closed form + PGD) + gradient-masking probe (soft-top-k/BPDA)
src/csc/sweep.py        grid runner + token_sweep (vary N at fixed d_eff)
src/csc/gates.py        G0 gate + G0-TOKEN (N→ρ_rd→S) + masking-detection check
src/csc/plotting.py     Figure 0, Figure T (token), Figure M (masking)
scripts/run_phase_a.py       end-to-end (random substrate): sweep → gate → figure
scripts/run_token_substrate.py  token N-sweep + gradient-masking probe
tests/test_phase_a.py   invariants (Welch bound, Prop-1 identity, effect sign)
```

## Run
```bash
pip install -r requirements.txt
python scripts/run_phase_a.py --quick          # fast smoke run + gate
python scripts/run_phase_a.py --pgd            # + empirical cross-check
python scripts/run_phase_a.py --trained        # + trained-Elhage geometry (EXPERIMENTAL)
python tests/test_phase_a.py                   # invariants
```
Outputs land in `results/`: `phase_a_pairs.csv`, `phase_a_cells.csv`, `figure0.png`.

### Token substrate + gradient-masking probe (hardened Phase A)
```bash
python scripts/run_token_substrate.py --quick   # fast
python scripts/run_token_substrate.py           # full N-sweep + masking probe
```
Outputs: `token_sweep.csv`, `figure_token.png`, `masking_probe.csv`, `figure_masking.png`.

- **Token substrate (`geometry="token"`).** N carrier tokens each contribute
  `k_per_token` read directions; pooling gives a read subspace of rank
  `r_pool = min(N·k_per_token, d_eff)`. The F features live in that subspace, so
  fewer tokens ⇒ smaller `r_pool` ⇒ the same features are forced into fewer
  effective read dimensions ⇒ `ρ_rd` up, leverage up, S down. This makes the
  **N→ρ_rd arrow empirical** (measured, not assumed) — the honest bridge to the
  VLM harness. `G0-TOKEN` checks `Spearman(N, ρ_rd) ≤ −0.8` and
  `Spearman(ρ_rd, S) ≤ −0.6` with the pooled Welch floor respected.
  *Use `k_per_token=1` when sweeping so `r_pool = min(N, d_eff)` varies across the
  whole N range instead of saturating at `d_eff`.*
- **Gradient-masking probe (R2).** A soft-top-k read gate is a defense that hides
  gradients. The naive (honest-gradient) attacker stalls at the budget cap
  ("looks safe"); BPDA (straight-through backward) and a gradient-free search
  push through and recover the true, small perturbation. `evaluate_masking`
  reports the median masking ratio `S_naive / min(S_bpda, S_gradfree)`; a value
  ≥ 2 means the machinery correctly detects erosion hidden behind an obfuscating
  defense. (The gradient-free search is the reliable unmasker; BPDA's PGD
  occasionally caps under the moving threshold — the ratio uses the min, so it is
  robust to that.)

## G0 gate (toy positive control → foreshadows VLM G-A1..G-A4)
- **COLLAPSE** S falls tightly with ρ_rd  → G-A2
- **MEDIATE**  geometry (A **and** leverage) mediates ρ_rd→S  → G-A3
- **WELCH**    coherence respects the floor and rises with load  → G-A4 / L1
- **NOSUPER**  the no-superposition regime is safe  → Stevinson/Gorton control

> **Status note.** The validated G0 result is on the **random/constructed**
> geometry (an exact-geometry frame with controllable coherence — sufficient for
> the Welch/Prop-1 positive control). The `--trained` Elhage mode has been
> **repaired** (near-uniform importance so all features are represented, cosine
> LR decay, longer default training, and dead-column re-seed so unused features
> can't spike coherence to 1) but is still **experimental**: when you run it,
> confirm the reported coherence lands **below 1 and above the Welch floor**
> before relying on it. Proper onset (α) demonstration for G-A1 is still a
> next-iteration item.

Key Phase-A finding: **the mediator is two-dimensional** — log-alignment A alone
does *not* collapse the ρ→S path (partial ≈ 0.81); adding behaviour leverage
`log‖a‖` collapses it (partial ≈ 0.01). This refines the plan's mediator spec:
the VLM-scale H3 mediator must co-measure leverage, not only Γ_r.
