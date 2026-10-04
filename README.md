# Compress-under-a-Safety-Constraint (`csc`) — Phase A

Codebase for the safety-constrained visual-token-compression project. This is the **Phase A** slice: the exact-geometry toy substrate that proves the core relation — compressing a VLM's visual tokens makes it easier for an attacker to fool the model's internal safety monitor.

## The Core Claim

When visual tokens are compressed, the model is forced to pack the same amount of visual information into fewer representational dimensions. This changes the geometry of the internal activations in a way that reduces the minimum perturbation an attacker needs to blind a safety monitor. The toy substrate in `src/csc/` proves this holds exactly when geometry is fully controlled, before testing on real VLMs (see `csc_vlm/`).

## Layout

```
src/csc/
  estimators.py   Appendix-B geometry defs (coherence, Welch, d_eff, ρ_rd, Γ_r)
  tokens.py       Token-aggregation substrate: N tokens → pooled read
  model.py        ToyConfig / ToyModel (geometry = random | trained | token)
  attack.py       Closed-form + PGD attack; gradient-masking probe
  sweep.py        Grid runner + token_sweep
  gates.py        G0 gate checks (COLLAPSE, MEDIATE, WELCH, NOSUPER)
  plotting.py     Figure 0, Figure T (token), Figure M (masking)

scripts/
  run_phase_a.py            End-to-end toy run (random substrate)
  run_token_substrate.py    Token N-sweep + gradient-masking probe

tests/
  test_phase_a.py           Invariant tests (Welch bound, Prop-1, effect sign)

csc_vlm/                    Phase 3 — real VLM harness (see csc_vlm/README.md)
results/                    Phase A run artifacts (CSVs, figures)
```

## Run (Phase A toy)

```bash
pip install -r requirements.txt
python scripts/run_phase_a.py --quick        # fast smoke run + gate
python scripts/run_phase_a.py --pgd          # + empirical cross-check
python tests/test_phase_a.py                 # invariants
```

Outputs land in `results/`: `phase_a_pairs.csv`, `phase_a_cells.csv`, `figure0.png`.

## Phase A Status

The G0 gate is validated on the **random/constructed** geometry (exact-geometry frame with controllable coherence — sufficient for the Welch/Prop-1 positive control).

Key Phase A finding: **the mediator is two-dimensional** — alignment `A` alone does not collapse the ρ→S path (partial ≈ 0.81); adding behaviour leverage `‖a‖` collapses it (partial ≈ 0.01). The VLM-scale H3 mediator must co-measure leverage, not only Γ_r.

> For the real-VLM empirical pipeline, see [`csc_vlm/README.md`](csc_vlm/README.md).
