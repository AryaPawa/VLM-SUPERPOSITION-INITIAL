# CSC-VLM: does visual-token compression make VLM internals harder to audit?

This README covers **pipeline v2 (2026-11-07)**, on branch `p2`. The legacy
grid-search pipeline (`run.py`, the S = β/(Γ·‖a‖) metric) is kept at the bottom
for reference only. That metric is measurement-broken (see the project doc
`claude/p2-grid-diagnosis-2026-10-15.md`).

---

## What v2 changes, and why

A first-principles review of the p2 code found that our operators and metric did
not test what their names said. v2 fixes this:

| Problem found | Fix in v2 |
|---|---|
| Old `merge` kept all N0 positions (duplicates diverge after layer 1). It reduced *information*, not *carriers*. | `compression.py`: **real shortening** operators: `pool` (2-D average pooling on the known image grid), `drop_norm`, `drop_random`, `drop_query`. The old merge is kept as `merge_dup` (a fixed-N information-bottleneck control). |
| Mask-prune never verified against a real drop | Positions are preserved. `mask_norm == drop_norm` is verified to float precision on real LLaVA / Qwen2.5-VL classes, and again on your model in `preflight.py`. |
| transformers ≥5 treats position-id gaps as packed-sequence boundaries when there is no attention mask | We always pass an explicit 2-D attention mask with shortened plans. |
| Stage-1 merge leaf had the wrong perturbation space | The leaf is always the **k distinct carriers**. Gradients are checked against finite differences. |
| Γ ≈ 0.99 at r=8, so log Γ ≈ 0 and the auditability term cannot move OC; the isotropic placebo was vacuous | Report the **obfuscation tax** T = 1/Γ and the **log-alignment** A = −log cap. Use **geometry-matched placebos** (a random answer pair; a random read direction). Report nested Γ(r) curves and read-space rank-to-audit. |
| The decomposition was taken on medians (not an identity), with absolute "shares" | **Per-image** log OC = log m − log‖g‖ + log T. Paired (within-image) slopes with **cluster-bootstrap CIs**. Exact signed shares. |
| cats_vs_dogs used pos="cat" while label 1 = dog; one first-token answer id | Polarity is fixed. The margin is a **logsumexp over answer variants** (' cat', 'Cat', …), with whitespace-only lead tokens skipped. |
| X1 toy assumed its own result | **X1b**: a trained toy in which the geometry emerges. It measures coverage vs interference exactly. |
| No check that first-order OC is real | Stage 1 applies the first-order minimal evading δ in the true model on `--n-check` images per cell, and logs the flip rate and the monitor drift. |
| The model was loaded in fp16 then upcast (Qwen fp16 overflow) | Load directly in the target dtype. `auto` = bf16 on Ampere+ GPUs. |

Every slope is reported **two-sided**. beta = d log X / d log(actual kept
fraction), so beta > 0 means X **falls** under compression.

---

## Pipeline v3 (2026-11-11): real attacks in a fixed space (`attack.py`)

The cluster runs (Stage 0 on Qwen-3B/LLaVA-7B; Stage 1 on Qwen-3B/SmolVLM) showed:
- **packing (ρ_blk) is null;**
- **the first-order monitor tax T never falls** (0/32 cells);
- **the first-order OC is not trustworthy.** It is linear (Qwen reaches 39–47 % of the predicted change), measured in a keep-dependent space, uses an equality constraint, freezes the selection, and defines the attack direction by the current answer. v3 measures the costs with a real optimiser.

| quantity | definition (min ‖δ‖ in a FIXED space) | meaning |
|---|---|---|
| `OC_free` | answer becomes the WRONG label | robustness |
| `OC_thr` | wrong answer AND the rank-1 probe tripwire stays silent | cost of a hidden attack |
| `OC_eq` | wrong answer AND probe reading unchanged (±5 % of the class gap) | Stage-1-comparable |
| `T_true` | `OC_thr / OC_free` (`log OC_thr = log OC_free + log T_true` exactly) | the monitor's leverage |

- **Protocol:** the probe is a *tripwire* that fires when it reads the target class. Under a disagreement-flagging protocol the same numbers read the other way, which is why all three costs are reported.
- **Leaves:**
  - `patch` = pre-compression visual-token embeddings (the instrument check);
  - `pixel` = normalised pixels clipped to valid images.
- **Selection:**
  - `frozen` reuses the clean plan;
  - `live` rebuilds drop_norm / drop_query on the perturbed input (BPDA, so the measured steering advantage is a lower bound).
- **Optimiser:** constrained FAB (Croce & Hein 2020). Each step solves the linearised min-norm problem exactly (`min_norm_point`, active-set KKT), and the run ends with a ray bisection.
- **Every reported cost is a verified success in the true model:** an upper bound on the true minimum. On linear targets it matches the closed form within 0.3 %; on curved 2-D targets it is within 1.5 % of a brute-force minimum (tests).

**Pre-registered reading** (`scripts/analyze_attack.py`):

| check | rule |
|---|---|
| instrument (patch, frozen) | PASS iff Spearman(lin, true) ≥ 0.5 at every keep with ≥ 8 attacks AND the CI of the slope of log(true/lin) lies inside ±0.1. Otherwise OC_lin is diagnostic only and the true costs are primary. |
| robustness | slope of log OC_free (β > 0: attacks get cheaper under compression), split by attack direction (y=1 hide / y=0 induce) |
| monitor leverage | slope of log T_true; check `T<1` (should be ≈ 0: a measured T < 1 means the free attack was sub-optimal) |
| hidden attack | slope of log OC_thr = log OC_free + log T_true |
| incidental detection | P(tripwire fires \| plain attack succeeded) per keep, Wilson CI, paired k_min − k_1 |
| **H-R2** (selection steering) | log(OC_frozen/OC_live) > 0 in images whose live selection CHANGED, beyond the noise floor of images whose selection did not; R2 index = steering(thr) − steering(free) > 0 means steering specifically helps HIDDEN attacks |

Hypotheses updated after the cluster runs:
- H1/H3 (compression → packing → cheaper monitor evasion) are refuted at first order.
- **New hypotheses:**
  - **H-R2:** live selection lowers OC_thr more than OC_free.
  - **H-stale:** the frozen-probe AUROC falls on non-saturated concepts (seen in Stage 0).
  - **H-bias:** compression shifts answers toward the default label, asymmetrically by direction (seen in Stage 0).

```
python scripts/run_attack.py --self-check                     # CPU, < 1 min
sbatch scripts/run_attack.slurm                               # 4 array tasks: {patch,pixel} x {bottle,cats}
python scripts/analyze_attack.py runs_attack/*/attack_results.json
```

---

## Run order (each step decides the next)

```
0. CPU, anytime:   pytest tests/ -v                         # smoke + tiny real-model tests
                   python scripts/run_x1b_toy.py --quick     # trained toy, ~1-2 min
1. GPU, ~5 min:    python scripts/preflight.py --model qwen-3b --grad
2. GPU, 1-4 h:     python scripts/run_stage0.py --model qwen-3b --dataset hf:microsoft/cats_vs_dogs --out runs_stage0/qwen3b_cats
3. GPU, 2-8 h:     CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1.py --model qwen-3b --out runs_stage1/qwen3b_cats
4. CPU:            python scripts/analyze_stage1.py runs_stage1/qwen3b_cats/stage1_results.json --r 4
5. GPU, 3-9 h:     sbatch scripts/run_attack.slurm      # pipeline v3 (instrument check + R2 test)
```

### Kaggle (GPU T4 / P100, Internet ON)
```bash
!git clone -b p2 https://github.com/AryaPawa/VLM-SUPERPOSITION-INITIAL.git
%cd VLM-SUPERPOSITION-INITIAL/csc_vlm
!pip -q install -U "transformers>=4.49" accelerate datasets qwen-vl-utils sentencepiece pytest
!python -m pytest tests/ -q                      # ~1-3 min on CPU/GPU, no downloads
!python scripts/run_x1b_toy.py --quick
!python scripts/preflight.py --model qwen-3b --dataset hf:microsoft/cats_vs_dogs --dtype float16 --grad
!python scripts/run_stage0.py --model qwen-3b --dtype float16 --out runs_stage0/qwen3b_cats
!CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1.py --model smolvlm-500m --dtype float32 --n-images 48 --out runs_stage1/smol500m_cats
```
* A T4 has no bf16. Use fp16 for Stage 0 (forward only), and **fp32** for
  Stage 1 with ≤3B models.
* If Qwen-3B fp16 gives NaN/inf logits (preflight reports it), use SmolVLM or
  InternVL-2B on a T4, or run Qwen on the H100 in bf16.

### Cluster (H100, SLURM)
`sbatch scripts/run_stage0.slurm`, then `sbatch scripts/run_stage1.slurm`. Both
run the preflight first and skip any model that fails it. Edit `MODELS` and
`DATASETS` at the top of each script. Pipeline v3: `sbatch scripts/run_attack.slurm`.
It checkpoints after every image and resumes when resubmitted.

### Datasets / concepts
| spec | concept | note |
|---|---|---|
| `hf:microsoft/cats_vs_dogs` | dog (1) vs cat (0) | probe AUROC likely **saturates**; fine for R1 / margin |
| `coco_person` | person present | `python scripts/build_person_dataset.py --download` |
| `coco:<category>[:path]` | any COCO category | `python scripts/build_person_dataset.py --download --category bottle`; small objects give a **non-saturating** concept |

---

## Compression methods (`csc_vlm/compression.py`)
| name | carriers ↓? | info kept? | role |
|---|---|---|---|
| `pool` | yes (real shortening) | mostly | merge-family arm of H2b |
| `drop_norm` | yes | no | old prune criterion, real drop |
| `drop_random` | yes | no | **G1 / H5 null** (how many vs which) |
| `drop_query` | yes | no | query-conditioned selection (**R2**; ICML'26 sign-flip family) |
| `merge_dup` | **no** (N fixed) | input info ↓ | X3-like fixed-N information bottleneck control |
| `mask_norm` | via mask | no | verification only (must equal `drop_norm`) |
| `readout_only` | no | yes | Stage-0 read-artifact control |

---

## How to read the results → what to do next

**Preflight:** any FAIL means stop. Fix it before running Stage 0/1. A WARN on
`answer_in_top5` means the model answers with other words: edit `BEHAVIOURS` in
`stage0.py`.

**Stage 0**
* Accuracy collapses for every method → the effect is R1 (information loss).
  The matched-accuracy control must lead before any Stage-1 claim.
* `pool` holds while `drop_*` lose accuracy → the H2b contrast is live (merge
  family vs deletion).
* Probe AUROC CI is above 0.98 at keep=1 (saturated) → re-run on `coco:bottle`.
* `cov_top8` > 0.8 → coverage regime: a rank-8 monitor already spans most of the
  read. Read the Stage-1 Γ(r) curves, not one rank (Problem D).

**Stage 1** (per method). Two summaries are printed: **r = 1, the canonical
rank-1 probe** (class-mean-difference axis, midpoint threshold; the headline —
no rank or PC choice to defend) and r = r_max (robustness). Re-analyse any rank
offline with `scripts/analyze_stage1.py`. The alignment slope splits exactly as

  `A = bypass(A − A_readpath) + specific(A_readpath − A_match) + A_match`

* `A_match` (matched placebo: same monitored component as the behaviour's read
  gradient, random unmonitored part) is the **generic coverage** null with no
  ceiling/floor asymmetry. Use it, not `A_pl_out`/`A_pl_read`, for specificity.
* `bypass` = paths the last-token read cannot see (other positions, later layers).
* `full-capacity detector cost` = |b|/sqrt(cᵀΣc): an optimal read-space detector.
* `incidental` = fraction of images where the canonical probe flags the
  **plain** (monitor-unaware) minimal attack; `unc_flip/probe` = the same in the
  true model on n_check images.

| outcome (r = 1, CIs) | meaning | next |
|---|---|---|
| log T β > 0 **and** `specific` β < 0 | compression lowers the tax beyond coverage geometry | build Stage 2 (pixel PGD + gradient-free) for this method |
| log T β > 0, `specific` ≈ 0, `bypass` β < 0 | the behaviour moves into channels the last-token monitor cannot read | claim is about **where** monitors read; test multi-position/later-layer monitors |
| log T β < 0, `specific` β < 0 | coverage dominates the net; behaviour-specific erosion underneath | claim it at **fixed coverage** (rank-to-audit), not at fixed r |
| log T and full-capacity cost both ≈ 0, OC moved by m or ‖g‖ | no auditability effect; robustness / info-loss result | mentor decision: reframe as a robustness paper |
| log T moves but full-capacity cost ≈ 0 | the effect is about **which** monitor, not detectability itself | report as a monitor-choice result |
| incidental ≈ 1 and stays ≈ 1 | the plain attack is already caught: evasion needs the monitor-aware attack | headline the monitor-aware OC |
| T ≈ 1 at all keeps | a linear probe imposes no tax even uncompressed | use stronger or higher-rank monitors before any claim |
| `flip/drift` ≠ ~1.00 / ≪1 | first-order OC is not realised in the true model | Stage 2 is required before any OC number |

Also compare `merge_dup` with `pool` (information bottleneck vs carrier count),
and `drop_random` with `drop_norm` (G1).

**X1b toy:** run the full 3-seed sweep (`python scripts/run_x1b_toy.py`, ~5 min).
Regimes `carrier_bottleneck` and `subtle_behaviour`; ranks 1, 2, 4. Trust a sign
only if `[seeds n/n]` agree. The signs of `log T`, `A − A_matched` and the
full-capacity cost per regime are the map the VLM numbers are compared against.

---

## Tests
```
python -m pytest tests/ -v
```
* `tests/smoke_test.py`: legacy mock pipeline plus the X1, X1b, Stage 0 and
  Stage 1 self-checks (numpy only).
* `tests/test_pipeline_v2.py`: **real transformers classes** (tiny random LLaVA +
  Qwen2.5-VL, no downloads). It checks that every plan runs, that
  mask == drop, that the identity leaf reproduces the logits, that leaf
  gradients match finite differences, that the Gram capture matches QR, that
  Stage 0/1 run end to end, and the X1b self-check. It is skipped without torch.
* `tests/test_attack.py` (v3) checks:
  - a zero `span_delta` reproduces the compressed logits;
  - patch- and pixel-leaf gradients match finite differences;
  - a live selection follows the perturbed input;
  - the exact min-norm solver and closed-form agreement;
  - brute-force optimality on curved targets;
  - the full grid runs end to end with verified successes, and resume works.

## Layout
```
csc_vlm/  compression.py vlm.py stats.py stage0.py evasion.py toy.py toy_trained.py data.py
          attack.py attack_analysis.py                      (pipeline v3)
          (legacy) backends.py geometry.py solver.py runner.py
scripts/  preflight.py run_stage0.py run_stage1.py analyze_stage1.py run_x1_toy.py run_x1b_toy.py
          run_attack.py analyze_attack.py run_attack.slurm  (pipeline v3)
          run_stage0.slurm run_stage1.slurm build_person_dataset.py prepare_data.py
          (legacy) run_matrix.sh run_llava.slurm run_qwen.slurm run_csc.slurm
tests/    smoke_test.py test_pipeline_v2.py test_attack.py tiny_models.py
```

---

## Legacy grid pipeline (reference only; known-flawed metric)
`python run.py --model mock` / `bash scripts/run_matrix.sh` run the p2
grid-search over S = β/(Γ_r·‖a‖). Do not use its verdicts as evidence: the
behaviour sits inside the monitor, and Γ comes from a 24-sample covariance in
d = 4096. Kept so the RUN 1–8 diagnosis stays reproducible.
