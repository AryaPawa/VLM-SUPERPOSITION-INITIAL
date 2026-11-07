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

## Run order (each step decides the next)

```
0. CPU, anytime:   pytest tests/ -v                         # smoke + tiny real-model tests
                   python scripts/run_x1b_toy.py --quick     # trained toy, ~1-2 min
1. GPU, ~5 min:    python scripts/preflight.py --model qwen-3b --grad
2. GPU, 1-4 h:     python scripts/run_stage0.py --model qwen-3b --dataset hf:microsoft/cats_vs_dogs --out runs_stage0/qwen3b_cats
3. GPU, 2-8 h:     CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1.py --model qwen-3b --out runs_stage1/qwen3b_cats
4. CPU:            python scripts/analyze_stage1.py runs_stage1/qwen3b_cats/stage1_results.json --r 4
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
`DATASETS` at the top of each script.

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

**Stage 1** (per method). The headline is `log T` and `A_minus_pl_out`:

| outcome | meaning | next |
|---|---|---|
| log T beta > 0 (CI) **and** A−placebo beta < 0 (CI) | compression lowers the monitor's tax beyond what geometry predicts | build Stage 2 (pixel PGD + gradient-free) for this method |
| log T beta < 0 but A−placebo beta < 0 | coverage dominates the net; there is a behaviour-specific erosion underneath | claim it at **fixed coverage** (rank-to-audit), not at fixed r |
| log T ≈ 0 and A−placebo ≈ 0, OC moved by m or ‖g‖ | no auditability effect; this is a robustness / info-loss result | mentor decision: reframe as a robustness paper |
| T ≈ 1 at all keeps | a rank-r linear probe imposes no tax even uncompressed | use stronger or higher-rank monitors before any claim |
| `flip/drift` ≠ ~1.00 / ≪1 | first-order OC is not realised in the true model | Stage 2 is required before any OC number |

Also compare `merge_dup` with `pool` (information bottleneck vs carrier count),
and `drop_random` with `drop_norm` (G1).

**X1b toy:** the sign of `log T` and of `A − A_placebo` in each regime is the
regime map that the VLM numbers should be compared against.

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

## Layout
```
csc_vlm/  compression.py vlm.py stats.py stage0.py evasion.py toy.py toy_trained.py data.py
          (legacy) backends.py geometry.py solver.py runner.py
scripts/  preflight.py run_stage0.py run_stage1.py analyze_stage1.py run_x1_toy.py run_x1b_toy.py
          run_stage0.slurm run_stage1.slurm build_person_dataset.py prepare_data.py
          (legacy) run_matrix.sh run_llava.slurm run_qwen.slurm run_csc.slurm
tests/    smoke_test.py test_pipeline_v2.py tiny_models.py
```

---

## Legacy grid pipeline (reference only; known-flawed metric)
`python run.py --model mock` / `bash scripts/run_matrix.sh` run the p2
grid-search over S = β/(Γ_r·‖a‖). Do not use its verdicts as evidence: the
behaviour sits inside the monitor, and Γ comes from a 24-sample covariance in
d = 4096. Kept so the RUN 1–8 diagnosis stays reproducible.
