# CSC-VLM — Compression–Safety Frontier on Real VLMs

Standalone harness that measures the **compression–safety trade-off** inside a vision-language model. It compresses a model's visual tokens by a controllable amount (using either **token merging** or **token pruning**) and maps the full Pareto curve of safety vs. compression via a direct **Grid Search** — no optimizer, no fragile convergence.

---

## What it produces

For each run directory (`--out`):

| File | Contents |
|---|---|
| `frontier.csv` | One row per grid point: `keep, compression, S, S_geom, leverage, gamma_r, d_eff` |
| `figure_frontier.png` | 4-panel figure: frontier curve, S vs keep, H3 mediators, read-dim |
| `verdict.json` | Gate results (`frontier_monotone`, `feasible`) + inference stats (`mechanism_H3`) |
| `checkpoint.json`, `monitor.npz` | Resume state (safe to keep between runs) |

The scientific verdict has two hard gates: **frontier_monotone** (more compression => lower S) and **feasible** (always True for grid search — we measure directly). The **mechanism_H3** stat (does leverage rise with compression?) is logged for analysis but is **not a gate**.

---

## Experimental matrix

The harness supports the 2×2×2 matrix:

| Dimension | Options |
|---|---|
| **Model** | `llava` (LLaVA-1.5-7B), `qwen` (Qwen2.5-VL-7B) |
| **Dataset** | `hf:microsoft/cats_vs_dogs` (benign), `coco_person` (safety-relevant) |
| **Compression method** | `merge` (token averaging), `prune` (activation-magnitude pruning) |

---

## Setup

```bash
# CPU core only (no GPU needed for mock runs):
pip install -r requirements.txt

# GPU stack (for real LLaVA / Qwen runs):
conda env create -f environment.yml
conda activate csc_vlm
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

---

## Quick start

```bash
# Sanity check on CPU (no model, no download, ~5 seconds):
python run.py --model mock

# Plumbing self-check on real model (validates load + hook):
python run.py --model qwen --model-id Qwen/Qwen2.5-VL-3B-Instruct --load-4bit --self-check

# Smoke test (CPU, no GPU, verifies the full pipeline end-to-end):
python -m pytest tests/smoke_test.py -v
```

---

## Full 2×2×2 runs

### ⚠️ Critical Step for SLURM Clusters
Compute nodes are often air-gapped (no internet). If you submit a job with `HF_HUB_OFFLINE=1`, it will instantly crash if the models are not already downloaded. **Before submitting the job**, run these commands on the **login node** to pre-cache the heavy model weights:

```bash
# Install the HF CLI if you don't have it
pip install -U "huggingface_hub[cli]"

# Cache the weights directly to your HF_HOME
huggingface-cli download llava-hf/llava-1.5-7b-hf
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

### Running the Matrix

The easiest way to execute the full 8-run experimental matrix (both models, both datasets, both methods) is to use the provided all-in-one bash script. It will automatically download/cache the datasets and loop through all combinations sequentially:

```bash
bash scripts/run_matrix.sh
```

If a run gets pre-empted, simply run the script again. It uses `--resume` under the hood and will instantly skip all models and grid points that have already completed.

### Resuming after pre-emption
`--resume` is safe on first run too. It reloads the monitor and skips grid points already in `checkpoint.json`:
```bash
python run.py --model llava --dataset hf:microsoft/cats_vs_dogs --compression-method merge --out runs/llava_merge --resume
```

---

## Compression methods

### Token Merging (`--compression-method merge`)
Groups contiguous visual tokens into `k = round(keep_ratio × T)` buckets and replaces each group with the group mean. Sequence length is unchanged. Implemented in `csc_vlm/backends.py::tome_merge_torch`.

### Token Pruning (`--compression-method prune`)
**Activation-Magnitude Pruning via Attention-Mask Zeroing.** Ranks all visual tokens by their embedding L2 norm. Keeps the top-k highest-energy tokens; zeros the `attention_mask` for the rest. The model ignores pruned positions but sequence shape is unchanged — no Hugging Face shape crashes. Implemented in `csc_vlm/backends.py::tome_prune_mask`.

---

## Key flags

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `mock` | `mock` (CPU), `llava`, `qwen` |
| `--model-id` | per-lineage | HF model id |
| `--dataset` | `hf:microsoft/cats_vs_dogs` | see above |
| `--compression-method` | `merge` | `merge` or `prune` |
| `--load-4bit` | off | 4-bit quant (fits 7B on small GPU) |
| `--read-layer-frac` | 0.70 | decoder depth for monitor read |
| `--n-pairs` | 24 | images per grid-point evaluation |
| `--per-class` | 200 | images per class to load |
| `--resume` | off | continue from `--out` checkpoint |

---

## Troubleshooting

- **`AutoModelForImageTextToText` not found** → `pip install -U "transformers>=4.49"`
- **bitsandbytes / 4-bit errors** → drop `--load-4bit`; H100 runs 7B in fp16 fine.
- **HF rate-limit / gated warnings** → `export HF_TOKEN=...`
- **No internet on compute nodes** → run `scripts/prepare_data.py` on login node first.
- **`hook_fired=False` in `--self-check`** → check `backends.py::HFVLMBackend._pre_hook` / `_visual_span` for the specific model.

---

## Layout

```
run.py                       CLI entry point
csc_vlm/
  geometry.py                Monitor fit + obfuscation cost S = β/(Γ_r·‖a‖)
  backends.py                Mock + HF (LLaVA/Qwen) backends; merge & prune hooks
  solver.py                  Grid search + evaluate_frontier (H3 is inference-only)
  data.py                    Dataset resolver (hf / folder / coco_person / synthetic)
  runner.py                  Checkpointed orchestration + figure + verdict
tests/
  smoke_test.py              Fast CPU smoke test (mock backend, both methods)
scripts/
  prepare_data.py            Warm dataset cache on login node
  build_person_dataset.py    Build COCO person/no_person ImageFolder
  run_qwen.slurm             SLURM job template
  run_llava.slurm            SLURM job template
environment.yml              Blank conda env (CPU core)
requirements.txt             CPU core deps
requirements-gpu.txt         torch / transformers / bitsandbytes
```
