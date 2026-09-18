# CSC-VLM — Compression–Auditability Frontier on Real VLMs

Standalone, checkpointed codebase that measures the **compression–safety trade-off**
inside a vision-language model. It compresses a model's visual tokens by a
controllable amount, and for each safety floor δ it finds the *most* compression
allowed while a monitor stays hard to evade — tracing the Pareto frontier and the
shadow price of safety.

It is fully self-contained: only `numpy/scipy/pandas/matplotlib` for the core, plus
`torch/transformers/datasets` for the real-model runs. Nothing else from the wider
research project is required.

The pipeline runs **end to end**: it auto-downloads a dataset, loads the model on
GPU, fits a monitor once, sweeps δ, **checkpoints after every step**, and writes a
CSV + figure + machine-readable verdict. If a job is pre-empted, `--resume` picks up
where it left off.

---

## 1. What it produces

For each run directory (`--out`):

| file | contents |
|---|---|
| `frontier.csv` | one row per δ: `keep, compression, S, λ, Γ_r, ‖a‖, d_eff, …` |
| `figure_frontier.png` | 4 panels: frontier, shadow price, H3 mediators, read-dim |
| `verdict.json` | the three gate results + summary statistics |
| `checkpoint.json`, `monitor.npz` | resume state (safe to keep) |

The scientific verdict has three gates: **frontier_monotone** (more safety ⇒ less
compression), **mechanism_H3** (leverage rises with compression — the causal story),
and **feasible** (each point meets its floor within measurement noise).

---

## 2. Setup on the cluster (blank conda env)

```bash
# a) create the env (CPU core only)
conda env create -f environment.yml
conda activate csc_vlm

# b) install PyTorch matching the cluster's CUDA (example: CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# c) install the GPU model stack
pip install -r requirements-gpu.txt
```

Verify the core works with **zero GPU and zero download** (a few seconds):

```bash
python run.py --model mock
# expect: VERDICT: PASSED  (frontier_monotone / mechanism_H3 / feasible all PASS)
```

---

## 3. Validate the real model cheaply (do this before the big run)

One command loads a small model and checks the whole chain — load, activation hook,
**visual-token merge**, monitor fit — on throwaway synthetic images:

```bash
python run.py --model qwen --model-id Qwen/Qwen2.5-VL-3B-Instruct --load-4bit --self-check
```

Look for `SELF-CHECK: PASS` and `merge_hook_fired=True`. If the merge hook does **not**
fire, the message tells you exactly which function to check before spending GPU hours.
(The geometry numbers here are meaningless — synthetic images are near-identical — the
check is purely mechanical.)

---

## 4. Full frontier runs

Pre-download the dataset on the login node (so compute nodes can run offline):

```bash
python scripts/prepare_data.py --dataset hf:microsoft/cats_vs_dogs
```

Then submit the jobs (SLURM templates provided — edit the `module load` / `conda`
lines for your cluster):

```bash
sbatch scripts/run_qwen.slurm
sbatch scripts/run_llava.slurm
```

…or run directly on a GPU node:

```bash
python run.py --model qwen  --dataset hf:microsoft/cats_vs_dogs --out runs/qwen
python run.py --model llava --dataset hf:microsoft/cats_vs_dogs --out runs/llava
```

Each run is roughly **1–3 h on one H100** (7B, `--n-pairs 24 --steps 60 --n-deltas 6`).

### Resuming after a pre-emption

`--resume` is in the SLURM scripts and is safe on the first run too. It reloads the
monitor and calibration and skips δ points already in `checkpoint.json`, so a requeue
costs at most one δ of work:

```bash
python run.py --model qwen --dataset hf:microsoft/cats_vs_dogs --out runs/qwen --resume
```

---

## 5. Choosing the concept / dataset

`--dataset` accepts three forms:

- `synthetic` — built-in, no download (plumbing only).
- `hf:<name>` — any HuggingFace image-classification dataset, auto-downloaded and
  binarised. Default `hf:microsoft/cats_vs_dogs` is a clean **benign** concept
  ("is there a dog") — the right first pass. Options: `hf:<name>:label=<col>:pos=<ids>`.
- `folder:/path` — your own images as `path/<classA>/*.jpg` and `path/<classB>/*.jpg`
  (first folder = label 0). This is how you plug in the **safety-facing** concept from
  the proposal — e.g. `data/concept/no_person/` and `data/concept/person/` built from
  COCO/POPE — once the benign pass is confirmed:

```bash
python run.py --model qwen --dataset folder:data/concept --out runs/qwen_person
```

### Safety-facing concept: COCO "person" (`coco_person`)

The safety-relevant concept — *is there a person?* — is a first-class dataset. Build
it once from COCO val2017 (one command; downloads ~1.2 GB and sorts into an
ImageFolder), then run:

```bash
# build data/concept/{person,no_person} (auto-downloads COCO if missing)
python scripts/build_person_dataset.py --download --output data/concept

# on a cluster, do the build on the login node via prepare_data:
python scripts/prepare_data.py --dataset coco_person

# then run the frontier on it (person = label 1, no_person = label 0)
python run.py --model qwen  --dataset coco_person --out runs/qwen_person
python run.py --model llava --dataset coco_person --out runs/llava_person
```

`coco_person` is equivalent to `folder:data/concept`; use `coco_person:/your/path` to
point at a build elsewhere. Extend to other COCO categories (knife, car, …) with
`build_person_dataset.py` as a template. cats-vs-dogs remains the default benign
dataset — nothing about it changes.

A few hundred images per class is plenty (`--per-class`).

---

## 6. Key knobs

| flag | default | meaning |
|---|---|---|
| `--model` | `mock` | `mock` (CPU), `llava`, `qwen` |
| `--model-id` | per-lineage | HF id (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) |
| `--dataset` | `hf:microsoft/cats_vs_dogs` | see §5 |
| `--load-4bit` | off | 4-bit quant (fits 7B on a small GPU; H100 can skip it) |
| `--read-layer-frac` | 0.70 | decoder depth for the monitor read position |
| `--n-pairs` | 24 | images per safety evaluation (↑ = less noise, more compute) |
| `--n-deltas` | 6 | number of frontier points |
| `--steps` | 60 | PID iterations per δ |
| `--per-class` | 200 | images per class to load |
| `--resume` | off | continue from `--out` checkpoint |

If `feasible` fails with a small `median|S-δ|`, that is sampling noise near the
boundary, not a mechanism failure — raise `--n-pairs` to 32–48. The load-bearing
scientific results are `frontier_monotone` and `mechanism_H3`.

---

## 7. Troubleshooting

- **`AutoModelForImageTextToText` not found** → `transformers` too old; `pip install -U
  "transformers>=4.49"` (Qwen2.5-VL needs it).
- **bitsandbytes / 4-bit errors** → drop `--load-4bit`; an H100 runs 7B in fp16 fine.
- **HF rate-limit / gated warnings** → `export HF_TOKEN=...` (a free token).
- **No internet on compute nodes** → run `scripts/prepare_data.py` on the login node
  first (it warms the cache), then submit the job.
- **`merge_hook_fired=False` in `--self-check`** → the visual-token span for that model
  needs adjusting; see `csc_vlm/backends.py::HFVLMBackend._pre_hook` / `_visual_span`.

---

## 8. Layout

```
run.py                     CLI entry point
csc_vlm/
  geometry.py              monitor fit + obfuscation cost S = β/(Γ_r·‖a‖)
  backends.py              mock + HF (LLaVA/Qwen) backends, visual-token merge
  solver.py                safety oracle, calibration, PID-Lagrangian, evaluation
  data.py                  dataset resolver (hf / folder / synthetic)
  runner.py                checkpointed orchestration + figure + verdict
scripts/
  prepare_data.py          warm the dataset cache on the login node
  build_person_dataset.py  build COCO person/no_person ImageFolder (safety concept)
  run_qwen.slurm           SLURM job (one H100, auto-resume)
  run_llava.slurm          SLURM job (one H100, auto-resume)
environment.yml            blank conda env (CPU core)
requirements.txt           CPU core deps
requirements-gpu.txt       transformers / datasets / bitsandbytes (torch separate)
```
