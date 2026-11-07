#!/bin/bash
# run_stage1.slurm  --  pipeline v2: gradient preflight + Stage 1 on ONE GPU.
# Submit from the csc_vlm/ project dir:   sbatch scripts/run_stage1.slurm
#
# Stage 1 takes gradients:
#   * one GPU only (the handle loads with a single-device map);
#   * float32 by default (bfloat16 is fine on H100/A100; never float16);
#   * memory: fp32 3B ~12 GB + activations; 7B fp32 ~28 GB (H100 ok).
#SBATCH --job-name=csc_stage1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --requeue

set -uo pipefail
mkdir -p logs runs_stage1 runs_preflight
# module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csc_vlm
export CUDA_VISIBLE_DEVICES=0

MODELS=( "qwen-3b" "smolvlm-500m" )        # H100: add "llava" "qwen"
DATASETS=( "hf:microsoft/cats_vs_dogs" "coco:bottle:data/concept_bottle" )
DTYPE=float32
NIMG=64
RMAX=8

for model in "${MODELS[@]}"; do
  tag0=$(echo "${model}" | tr '/:' '__')
  echo "==== preflight (--grad): $model ===="
  if ! srun python scripts/preflight.py --model "$model" --dataset "hf:microsoft/cats_vs_dogs" \
        --dtype "$DTYPE" --grad --out "runs_preflight/${tag0}_grad"; then
    echo "[FAIL] gradient preflight failed for $model -- skipping"
    continue
  fi
  for ds in "${DATASETS[@]}"; do
    tag=$(echo "${model}_${ds}" | tr '/:' '__')
    echo "==== Stage 1: model=$model dataset=$ds ===="
    srun python scripts/run_stage1.py --model "$model" --dataset "$ds" --dtype "$DTYPE" \
        --n-images "$NIMG" --r-max "$RMAX" --out "runs_stage1/${tag}" \
      || echo "[warn] stage1 $model $ds exited non-zero; continuing"
  done
done
echo "Stage 1 done. Results: runs_stage1/<tag>/{stage1_results.json,stage1.png}"
