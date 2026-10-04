#!/bin/bash
#SBATCH --job-name=csc_qwen
#SBATCH --gres=gpu:1              # one H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/csc_qwen_%j.out
#SBATCH --requeue                 # allow requeue after pre-emption (we resume)

set -euo pipefail
mkdir -p logs

# --- environment (edit module/conda lines for your cluster) ---
# module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csc_vlm

OUT=runs/qwen_v4

# --resume is safe on first run too (nothing to resume) and after a requeue.
srun python run.py \
    --model qwen \
    --dataset hf:microsoft/cats_vs_dogs \
    --out "$OUT" \
    --resume \
    --n-pairs 40 --n-deltas 6 --steps 40 --per-class 200
