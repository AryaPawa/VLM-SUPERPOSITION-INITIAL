#!/bin/bash
#SBATCH --job-name=csc_llava
#SBATCH --gres=gpu:1              # one H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/csc_llava_%j.out
#SBATCH --requeue

set -euo pipefail
mkdir -p logs

# module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csc_vlm

OUT=runs/llava

srun python run.py \
    --model llava \
    --dataset hf:microsoft/cats_vs_dogs \
    --out "$OUT" \
    --resume \
    --n-pairs 24 --n-deltas 6 --steps 60 --per-class 200
