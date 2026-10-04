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

OUT=runs/llava_v4

srun python run.py \
    --model llava \
    --dataset hf:microsoft/cats_vs_dogs \
    --out "$OUT" \
    --resume \
    --n-pairs 40 --n-deltas 6 --steps 40 --per-class 200
