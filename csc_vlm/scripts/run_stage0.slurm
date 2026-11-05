#!/bin/bash
# run_stage0.slurm  --  Stage 0 (forward-only VLM sanity) on a SLURM GPU cluster.
# Submit from the csc_vlm/ project dir:   sbatch scripts/run_stage0.slurm
# Edit MODELS / DATASETS / the conda env name below for your cluster.
#SBATCH --job-name=csc_stage0
#SBATCH --gres=gpu:1                 # one GPU; 7B fp16 fits in ~16 GB (add --load-4bit if not)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage0_%j.out
#SBATCH --requeue

set -euo pipefail
mkdir -p logs runs_stage0

# --- environment (edit to match your cluster) ---
# module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csc_vlm

# --- build the person dataset once (safe to re-run; skips if cached) ---
python scripts/build_person_dataset.py --download --output data/concept || \
    echo "[warn] person dataset build failed; coco_person runs will be skipped"

MODELS=( "llava" "qwen-3b" )                               # add: llava-13b qwen internvl-2b smolvlm ...
DATASETS=( "hf:microsoft/cats_vs_dogs" "coco_person" )     # drop coco_person if not built

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    tag=$(echo "${model}_${ds}" | tr '/:' '__')
    echo "==== Stage 0: model=$model dataset=$ds ===="
    srun python scripts/run_stage0.py \
        --model "$model" --dataset "$ds" \
        --n-eval 256 --n-monitor 144 \
        --out "runs_stage0/${tag}" \
      || echo "[warn] stage0 $model $ds exited non-zero; continuing"
  done
done

echo "Stage 0 done. Results in runs_stage0/<tag>/{stage0_results.json,stage0.png}"