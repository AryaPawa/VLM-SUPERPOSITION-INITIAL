#!/bin/bash
# run_stage1.slurm  --  Stage 1 (per-example evasion metric) on a SLURM GPU cluster.
# Submit from the csc_vlm/ project dir:   sbatch scripts/run_stage1.slurm
#
# Stage 1 computes gradients, so:
#   * use ONE GPU (gres=gpu:1) — the backend shards with device_map='auto', and the
#     fp32 cast + autograd want a single device;
#   * use --dtype float32 (fp16 underflows; bf16 only on Ampere+). On an 80 GB H100,
#     float32 fits 7B (~28 GB) and 13B (~52 GB). On <=16 GB GPUs, stick to <=3B / SLM.
#SBATCH --job-name=csc_stage1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/stage1_%j.out
#SBATCH --requeue

set -euo pipefail
mkdir -p logs runs_stage1

# module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csc_vlm
export CUDA_VISIBLE_DEVICES=0            # pin a single GPU for the gradient path

python scripts/build_person_dataset.py --download --output data/concept || \
    echo "[warn] person dataset build failed; coco_person runs will be skipped"

MODELS=( "qwen-3b" "smolvlm-500m" )     # H100: you may add "llava" "qwen" (7B, float32 fits 80 GB)
DATASETS=( "hf:microsoft/cats_vs_dogs" "coco_person" )
DTYPE=float32                            # H100: float32. <=16 GB GPU: keep models <=3B.
NIMG=96
RANK=8

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    tag=$(echo "${model}_${ds}" | tr '/:' '__')
    echo "==== Stage 1: model=$model dataset=$ds dtype=$DTYPE ===="
    srun python scripts/run_stage1.py \
        --model "$model" --dataset "$ds" \
        --dtype "$DTYPE" --n-images "$NIMG" --rank "$RANK" \
        --out "runs_stage1/${tag}" \
      || echo "[warn] stage1 $model $ds exited non-zero; continuing"
  done
done

echo "Stage 1 done. Results in runs_stage1/<tag>/{stage1_results.json,stage1.png}"