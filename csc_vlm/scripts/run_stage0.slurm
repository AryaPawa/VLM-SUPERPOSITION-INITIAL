#!/bin/bash
# run_stage0.slurm  --  pipeline v2: preflight + Stage 0 (forward only) on one GPU.
# Submit from the csc_vlm/ project dir:   sbatch scripts/run_stage0.slurm
# Edit MODELS / DATASETS / the conda env name below for your cluster.
#SBATCH --job-name=csc_stage0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/stage0_%j.out
#SBATCH --requeue

set -uo pipefail                       # not -e: one failed model must not stop the others
mkdir -p logs runs_stage0 runs_preflight

# module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csc_vlm

# concept datasets (skip if cached). coco:bottle is a NON-saturating concept.
python scripts/build_person_dataset.py --download --output data/concept || echo "[warn] person build failed"
python scripts/build_person_dataset.py --category bottle --output data/concept_bottle || echo "[warn] bottle build failed"

MODELS=( "qwen-3b" "llava" )                       # add: smolvlm-500m internvl-2b qwen llava-13b
DATASETS=( "hf:microsoft/cats_vs_dogs" "coco_person" "coco:bottle:data/concept_bottle" )
DTYPE=auto                                          # bf16 on H100/A100, fp16 on older GPUs

for model in "${MODELS[@]}"; do
  tag0=$(echo "${model}" | tr '/:' '__')
  echo "==== preflight: $model ===="
  if ! srun python scripts/preflight.py --model "$model" --dataset "hf:microsoft/cats_vs_dogs" \
        --dtype "$DTYPE" --out "runs_preflight/${tag0}"; then
    echo "[FAIL] preflight failed for $model -- skipping its Stage 0 runs (see runs_preflight/${tag0})"
    continue
  fi
  for ds in "${DATASETS[@]}"; do
    tag=$(echo "${model}_${ds}" | tr '/:' '__')
    echo "==== Stage 0: model=$model dataset=$ds ===="
    srun python scripts/run_stage0.py --model "$model" --dataset "$ds" --dtype "$DTYPE" \
        --n-eval 256 --n-monitor 144 --out "runs_stage0/${tag}" \
      || echo "[warn] stage0 $model $ds exited non-zero; continuing"
  done
done
echo "Stage 0 done. Results: runs_stage0/<tag>/{stage0_results.json,stage0.png}"
