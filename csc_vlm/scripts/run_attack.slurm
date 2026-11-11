#!/bin/bash
# run_attack.slurm  --  pipeline v3: monitor-constrained minimum-norm attacks (Qwen2.5-VL-3B)
# Submit from the csc_vlm/ project dir:   sbatch scripts/run_attack.slurm
#
# Array tasks (one GPU each):
#   0  patch leaf, coco:bottle   instrument check + fixed-space OC/T (+ live drop_norm)
#   1  patch leaf, cats_vs_dogs  same, saturated concept
#   2  pixel leaf, coco:bottle   R2 test: live vs frozen selection, fixed pixel space
#   3  pixel leaf, cats_vs_dogs  same
# Rough H100 cost (fp32, 32 images): patch ~3-5 h, pixel ~6-9 h per task. Each task
# checkpoints after every image and stops cleanly at TIME_LIMIT_H; with --requeue
# (or a manual resubmit of the same task) it RESUMES where it stopped.
#SBATCH --job-name=csc_attack
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/attack_%A_%a.out
#SBATCH --requeue

set -uo pipefail
mkdir -p logs runs_attack runs_preflight
# module load cuda/12.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csc_vlm
export CUDA_VISIBLE_DEVICES=0

MODEL=qwen-3b
DTYPE=float32
NIMG=32
TIME_LIMIT_H=11.5
LEAVES=( patch patch pixel pixel )
DATASETS=( "coco:bottle:data/concept_bottle" "hf:microsoft/cats_vs_dogs" \
           "coco:bottle:data/concept_bottle" "hf:microsoft/cats_vs_dogs" )
ID=${SLURM_ARRAY_TASK_ID:-0}
LEAF=${LEAVES[$ID]}
DS=${DATASETS[$ID]}
TAG=$(echo "${MODEL}_${DS}_${LEAF}" | tr '/:' '__')

if [[ "$DS" == coco:bottle* ]] && [ ! -d data/concept_bottle ]; then
  python scripts/build_person_dataset.py --category bottle --output data/concept_bottle \
    || { echo "[FAIL] could not build the bottle dataset"; exit 1; }
fi

echo "==== self-check (CPU logic) ===="
python scripts/run_attack.py --self-check || { echo "[FAIL] attack self-check"; exit 1; }

echo "==== gradient preflight: $MODEL ===="
srun python scripts/preflight.py --model "$MODEL" --dataset "hf:microsoft/cats_vs_dogs" \
     --dtype "$DTYPE" --grad --out "runs_preflight/${MODEL}_grad_attack" \
  || { echo "[FAIL] gradient preflight failed -- not running attacks"; exit 1; }

if [ "$LEAF" == "patch" ]; then
  MODES="free,thr,eq"
  METHODS="pool,drop_random,drop_norm"
else
  MODES="free,thr"
  METHODS="pool,drop_random,drop_norm,drop_query"
fi

echo "==== attacks: leaf=$LEAF dataset=$DS modes=$MODES methods=$METHODS ===="
srun python scripts/run_attack.py --model "$MODEL" --dataset "$DS" --dtype "$DTYPE" \
     --leaf "$LEAF" --modes "$MODES" --methods "$METHODS" --selections frozen,live \
     --keeps 1,0.25,0.0625 --n-images "$NIMG" --time-limit-hours "$TIME_LIMIT_H" \
     --out "runs_attack/${TAG}"
echo "done: runs_attack/${TAG}/{attack_results.json,attack_summary.txt}"
echo "re-analyse all:  python scripts/analyze_attack.py runs_attack/*/attack_results.json"
