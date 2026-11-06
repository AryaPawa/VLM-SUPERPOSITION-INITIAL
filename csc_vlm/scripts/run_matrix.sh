#!/bin/bash
# run_matrix.sh
# Runs the 2x2x2 experimental matrix (8 runs) sequentially on a single GPU:
#   models   = llava, qwen         (edit MODELS below to add more)
#   datasets = cats_vs_dogs, coco_person
#   methods  = merge, prune
# Matches the README ("Full 8-run runs"). Estimated time on one H100: ~30-45 min.
#
# RESUME: re-running after a pre-emption RESUMES — each run uses --resume and skips
# grid points already in its checkpoint.json. Nothing is deleted.
# FRESH RUN: to force every grid point to be re-measured (e.g. after changing the
# metric/hook code), start fresh with:   FRESH=1 bash scripts/run_matrix.sh
# (that moves any existing runs/ aside to runs_backup_<timestamp>/).

set -uo pipefail                 # NOT -e: one failed run/download must not abort the matrix

FRESH="${FRESH:-0}"
if [ "$FRESH" = "1" ] && [ -d runs ] && [ -n "$(ls -A runs 2>/dev/null)" ]; then
    backup="runs_backup_$(date +%Y%m%d_%H%M%S)"
    echo "[fresh] FRESH=1 set -> moving existing runs/ to $backup"
    mv runs "$backup"
fi
mkdir -p runs

# --------------------------------------------------------------------- #
# Model stack. DEFAULT = the two primary models (README's 2x2x2 = 8 runs).
# Uncomment the rest for the full proposal stack (then pre-cache ALL of them on
# the login node, and expect ~36 runs).
# --------------------------------------------------------------------- #
MODELS=(
    "llava"           # LLaVA-1.5 7B
    "qwen"            # Qwen2.5-VL 7B
    # "llava-13b"     # LLaVA-1.5 13B
    # "llava-next"    # LLaVA-NeXT 7B
    # "qwen-3b"       # Qwen2.5-VL 3B
    # "internvl-2b"   # InternVL3 2B  (uses OpenGVLab/InternVL3-2B-hf)
    # "internvl"      # InternVL3 8B  (uses OpenGVLab/InternVL3-8B-hf)
    # "smolvlm-500m"  # SmolVLM 500M
    # "smolvlm"       # SmolVLM2 2.2B
)
DATASETS=("hf:microsoft/cats_vs_dogs" "coco_person")
METHODS=("merge" "prune")

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#METHODS[@]} ))
echo "======================================================"
echo " CSC-VLM: ${#MODELS[@]}x2x2 matrix = $TOTAL runs"
echo " Models: ${MODELS[*]}"
echo "======================================================"

# 1. Build the COCO Person dataset (non-fatal: cats_vs_dogs runs still proceed if it fails)
echo "[1/3] Preparing COCO Person concept dataset..."
python scripts/build_person_dataset.py --download --output data/concept \
    || echo "[warn] person dataset build failed; coco_person runs may be skipped"

# 2. Cache Cats vs Dogs
echo "[2/3] Caching Hugging Face Cats vs Dogs..."
python scripts/prepare_data.py --dataset hf:microsoft/cats_vs_dogs \
    || echo "[warn] cats_vs_dogs cache step failed; the run will try to download on first use"

echo "[3/3] Commencing ${#MODELS[@]}x2x2 matrix..."
COUNT=1
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for method in "${METHODS[@]}"; do
            dataset_name="cats_dogs"
            [ "$dataset" = "coco_person" ] && dataset_name="person"
            out_dir="runs/${model}_${method}_${dataset_name}"
            echo ""
            echo "------------------------------------------------------"
            echo " RUN $COUNT/$TOTAL: model=$model data=$dataset_name method=$method"
            echo " out: $out_dir"
            echo "------------------------------------------------------"
            python run.py \
                --model "$model" \
                --dataset "$dataset" \
                --compression-method "$method" \
                --out "$out_dir" \
                --resume \
              || echo "[warn] run $COUNT ($model/$dataset_name/$method) exited non-zero; continuing"
            COUNT=$((COUNT+1))
        done
    done
done

echo ""
echo "======================================================"
echo " Matrix finished ($TOTAL runs attempted). Results in runs/"
echo " Re-run to resume anything that failed/was pre-empted."
echo "======================================================"