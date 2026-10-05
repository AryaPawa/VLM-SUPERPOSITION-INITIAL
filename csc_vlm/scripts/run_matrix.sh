#!/bin/bash
# run_matrix.sh
# Runs the full 9x2x2 experimental matrix (36 runs) sequentially on a single GPU.
# Estimated time on H100: ~2.2 hours.
# Safe to re-submit after pre-emption: --resume skips completed grid points.

set -e

echo "======================================================"
echo " CSC-VLM: Full Experimental Matrix (36 runs)"
echo " Models:  9 (LLaVA-1.5 7B/13B, LLaVA-NeXT, Qwen 3B/7B,"
echo "             InternVL3 2B/8B, SmolVLM 500M/2.2B)"
echo " Datasets: 2 (cats_vs_dogs, coco_person)"
echo " Methods:  2 (merge, prune)"
echo "======================================================"

# 1. Ensure the COCO Person dataset is built first
echo ""
echo "[1/3] Preparing COCO Person concept dataset..."
python scripts/build_person_dataset.py --download --output data/concept

# 2. Ensure Cats vs Dogs is cached
echo "[2/3] Caching Hugging Face Cats vs Dogs..."
python scripts/prepare_data.py --dataset hf:microsoft/cats_vs_dogs

echo "[3/3] Commencing 9x2x2 Matrix..."

# --------------------------------------------------------------------- #
# Model stack (matches research proposal table exactly)
# Format: "short_name"
# --------------------------------------------------------------------- #
MODELS=(
    # PRIMARY
    "llava"           # LLaVA-1.5 7B
    "llava-13b"       # LLaVA-1.5 13B
    "llava-next"      # LLaVA-NeXT 7B
    "qwen-3b"         # Qwen2.5-VL 3B
    "qwen"            # Qwen2.5-VL 7B
    # SECONDARY
    "internvl-2b"     # InternVL3 2B
    "internvl"        # InternVL3 8B
    # SLM ARM
    "smolvlm-500m"    # SmolVLM 500M
    "smolvlm"         # SmolVLM 2.2B
)

DATASETS=("hf:microsoft/cats_vs_dogs" "coco_person")
METHODS=("merge" "prune")

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} * ${#METHODS[@]} ))
COUNT=1

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for method in "${METHODS[@]}"; do

            # Format a clean run directory name
            dataset_name="cats_dogs"
            if [ "$dataset" = "coco_person" ]; then
                dataset_name="person"
            fi

            out_dir="runs/${model}_${method}_${dataset_name}"

            echo ""
            echo "------------------------------------------------------"
            echo " RUN $COUNT/$TOTAL: Model=$model | Data=$dataset_name | Method=$method"
            echo " Output: $out_dir"
            echo "------------------------------------------------------"

            python run.py \
                --model "$model" \
                --dataset "$dataset" \
                --compression-method "$method" \
                --out "$out_dir" \
                --resume

            COUNT=$((COUNT+1))
        done
    done
done

echo ""
echo "======================================================"
echo " All $TOTAL runs completed successfully!"
echo " Results are located in the 'runs/' directory."
echo "======================================================"
