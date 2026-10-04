#!/bin/bash
# run_matrix.sh
# Runs the full 2x2x2 experimental matrix sequentially on a single GPU.

set -e

echo "======================================================"
echo " CSC-VLM: Full Experimental Matrix (8 runs)"
echo "======================================================"

# 1. Ensure the COCO Person dataset is built first
echo "[1/3] Preparing COCO Person concept dataset..."
python scripts/build_person_dataset.py --download --output data/concept

# 2. Ensure Cats vs Dogs is cached
echo "[2/3] Caching Hugging Face Cats vs Dogs..."
python scripts/prepare_data.py --dataset hf:microsoft/cats_vs_dogs

echo "[3/3] Commencing 2x2x2 Matrix..."

# Define arrays for the loops
MODELS=("llava" "qwen")
DATASETS=("hf:microsoft/cats_vs_dogs" "coco_person")
METHODS=("merge" "prune")

TOTAL=8
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
echo " All 8 runs completed successfully!"
echo " Results are located in the 'runs/' directory."
echo "======================================================"
