#!/usr/bin/env bash
#SBATCH --job-name=cscvlm_matrix
#SBATCH --partition=dgx_all
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=logs/cscvlm_%j.out
#SBATCH --error=logs/cscvlm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=debdan.samanta02@gmail.com

set -euo pipefail
mkdir -p logs runs

cd $HOME/kdag_projects/VLM/csc_vlm

source $HOME/miniconda3/bin/activate vlmenv

export HF_HOME=$HOME/hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo " CSC-VLM Matrix Run"
echo " Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo " Node: $(hostname)"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=================================================================="

bash scripts/run_matrix.sh

echo "=================================================================="
echo " Done | $(date)"
echo "=================================================================="
