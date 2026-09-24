#!/bin/bash
#SBATCH --job-name=seqdance_bender_ood
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/seqdance_bender_ood_%j.out
#SBATCH --error=logs/seqdance_bender_ood_%j.err

# NOTE: create logs/ BEFORE submitting:
#   mkdir -p ./scripts/logs

source ~/.bashrc
conda activate seqdance

SCRIPT_DIR=./scripts
DATA_DIR=./data
RESULTS_DIR=./results
CHECKPOINT=./model.safetensors

mkdir -p ${RESULTS_DIR}

# --- Step 1: extract embeddings for both in-distribution and OOD viral sets ---
python ${SCRIPT_DIR}/extract_bender_embeddings.py \
    --input ${DATA_DIR}/merged.csv \
    --checkpoint ${CHECKPOINT} \
    --output_prefix ${DATA_DIR}/bender \
    --model_select seqdance \
    --max_len 1024

# --- Step 2: train regression, evaluate OOD R^2, compute ES_OOD ---
# Fill in --kestrel_r2 with your own KESTREL R^2 values on this same viral holdout
python ${SCRIPT_DIR}/eval_bender_ood.py \
    --input ${DATA_DIR}/merged.csv \
    --train_emb ${DATA_DIR}/bender_train_emb.pkl \
    --viral_emb ${DATA_DIR}/bender_viral_emb.pkl \
    --pca_dim 200 \
    --kestrel_r2 "nu=0.8915,delta=0.8590,a0=0.7774,rg=0.9417" \
    --output ${RESULTS_DIR}/bender_ood_results.csv

echo "Done."
