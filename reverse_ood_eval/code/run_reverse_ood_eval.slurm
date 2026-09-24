#!/bin/bash
#SBATCH --job-name=reverse_ood_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/reverse_ood_eval_%j.out
#SBATCH --error=logs/reverse_ood_eval_%j.err

# NOTE: create logs/ BEFORE submitting:
#   mkdir -p ./scripts/logs

source ~/.bashrc
conda activate seqdance   # or whichever env has torch/pandas/transformers for you

# --- paths (edit these) ---
CODE_DIR=./scripts          # where kestrel_ood_virus.py + idp_esm2_virus_fixed.py live
BENDER_CSV=./data/merged.csv
KESTREL_CKPT_DIR=./checkpoints
ESM2_CKPT_DIR=./checkpoints
OUT_DIR=./results/reverse_ood

python ${CODE_DIR}/run_reverse_ood_eval.py \
    --bender ${BENDER_CSV} \
    --code_dir ${CODE_DIR} \
    --kestrel_ckpt_dir ${KESTREL_CKPT_DIR} \
    --esm2_ckpt_dir ${ESM2_CKPT_DIR} \
    --seeds 42 67 93 \
    --out ${OUT_DIR}

echo "Done."
