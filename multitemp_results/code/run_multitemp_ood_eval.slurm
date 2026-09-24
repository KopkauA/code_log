#!/bin/bash
#SBATCH --job-name=multitemp_ood_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/multitemp_ood_eval_%j.out
#SBATCH --error=logs/multitemp_ood_eval_%j.err

# NOTE: create logs/ BEFORE submitting:
#   mkdir -p ./scripts/logs

source ~/.bashrc
conda activate seqdance

CODE_DIR=./scripts
BENDER_CSV=./merged.csv
KESTREL_CKPT_DIR=./checkpoints
ESM2_CKPT_DIR=./checkpoints
OUT_DIR=./results/multitemp_ood

# EDIT these 5 paths to wherever you place the temperature CSVs on Gaivi
TEMP_278=./bender_278K.csv
TEMP_288=./bender_288K.csv
TEMP_300=./bender_300K.csv
TEMP_310=./bender_310K.csv
TEMP_320=./bender_320K.csv

python ${CODE_DIR}/run_multitemp_ood_eval.py \
    --bender ${BENDER_CSV} \
    --temp_csvs 278:${TEMP_278} 288:${TEMP_288} 300:${TEMP_300} 310:${TEMP_310} 320:${TEMP_320} \
    --code_dir ${CODE_DIR} \
    --kestrel_ckpt_dir ${KESTREL_CKPT_DIR} \
    --esm2_ckpt_dir ${ESM2_CKPT_DIR} \
    --seeds 42 67 93 \
    --out ${OUT_DIR}

echo "Done."
