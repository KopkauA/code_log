#!/bin/bash
# =============================================================================
# train_geograph_bender.sh
#
# Submits a 3-element SLURM array job (one task per seed).
# Each task trains GeoGraph independently; results land in seed<N>/.
#
# Usage:
#   bash train_geograph_bender.sh
# =============================================================================

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
SEEDS=(42 69 93)
NUM_SEEDS=${#SEEDS[@]}

TRAIN_PY="./train_geograph_bender.py"
CSV="./data/bender_calvados_complete.csv"
RES_DIR="./residue_features"
OUT_ROOT="./geograph_training"
LOG_DIR="${OUT_ROOT}/slurm_logs"

mkdir -p "${LOG_DIR}"

for SEED in "${SEEDS[@]}"; do
    mkdir -p "${OUT_ROOT}/seed${SEED}"
done

# ── submit array job ──────────────────────────────────────────────────────────
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash

#SBATCH --job-name=geograph_bender
#SBATCH --output=${LOG_DIR}/%A_%a.out
#SBATCH --error=${LOG_DIR}/%A_%a.err
#SBATCH --array=0-$((NUM_SEEDS - 1))
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

export PYTHONUNBUFFERED=1

SEEDS=(${SEEDS[@]})
SEED=\${SEEDS[\$SLURM_ARRAY_TASK_ID]}
SEED_OUT="${OUT_ROOT}/seed\${SEED}"

echo "=== GeoGraph — BENDER training ==="
echo "Seed: \${SEED}   Task ID: \${SLURM_ARRAY_TASK_ID}"
echo "Start: \$(date)"

python3 ${TRAIN_PY} \
    --csv "${CSV}" \
    --residue-dir "${RES_DIR}" \
    --output-dir "\${SEED_OUT}" \
    --max-seq-len 256 \
    --epochs 100 \
    --batch-size 512 \
    --lr 5e-4 \
    --seed \${SEED} \
    --device cuda

echo "Done: \$(date)"
EOF
)

echo "Submitted array job: ${JOB_ID} (${NUM_SEEDS} tasks, seeds: ${SEEDS[*]})"
echo "Slurm logs (by task index): ${LOG_DIR}/${JOB_ID}_<task_id>.out/.err"
echo "Training results (by seed): ${OUT_ROOT}/seed{${SEEDS[*]}}/"