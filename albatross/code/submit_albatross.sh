#!/bin/bash
# =============================================================================
# submit_albatross.sh
#
# Multi-seed ALBATROSS retrain on BENDER_BIO.csv.
# One Slurm array task per seed → trains all 10 targets for that seed.
# An aggregation job fires once all seeds finish.
#
# Usage:
#   bash submit_albatross.sh [--seeds "42 123 456 789 1337"] [--targets "rg ree nu"] [--data /path/to/BENDER_BIO.csv]
#
# Defaults: 3 seeds, all 10 targets.
# =============================================================================

set -euo pipefail

# ── CONFIGURABLE ─────────────────────────────────────────────────────────────
SEEDS=(42 67 93)
TARGETS="rg ree nu delta a0 global_efficiency fragmentation_index avg_clustering transitivity degree_assortativity"

DATA="/path/to/merged.csv"           # ← update on HPC
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SCRIPT="$SCRIPT_DIR/albatross_retrain.py"
OUT_ROOT="$SCRIPT_DIR/output"
LOG_DIR="$SCRIPT_DIR/logs"

MAX_LEN=256
BATCH_SIZE=64
EPOCHS=100
LR=1e-3
# ─────────────────────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)   IFS=' ' read -r -a SEEDS <<< "$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --data)    DATA="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NUM_SEEDS=${#SEEDS[@]}
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "ALBATROSS multi-seed retrain"
echo "Seeds (${NUM_SEEDS}): ${SEEDS[*]}"
echo "Targets: $TARGETS"
echo "Output root: $OUT_ROOT"
echo "================================================================"

# Write seeds to file so array tasks can look up their seed by index
SEED_FILE="$SCRIPT_DIR/seeds.txt"
printf '%s\n' "${SEEDS[@]}" > "$SEED_FILE"

# ── ARRAY JOB  (one task per seed) ───────────────────────────────────────────
TRAIN_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=albatross_retrain
#SBATCH --output=${LOG_DIR}/train_seed%a_%j.out
#SBATCH --error=${LOG_DIR}/train_seed%a_%j.err
#SBATCH --array=1-${NUM_SEEDS}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

SEED=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${SEED_FILE}")
OUT_DIR="${OUT_ROOT}/seed_\${SEED}"

echo "Task \${SLURM_ARRAY_TASK_ID} / seed \${SEED} → \${OUT_DIR}"

python "${SCRIPT}" --data "${DATA}" --out "\${OUT_DIR}" --targets ${TARGETS} --max_len ${MAX_LEN} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --seed \${SEED}
EOF
)

echo "Training array job : ${TRAIN_JOB_ID}  (${NUM_SEEDS} tasks)"

# ── AGGREGATION JOB  (fires after all seeds) ─────────────────────────────────
AGG_JOB_ID=$(sbatch --parsable \
    --job-name=albatross_agg \
    --output="${LOG_DIR}/aggregate_%j.out" \
    --error="${LOG_DIR}/aggregate_%j.err" \
    --dependency="afterany:${TRAIN_JOB_ID}" \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=4G \
    --wrap="python ${SCRIPT_DIR}/aggregate_seeds.py --out_root ${OUT_ROOT} --seeds ${SEEDS[*]}")

echo "Aggregation job    : ${AGG_JOB_ID}  (fires after ${TRAIN_JOB_ID})"
echo "Done."
