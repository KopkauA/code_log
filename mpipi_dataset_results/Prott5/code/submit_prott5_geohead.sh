#!/bin/bash
# =============================================================================
# submit_prott5_geohead.sh
#
# Submits ProtT5-XL + GeoHead evaluation to Slurm:
#
#   Job 1 (embed)  — extract ProtT5 embeddings once and cache to disk
#   Job 2 (train)  — Slurm array [1-3], task ID maps to seed (42, 67, 93)
#   Job 3 (agg)    — aggregate per-seed CSVs after the array finishes
#
# Usage:
#   bash submit_prott5_geohead.sh               # full pipeline
#   bash submit_prott5_geohead.sh --skip-extract # if cache already exists
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
SEEDS=(42 67 93)          # task 1 → 42, task 2 → 67, task 3 → 93
N_SEEDS=${#SEEDS[@]}      # 3

# --- Gaivi paths (edited from the original macOS /Volumes/... paths) ---
BENDER_CSV="./mpipi_results_raw.csv"
OUT_DIR="./results/prott5_newdataset"
EMB_CACHE="./prott5_emb_cache"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_prott5_geohead_on_bender_newdata.py"

CONDA_ENV="seqdance"   # confirmed working env with torch/transformers on Gaivi

# Slurm resource requests
EXTRACT_MEM="48G";  EXTRACT_CPUS=4;  EXTRACT_GPUS=1
TRAIN_MEM="24G";    TRAIN_CPUS=4;    TRAIN_GPUS=1
AGG_MEM="4G";       AGG_CPUS=1

# Gaivi partition -- confirmed via `sinfo`: general* is the default partition
SBATCH_PARTITION="--partition=general"

# ── Parse flags ───────────────────────────────────────────────────────────────
SKIP_EXTRACT=false
for arg in "$@"; do
    [[ "$arg" == "--skip-extract" ]] && SKIP_EXTRACT=true
done

# ── Setup ─────────────────────────────────────────────────────────────────────
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

SEEDS_STR="${SEEDS[*]}"

ACTIVATE=""
[[ -n "$CONDA_ENV" ]] && ACTIVATE="source ~/.bashrc && conda activate ${CONDA_ENV} &&"

# ── Step 1: Embedding extraction (one-time, GPU-heavy job) ────────────────────
NON_OOD_EMB="${EMB_CACHE}/non_ood_emb.pt"
OOD_EMB="${EMB_CACHE}/ood_emb.pt"

if $SKIP_EXTRACT || { [[ -f "$NON_OOD_EMB" ]] && [[ -f "$OOD_EMB" ]]; }; then
    echo "Cache found at ${EMB_CACHE} — skipping extraction job."
    EMB_JOB_ID=""
    TRAIN_DEP_LINE=""
else
    echo "Submitting embedding extraction job …"
    EMB_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=prott5_extract
#SBATCH --output=${LOG_DIR}/extract_%j.out
#SBATCH --error=${LOG_DIR}/extract_%j.err
#SBATCH ${SBATCH_PARTITION}
#SBATCH --cpus-per-task=${EXTRACT_CPUS}
#SBATCH --mem=${EXTRACT_MEM}
#SBATCH --gres=gpu:${EXTRACT_GPUS}

${ACTIVATE} python "${PYTHON_SCRIPT}" \
    --bender      "${BENDER_CSV}" \
    --out         "${OUT_DIR}" \
    --emb-cache   "${EMB_CACHE}" \
    --extract-only
EOF
    )
    echo "  → Extraction job: ${EMB_JOB_ID}"
    TRAIN_DEP_LINE="#SBATCH --dependency=afterok:${EMB_JOB_ID}"
fi

# ── Step 2: Seed training — Slurm array [1-N_SEEDS] ──────────────────────────
# Array task ID is 1-based: task 1 → SEEDS[0]=42, task 2 → SEEDS[1]=67, etc.
echo "Submitting training array job (${N_SEEDS} tasks) …"
echo "  Seeds: ${SEEDS[*]}"

TRAIN_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=prott5_train
#SBATCH --output=${LOG_DIR}/train_%a.out
#SBATCH --error=${LOG_DIR}/train_%a.err
#SBATCH --array=1-${N_SEEDS}
#SBATCH ${SBATCH_PARTITION}
#SBATCH --cpus-per-task=${TRAIN_CPUS}
#SBATCH --mem=${TRAIN_MEM}
#SBATCH --gres=gpu:${TRAIN_GPUS}
${TRAIN_DEP_LINE}

# Map 1-based task ID to seed value
SEEDS=(${SEEDS_STR})
SEED_IDX=\$(( SLURM_ARRAY_TASK_ID - 1 ))
SEED=\${SEEDS[\$SEED_IDX]}

echo "Array task \${SLURM_ARRAY_TASK_ID} → seed \${SEED}"

${ACTIVATE} python "${PYTHON_SCRIPT}" \
    --bender      "${BENDER_CSV}" \
    --out         "${OUT_DIR}" \
    --emb-cache   "${EMB_CACHE}" \
    --seeds       \${SEED}
EOF
)
echo "  → Training array job: ${TRAIN_JOB_ID} (tasks 1-${N_SEEDS})"

# ── Step 3: Aggregation (fires once ALL array tasks finish) ───────────────────
AGG_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=prott5_aggregate
#SBATCH --output=${LOG_DIR}/aggregate_%j.out
#SBATCH --error=${LOG_DIR}/aggregate_%j.err
#SBATCH ${SBATCH_PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${AGG_CPUS}
#SBATCH --mem=${AGG_MEM}
#SBATCH --dependency=afterok:${TRAIN_JOB_ID}

${ACTIVATE} python "${PYTHON_SCRIPT}" \
    --out            "${OUT_DIR}" \
    --seeds          ${SEEDS_STR} \
    --aggregate-only
EOF
)
echo "  → Aggregation job: ${AGG_JOB_ID}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Job graph:"
if [[ -n "${EMB_JOB_ID}" ]]; then
    echo "  extract  (${EMB_JOB_ID})"
    echo "    └─ train array (${TRAIN_JOB_ID})"
else
    echo "  [cache hit — no extract job]"
    echo "  train array (${TRAIN_JOB_ID})"
fi
echo "         task 1 → seed 42"
echo "         task 2 → seed 67"
echo "         task 3 → seed 93"
echo "    └─ aggregate (${AGG_JOB_ID})"
echo ""
echo "Logs → ${LOG_DIR}"
echo "Done."
