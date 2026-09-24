#!/bin/bash
# =============================================================================
# submit_saturation.sh
#
# Submits one Slurm job per random seed (42, 67, 93) in parallel.
# Each job trains KESTREL on n = 500 / 1000 / 2000 / 5000 sequences
# from the non-viral BENDER pool, evaluates OOD on viral sequences,
# and writes results to  OUT_ROOT/seed{S}/saturation_results_wide.csv.
#
# After all three seed jobs finish, a lightweight aggregation job
# concatenates the per-seed CSVs into  OUT_ROOT/saturation_results.csv.
#
# Usage:
#   bash submit_saturation.sh
# =============================================================================

set -euo pipefail

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KESTREL_PY="${SCRIPT_DIR}/kestrel_ood_virus.py"
DATA_CSV="./merged.csv"
OUT_ROOT="${SCRIPT_DIR}/saturation_output"

# ── experiment parameters ─────────────────────────────────────────────────────
SEEDS=(42 67 93)
SAT_N="500 1000 2000 5000"   # passed as space-separated list to --sat_n
SAT_EPOCHS=200
SAT_PATIENCE=25
LR=5e-4

# ── Slurm resources ───────────────────────────────────────────────────────────
             # change to your cluster's GPU partition
CPUS=4
MEM="24G"
             # 8 h should be ample for 4×n on one seed

mkdir -p "${OUT_ROOT}"

# ── submit one job per seed ───────────────────────────────────────────────────
SEED_JOB_IDS=()

for SEED in "${SEEDS[@]}"; do
    SEED_OUT="${OUT_ROOT}/seed${SEED}"
    mkdir -p "${SEED_OUT}"

    JOB_ID=$(sbatch --parsable \
        --job-name="saturation_seed${SEED}" \
        --output="${SEED_OUT}/slurm_%j.out" \
        --error="${SEED_OUT}/slurm_%j.err" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --gres=gpu:1 \
        --wrap="python3 ${KESTREL_PY} \
            --mode saturation \
            --data '${DATA_CSV}' \
            --out '${SEED_OUT}' \
            --sat_n ${SAT_N} \
            --sat_seeds ${SEED} \
            --sat_epochs ${SAT_EPOCHS} \
            --sat_patience ${SAT_PATIENCE} \
            --lr ${LR}")

    echo "Seed ${SEED} → Job ${JOB_ID}  (out: seed${SEED}/)"
    SEED_JOB_IDS+=("${JOB_ID}")
done

# ── aggregation job: runs after ALL seed jobs finish ──────────────────────────
DEPENDENCY="afterany"
for JID in "${SEED_JOB_IDS[@]}"; do
    DEPENDENCY="${DEPENDENCY}:${JID}"
done

AGG_PY=$(cat <<'PYEOF'
import pandas as pd, glob, os, sys

out_root = sys.argv[1]
pattern  = os.path.join(out_root, "seed*", "saturation_results_wide.csv")
files    = sorted(glob.glob(pattern))

if not files:
    print("ERROR: no saturation_results_wide.csv found under", out_root)
    sys.exit(1)

wide = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
wide = wide.sort_values(["n_train", "seed"]).reset_index(drop=True)
wide.to_csv(os.path.join(out_root, "saturation_results_wide.csv"), index=False)

# long form for plotting
long_rows = []
for _, row in wide.iterrows():
    for t in ["nu", "rg", "a0"]:
        col = f"{t}_r2"
        if col in row:
            long_rows.append({"n_train": row["n_train"],
                               "seed":   row["seed"],
                               "target": t,
                               "r2_ood": row[col]})
long = pd.DataFrame(long_rows)
long.to_csv(os.path.join(out_root, "saturation_results.csv"), index=False)

# print summary
print("\nSATURATION SUMMARY (mean ± std across seeds)")
print(f"{'n_train':>8}  {'nu':>12}  {'rg':>12}  {'a0':>12}")
for n, grp in wide.groupby("n_train"):
    row_str = f"{n:>8}"
    for t in ["nu", "rg", "a0"]:
        col = f"{t}_r2"
        m, s = grp[col].mean(), grp[col].std()
        row_str += f"  {m:.3f}±{s:.3f}"
    print(row_str)

print(f"\nSaved → {out_root}/saturation_results_wide.csv")
print(f"       → {out_root}/saturation_results.csv")
PYEOF
)

AGG_JOB_ID=$(sbatch --parsable \
    --job-name="saturation_agg" \
    --output="${OUT_ROOT}/slurm_agg_%j.out" \
    --error="${OUT_ROOT}/slurm_agg_%j.err" \
    --dependency="${DEPENDENCY}" \
    --ntasks=1 \
    --mem="2G" \
    --time="00:05:00" \
    --wrap="python3 -c '${AGG_PY}' '${OUT_ROOT}'")

echo ""
echo "Aggregation job: ${AGG_JOB_ID}"
echo "  → depends on seed jobs: ${SEED_JOB_IDS[*]}"
echo "  → fires once all 3 seeds finish"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Results at:    ${OUT_ROOT}/saturation_results_wide.csv"
echo "Done."
