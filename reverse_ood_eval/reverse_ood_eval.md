# KESTREL vs. IDP-ESM2-8M — Reverse (Non-Viral) Evaluation

## What this is
The "reverse" of the usual viral-OOD test: reloads the already-trained KESTREL
and IDP-ESM2-8M checkpoints (both trained on the 5 non-viral kingdoms —
Bacteria, Plants, Fungi, Mammals, Protists — with Viruses held out entirely)
and evaluates both on the **5 non-viral kingdoms themselves**, reporting R² per
kingdom per model. This is the script referenced (and explicitly distinguished
from) in the multi-temperature viral OOD evaluation's docstring.

Pass `--test_split_only` to instead reconstruct and evaluate only on each
model's original held-out test split (same seed + clustering logic used at
training time) — a cleaner, leak-free comparison, at the cost of a smaller
per-kingdom sample size. **Neither this run nor the results section below
specifies which mode was actually used — confirm and record that before citing
these numbers.**

## Method
- **KESTREL:** reconstructs the exact training split for each seed via
  `kestrel_ood_virus.py`'s own `make_splits`, rebuilds the kingdom vocabulary
  and target normalization stats from that split's training data, reloads
  `kestrel_best<seed>.pt`, and evaluates (full non-viral set or test-split-only
  per the flag), reporting R² broken out by kingdom via `r2_by_kingdom`.
- **IDP-ESM2-8M:** applies `idp_esm2_virus_fixed.py`'s own filtering (drop
  missing targets/sequence, truncate to `MAX_SEQ_LEN`) to match its training
  preprocessing exactly, reloads `IDP-ESM2-8M_geohead_best<seed>.pt`, extracts
  frozen embeddings once (embeddings don't depend on seed), and evaluates
  per-kingdom R² the same way.
- Both models use the **same cluster-aware 80/10/10 split logic**, so they're
  in an identical leakage situation — this is called out explicitly in the
  script for both.
- Outputs per-model per-seed per-kingdom R², a combined long-format table, and
  a seed-averaged summary (mean/std of R² across seeds, by model × kingdom ×
  target).

## Data
- `merged.csv` — full BENDER dataset (all kingdoms, including Viruses, though
  Viruses aren't used in this script)
- Requires `kestrel_ood_virus.py` and `idp_esm2_virus_fixed.py` to be
  importable (place alongside this script or point `--code_dir` at them)

## How it was run
```bash
sbatch run_reverse_ood_eval.slurm
```
which calls:
```bash
python run_reverse_ood_eval.py \
    --bender            merged.csv \
    --code_dir          scripts/ \
    --kestrel_ckpt_dir  checkpoints/ \
    --esm2_ckpt_dir     checkpoints/ \
    --seeds 42 67 93 \
    --out               results/reverse_ood
```

**Compute:** 1× GPU, 4 CPUs, 32GB RAM, 4h wall-time limit, `general`
partition, `seqdance` conda env.

## Outputs
- `kestrel_reverse_ood_by_kingdom.csv`, `idp_esm2_8m_reverse_ood_by_kingdom.csv`
  — per-model, per-seed, per-kingdom R² per target
- `combined_r2_long.csv` — both models combined, long format
  (model, seed, kingdom, target, r2, n)
- `combined_r2_seed_averaged.csv` — mean/std of R² across seeds, grouped by
  model × kingdom × target
