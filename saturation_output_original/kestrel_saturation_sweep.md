# KESTREL — Data-Saturation Sweep

## What this is
A sample-efficiency experiment layered onto the original all-kingdom KESTREL
script (`kestrel_ood_virus.py` gained a `--mode saturation` option): trains
KESTREL on progressively larger random subsamples of the non-viral training
pool (**n = 500, 1000, 2000, 5000**), evaluating each on the same fixed viral
OOD set, to see how OOD performance scales with training-set size.

- Same script, same architecture and viral-OOD evaluation protocol as the
  original all-kingdom run — `--mode saturation` swaps out the training loop
  for the sweep described below; `--mode train` (the default) still runs the
  original single full-dataset training.
- Reports R² only for **ν, Rg, A₀** (the sweep's target subset), not all 10
  KESTREL targets.

## Method
- Loads the full non-viral pool once; the viral OOD set is fixed and reused
  for every run in the sweep (no re-sampling of OOD).
- For each `n` in the sweep list, and for each of 3 seeds:
  1. Randomly sample `n` sequences from the non-viral pool (`random_state=seed`)
  2. 90/10 train/val split of that sample (no separate in-distribution test
     split — the point of this experiment is the OOD curve, not in-distribution
     performance)
  3. Normalization stats and kingdom vocabulary are computed from that sample's
     training split only (so they legitimately shrink/shift at small `n`, not
     just the model)
  4. Batch size adapts to sample size: `min(128, max(16, n_train // 8))`, to
     avoid single-batch validation sets at small `n`
  5. Train KESTREL from scratch on the sample, evaluate once on the fixed OOD
     viral set, record R² for ν/Rg/A₀, then delete the checkpoint (disk-space
     cleanup — only the metrics are kept per run, not per-run weights)
- 4 sample sizes × 3 seeds = 12 training runs total.

## How it was run
```bash
bash submit_saturation.sh
```
This submits **one SLURM job per seed** (all 4 sample sizes run sequentially
within each seed's job), then a lightweight aggregation job that fires once all
three seed jobs finish:

```bash
python3 kestrel_ood_virus.py \
    --mode         saturation \
    --data         merged.csv \
    --out          saturation_output/seed<SEED> \
    --sat_n        500 1000 2000 5000 \
    --sat_seeds    <SEED> \
    --sat_epochs   200 \
    --sat_patience 25 \
    --lr           5e-4
```
for seeds 42, 67, 93. The aggregation step concatenates each seed's
`saturation_results_wide.csv`, sorts by `n_train`/`seed`, reshapes to long
format, and prints a mean ± std summary per sample size.

**Compute:** per seed job — 1× GPU, 4 CPUs, 24GB RAM (no explicit time limit set
in the submission script).

## Outputs
- `seed<SEED>/saturation_results_wide.csv` — one row per (n, seed) with
  `nu_r2`, `rg_r2`, `a0_r2` columns (per-seed, before aggregation)
- `saturation_output/saturation_results_wide.csv` — all seeds concatenated,
  sorted by `n_train`/`seed` (overwritten in place by the aggregation step)
- `saturation_output/saturation_results.csv` — long format
  (`n_train`, `seed`, `target`, `r2_ood`) for plotting
- Console summary: mean ± std OOD R² for ν/Rg/A₀ at each sample size

