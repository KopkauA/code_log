# ALBATROSS Baseline — Per-Target BRNN Retrain

## What this is
Reimplementation/retrain of the **ALBATROSS** architecture (PARROT's `BRNN_MtO`:
bidirectional LSTM → linear head) on the `BENDER_BIO.csv` dataset, as a sequence-based
comparison baseline. Unlike KESTREL/IDP-ESM2/ProtT5-XL, which use one model with
shared representations across all targets, ALBATROSS trains **one independent model
per target** (10 separate BRNNs), matching the original PARROT/SPARROW convention.

- Also includes `aggregate_seeds.py`, a standalone helper that collects each seed's
  `summary_r2.csv` and reports mean ± std R² across seeds.

## Model & method
- **Architecture:** `BRNN_MtO` — bidirectional LSTM followed by a linear head over
  the concatenated final forward/backward hidden states (many-to-one), inlined
  from PARROT
- **Encoding:** PARROT one-hot (20 canonical amino acids, alphabetical order)
- **Per-target hidden size / layer count:** geometric targets (Rg, Ree, ν, Δ, A₀)
  use the same `(hidden_size, num_layers)` as the published SPARROW v2 weights,
  inferred from checkpoint tensor shapes; the 5 graph-topological targets have no
  published ALBATROSS analogue, so they default to `(64, 2)` (overridable via
  `--hidden_size`/`--num_layers`)
- **Loss:** L1 (matching PARROT), Adam optimizer
- **Stopping:** PARROT-style auto-stop — after a minimum of `--epochs` (default 25),
  stops once validation loss has failed to improve by more than 0.5% over that many
  consecutive epochs, up to a hard cap of 5000 epochs
- **Targets (10, independently):** 5 geometric (Rg, Ree, ν, Δ, A₀) + 5
  graph-topological (global efficiency, fragmentation index, avg. clustering,
  transitivity, degree assortativity) — graph columns are created as all-NaN if
  absent from the input CSV, so those targets are silently skipped (too few
  training rows) rather than erroring

## Data & splits
- Input: `BENDER_BIO.csv`
- Same splitting convention as `kestrel_ood_virus.py`: `kingdom == "Viruses"` held
  out entirely as OOD, remaining kingdoms split 80/10/10, cluster-aware
  (`cluster_id`-based) when available, random otherwise
- Split row-indices are persisted per seed (`split_{train,val,test,ood}.csv`) for
  reproducibility

## How it was run
```bash
bash submit_albatross.sh
```
which submits a SLURM array (one task per seed, all 10 targets trained
sequentially within each task):
```bash
python albatross_retrain.py --data BENDER_BIO.csv --out output/seed_<SEED> \
    --targets rg ree nu delta a0 global_efficiency fragmentation_index \
              avg_clustering transitivity degree_assortativity \
    --max_len 256 --batch_size 64 --epochs 100 --lr 1e-3 --seed <SEED>
```
for seeds 42, 67, 93, followed by an aggregation job that fires once all array
tasks finish:
```bash
python aggregate_seeds.py --out_root output/ --seeds 42 67 93
```
`submit_albatross.sh` also accepts `--seeds`, `--targets`, and `--data` overrides
on the command line.

**Compute:** training — 1× GPU, 4 CPUs, 16GB RAM per seed (no time limit set);
aggregation — 1 CPU, 4GB RAM.

## Outputs (SPARROW-compatible layout)
- `seed_<SEED>/<target>/network.pt` — trained weights for that target
- `seed_<SEED>/<target>/norm_stats.csv` — target mean/std used for normalization
- `seed_<SEED>/<target>/history.csv` — per-epoch train/val loss and val R²
- `seed_<SEED>/split_{train,val,test,ood}.csv` — row indices for each split
- `seed_<SEED>/summary_r2.csv` — per-target test/OOD R² for that seed
- `aggregate_test_r2.csv`, `aggregate_ood_r2.csv` — mean ± std R² per target,
  across all seeds

