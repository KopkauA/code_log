# ProtT5-XL Baseline — Non-Viral Training, Viral OOD

## What this is
The original ProtT5-XL comparison baseline: trains a lightweight `GeoHead` MLP
on top of **frozen ProtT5-XL embeddings**, on non-viral BENDER sequences, then
evaluates on held-out **viral sequences as OOD** — same protocol family as the
IDP-ESM2 and KESTREL baselines. This is the version run on the original
`merged.csv` dataset (as opposed to the later `mpipi_results_raw.csv` rerun,
documented separately).

## Model & method
- **Backbone:** frozen ProtT5-XL (`Rostlab/prot_t5_xl_uniref50`, hidden dim
  1024). Sequences are space-separated and non-standard amino acids (U, Z, O,
  B) mapped to X before tokenizing, per ProtT5 convention. Embeddings are
  mean-pooled over non-padding positions (EOS token included, standard
  practice for ProtT5).
- **Head (`GeoHead`):** `Linear(1024 → 128) → SiLU → Dropout(0.1) → Linear(128 → 5)`
  — identical to GeoGraph's `FeaturesHead`
- **Targets:** all 5 BENDER geometric targets (Rg, Ree, ν, Δ, A₀); rows missing
  any of the 5 (including A₀, which IDRome lacks but BENDER has) are dropped
- **Optimizer:** AdamW, lr 3e-3, cosine annealing over 200 epochs, early
  stopping (patience 20, on a smoothed val loss)

## Data & splits
- Input: BENDER `merged.csv`
- Viral sequences (`kingdom == "Viruses"`) separated out entirely as the OOD set
- Remaining non-viral sequences: 80/10/10 train/val/test, cluster-aware
  (`cluster_id`-based) when available, random otherwise
- Run across 3 seeds: 42, 67, 93

## How it was run
Same HPC pipeline shape as the mpipi ProtT5 rerun — embeddings are the
expensive step and are extracted **once**, shared across all 3 seeds, via a
3-job SLURM chain:

```bash
bash submit_prott5_geohead.sh               # full pipeline
bash submit_prott5_geohead.sh --skip-extract  # if the embedding cache already exists
```

1. **extract** — `--extract-only`, caches embeddings for the non-OOD and OOD
   (viral) splits to disk
2. **train array** — SLURM array `[1-3]`, one task per seed (task 1→42, 2→67,
   3→93), `--seeds <SEED>` against the cached embeddings, depends on the
   extract job (`afterok`)
3. **aggregate** — `--aggregate-only`, fires once all 3 array tasks finish

Equivalent manual invocation:
```bash
python run_prott5_geohead_on_bender.py --extract-only --emb-cache <dir>
python run_prott5_geohead_on_bender.py --seeds 42 --emb-cache <dir>
python run_prott5_geohead_on_bender.py --seeds 67 --emb-cache <dir>
python run_prott5_geohead_on_bender.py --seeds 93 --emb-cache <dir>
python run_prott5_geohead_on_bender.py --aggregate-only
```
(or `python run_prott5_geohead_on_bender.py` with no arguments to run all
seeds sequentially in one local process)

**Compute:** extract — 1× GPU, 4 CPUs, 48GB RAM; train — 1× GPU, 4 CPUs, 24GB
RAM per array task; aggregate — 1 CPU, 4GB RAM. All on `general` partition,
`seqdance` conda env.

## Outputs
- `prott5xl_seed<seed>_results.csv` — R² summary per seed (bender_test,
  ood_viruses)
- `prott5xl_seed<seed>_bender_test_predictions.csv`,
  `prott5xl_seed<seed>_ood_viruses_predictions.csv` — per-sequence
  true/predicted/residual values
- `all_seeds_results.csv`, `aggregated_results.csv` — combined across seeds,
  with mean ± std per target per split
