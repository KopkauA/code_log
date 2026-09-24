# KESTREL — Taxonomy Confound Control (Appendix E)

## What this is
Addresses a reviewer-raised confound in the existing Bacteria-only ablation
(Appendix E): that ablation compares KESTREL trained on Bacteria-only
(n ≈ 2,849) against KESTREL trained on full BENDER (n > 9,200), which
confounds **taxonomic diversity** with **sample size** — any OOD gain from A→C
could be due to either factor.

This script adds a size-matched control condition and runs all three side by
side on the same fixed viral OOD test set (n = 1,025):

| Condition | Description | n | Kingdoms |
|---|---|---|---|
| **A** `bacteria_only` | all Bacteria sequences | ≈2,849 | 1 |
| **B** `matched_multitaxon` *(new control)* | proportional sample across all 5 non-viral kingdoms, size-matched to A | ≈2,849 | 5 |
| **C** `full_bender` | all non-viral sequences | ≈9,247 | 5 |

**Interpretation guide** (as the script prints it):
- B ≈ C >> A → taxonomic diversity drives the gain, not sample size
- B ≈ A << C → sample size drives the gain, not diversity
- A < B < C → both contribute independently

Imports shared infrastructure (`IDPDataset`, `KESTREL`, `build_kingdom_vocab`,
`train_model`, `evaluate`, `r2_scores`, target/constant definitions) directly
from `kestrel_ood_virus.py` rather than duplicating it.

## Method
- **Condition B's proportional sampling:** each kingdom contributes
  `floor(n × weight)` sequences, where `weight` is that kingdom's share of the
  non-viral pool; leftover seats from flooring go to the kingdoms with the
  largest fractional remainders, so the sample sums to exactly n. A kingdom
  smaller than its quota is sampled *with* replacement (only relevant if a
  minority kingdom is smaller than its proportional share of n); otherwise
  sampling is without replacement.
- **Splits:** each condition gets a simple random 90/10 train/val split (not
  cluster-aware) — no in-distribution test split within conditions, since the
  comparison of interest is OOD generalization.
- **Model/training:** identical `KESTREL` architecture and `train_model` loop
  as the main script; normalization stats and kingdom vocabulary are
  recomputed fresh per condition, per seed, from that condition's own training
  split. Batch size adapts to sample size (`min(256, max(16, n_train // 8))`).
- Evaluated once per condition/seed on the fixed OOD viral set; checkpoints are
  deleted after evaluation to save disk (only metrics are retained).
- Also computes an **ES_OOD proxy** per condition: R²(condition) / R²(C, full
  BENDER), so each condition's OOD performance is expressed relative to the
  full-diversity ceiling.

## Data
- Same BENDER CSV as the main KESTREL runs (`merged.csv`) — must contain
  `kingdom` and `sequence` columns
- 3 conditions × 3 seeds = 9 training runs total

## How it was run
This is now bundled into the same submission script as the data-saturation
sweep (`submit_saturation.sh` — updated to run two independent experiments;
the saturation half is unchanged and documented separately):

```bash
bash submit_saturation.sh
```

The taxonomy-confound half submits **one SLURM job per seed** (all 3
conditions run sequentially within each seed's job), then an aggregation job
that fires once all three seed jobs finish:

```bash
python3 kestrel_taxonomy_confound_control.py \
    --data_csv merged.csv \
    --out_dir  taxonomy_confound_output/seed<SEED> \
    --seeds    <SEED> \
    --epochs   100 \
    --patience 15
```
for seeds 42, 67, 93. The aggregation step concatenates each seed's
`taxonomy_confound_results.csv`, sorts by condition/seed, and prints the
mean ± std R² and ES_OOD-proxy tables across seeds.

**Compute:** per seed job — 1× GPU, 4 CPUs, 24GB RAM, `general` partition (no
explicit time limit set).

## Outputs
- `seed<SEED>/taxonomy_confound_results.csv` — one row per condition
  (A/B/C) for that seed, with `n_train`, `n_kingdoms`, and per-target R²
- `taxonomy_confound_output/taxonomy_confound_results.csv` — all seeds
  concatenated (written by the aggregation job)
- Console summary: mean ± std OOD R² (Rg, ν, Δ, A₀) per condition, plus the
  ES_OOD-proxy table
