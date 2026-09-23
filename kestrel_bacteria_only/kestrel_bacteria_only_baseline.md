# KESTREL Baseline — Bacteria-Only Training

## What this is
Kingdom-restricted variant of KESTREL: trains on **Bacteria sequences only** (instead
of all non-viral kingdoms pooled together), while keeping the same viral OOD holdout,
to test whether the original all-kingdom training pool was diluting or helping
single-kingdom performance.

- **Original setup:** KESTREL trained on all non-viral kingdoms pooled together,
  evaluated OOD on held-out viral sequences.
- **This script:** same architecture and training loop, but the train/val/test pool
  is restricted to `kingdom == "Bacteria"` before splitting (`--train_kingdom`
  argument, default `"Bacteria"`). The viral OOD holdout is unaffected — it's carved
  out of the full dataset before any kingdom restriction is applied.
- Also retrains the **PhyschemMLP** ablation baseline on the same bacteria-only split,
  for a like-for-like comparison.

## Model & method
- **KESTREL:** transformer backbone over one-hot amino-acid sequences (max length 256),
  positional encoding, learned per-kingdom embedding (dim 16) concatenated in — unknown/OOD
  kingdoms (including the held-out Viruses) map to a zero embedding — with specialist
  heads over the 10 targets
- **Targets (10):** 5 geometric (Rg, Ree, ν, Δ, A₀) + 5 graph-topological (global
  efficiency, fragmentation index, avg. clustering, transitivity, degree assortativity)
- **PhyschemMLP:** ablation baseline using only the 13 physicochemical descriptor
  columns (κ, SCD, SHD, FCR, NCPR, etc.) instead of the sequence itself
- **Optimizer:** AdamW, cosine annealing, early stopping (patience 15 for KESTREL,
  20 for PhyschemMLP, on a 3-epoch smoothed val loss)

## Data & splits
- Input: BENDER `merged.csv`
- OOD split carved out first: all `kingdom == "Viruses"` sequences, held out entirely
- Remaining pool restricted to `kingdom == "Bacteria"`, then 80/10/10 train/val/test,
  cluster-aware (`cluster_id`-based) when available, random otherwise
- Kingdom vocabulary for the per-kingdom embedding is built from the training split
  only (so here, effectively just `"Bacteria"`), saved to `kingdom_vocab.txt`

## How it was run
The script isn't multi-seed aware internally, so each seed is a separate invocation:

```bash
bash kestrel_bacteria_only_run.sh
```

which calls:

```bash
python3 kestrel_bacteria_only.py --data merged.csv --out kestrel_bact_seed<SEED>/ --seed <SEED>
```
for seeds 42, 67, 93.

**Compute:** 1× GPU, 12GB RAM, 24h wall-time limit (Gaivi cluster, `general` partition).

## Outputs (per seed)
- `kingdom_vocab.txt` — kingdom → index mapping used for the run
- `{model}_best.pt` — best checkpoint for `kestrel` and `physchem_mlp`
- `{model}_history.csv` — per-epoch train/val loss
- `target_mean.csv`, `target_std.csv` — normalization stats from the training split
- `{model}_{test,ood}_r2.csv` — per-target R² on the bacteria test split and the
  viral OOD split
- `{model}_{test,ood}_kingdom_r2.csv` — R² broken out by kingdom
- `{model}_{test,ood}_preds.csv` — per-sequence predictions vs. ground truth

## Results
*(fill in after the run completes — KESTREL vs. PhyschemMLP R² on bacteria-test and
viral-OOD, per seed, and comparison against the original all-kingdom-trained model's
bacteria-subset and viral-OOD performance)*
