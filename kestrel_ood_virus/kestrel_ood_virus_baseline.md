# KESTREL Baseline — All-Kingdom Training, Viral OOD

## What this is
The original KESTREL training run: all 5 non-viral kingdoms pooled together for
training, with **Viruses held out entirely** as a genuine out-of-distribution (OOD)
test set. This is the baseline that the bacteria-only variant and the
multi-temperature OOD evaluation both build on and compare against.

- Trains **KESTREL** and a **PhyschemMLP** ablation baseline side by side on the
  same splits, so sequence-based and physicochemical-descriptor-based predictions
  can be compared directly.
- Checkpoints and normalization stats produced here (`kestrel_best<seed>.pt`,
  `target_mean.csv`, `target_std.csv`) are the ones later reloaded by the
  multi-temperature OOD evaluation script.

## Model & method
- **KESTREL:** transformer backbone over one-hot amino-acid sequences (max length
  256), positional encoding, learned per-kingdom embedding (dim 16) concatenated
  in — unknown/OOD kingdoms (including the held-out Viruses) map to a zero
  embedding — with specialist heads over the 10 targets
- **Targets (10):** 5 geometric (Rg, Ree, ν, Δ, A₀) + 5 graph-topological (global
  efficiency, fragmentation index, avg. clustering, transitivity, degree
  assortativity)
- **PhyschemMLP:** ablation baseline using only the 13 physicochemical descriptor
  columns (κ, SCD, SHD, FCR, NCPR, etc.) instead of the sequence itself
- **Optimizer:** AdamW, cosine annealing, early stopping (patience 15 for KESTREL,
  20 for PhyschemMLP, on a 3-epoch smoothed val loss)

## Data & splits
- Input: BENDER `merged.csv`
- Viral sequences (`kingdom == "Viruses"`) separated out entirely as the OOD set
- Remaining 5 kingdoms pooled together, then 80/10/10 train/val/test, cluster-aware
  (`cluster_id`-based) when available, random otherwise
- Kingdom vocabulary for the per-kingdom embedding is built from the training split
  only, saved to `kingdom_vocab.txt`

## How it was run
The script isn't multi-seed aware internally, so each seed is a separate invocation:

```bash
bash kestrel_ood_virus_run.sh
```

which calls:

```bash
python3 kestrel_ood_virus.py --data merged.csv --out kestrel_seed<SEED>/ --seed <SEED>
```
for seeds 42, 67, 93.

**Compute:** 1× GPU, 12GB RAM, 24h wall-time limit (Gaivi cluster, `general`
partition).

## Outputs (per seed)
- `kingdom_vocab.txt` — kingdom → index mapping used for the run
- `{model}_best.pt` — best checkpoint for `kestrel` and `physchem_mlp`
- `{model}_history.csv` — per-epoch train/val loss
- `target_mean.csv`, `target_std.csv` — normalization stats from the training split
  (reused later for OOD evaluation at other temperatures)
- `{model}_{test,ood}_r2.csv` — per-target R² on the non-viral test split and the
  viral OOD split
- `{model}_{test,ood}_kingdom_r2.csv` — R² broken out by kingdom
