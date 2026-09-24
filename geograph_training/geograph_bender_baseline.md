# GeoGraph Baseline — BENDER Training

## What this is
Training run for **GeoGraph** (comparison baseline) on the BENDER dataset, predicting
intrinsically disordered protein (IDP) structural properties directly from sequence.

- **Backbone:** ESM-style transformer, initialized from a specified pretrained tokenizer/config
- **Heads:** two linear heads on top of mean-pooled embeddings
  - Sequence-level head → 5 geometric targets (end-to-end distance, radius of gyration,
    asphericity, scaling-law exponent, scaling-law prefactor) + 8 graph/network-topology
    features (fragmentation index, avg. shortest path length, global efficiency, avg.
    clustering, transitivity, degree/charge/hydrophobicity assortativity)
  - Residue-level head → 7 per-residue graph features (degree/betweenness/harmonic
    centrality, PageRank, core number, local clustering coefficient, in-LCC)
- **Loss:** combined sequence-level + residue-level regression loss (residue loss masked
  to valid positions per sequence)
- **Optimizer/schedule:** Adam, OneCycleLR with cosine annealing and linear warmup

## Data & splits
- Input: `bender_calvados_complete.csv` (sequence-level targets) +
  `residue_features/` (per-UniProt-ID `.npz` files with per-residue targets)
- All targets z-score normalized (mean/std computed on the training split, stored with
  the checkpoint for inference-time de-normalization)
- **Splits:**
  - **OOD test:** all sequences with `kingdom == "Viruses"`, held out entirely
  - **Remaining kingdoms:** 80/10/10 train/val/test, split by `cluster_id` (via
    `GroupShuffleSplit`, so sequences from the same cluster never span splits)

## How it was run
Multi-seed run (seeds 42, 69, 93) submitted as a SLURM job array, one task per seed:

```bash
bash train_geograph_bender.sh
```

This wraps and submits:

```bash
python3 train_geograph_bender.py \
    --csv ./data/bender_calvados_complete.csv \
    --residue-dir ./residue_features \
    --output-dir ./geograph_training/seed<SEED> \
    --max-seq-len 256 \
    --epochs 100 \
    --batch-size 512 \
    --lr 5e-4 \
    --seed <SEED> \
    --device cuda
```

**Compute:** 1× A100 GPU, 4 CPUs, 32GB RAM, 48h wall-time limit per seed (Gaivi cluster).

## Outputs (per seed)
- `best_model.ckpt` — checkpoint with lowest validation loss (state dict + config +
  feature normalization stats)
- `epoch_<N>.ckpt` — periodic checkpoints every 10 epochs
- `history.json` — per-epoch train/val loss (sequence + residue components) and per-feature
  val R²
- Console/SLURM logs report final loss and per-feature R² (tagged `[GEO]`/`[GRF]`) on the
  held-out test split and the viral OOD split
