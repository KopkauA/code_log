# IDP-ESM2 Baseline — Viral-Only Retraining

## What this is
Variant of the IDP-ESM2 baseline comparison that asks: *how much of the original
viral OOD gap is a true generalization failure, vs. simply too little viral-specific
signal in the head?*

- **Original setup:** GeoHead MLP trained on non-viral BENDER sequences, evaluated
  on viral sequences as an out-of-distribution (OOD) test.
- **This script:** trains a separate GeoHead **from scratch on viral sequences only**
  (its own train/val/test split carved out of the viral subset), then evaluates on
  the viral test split. Backbone (frozen IDP-ESM2 embeddings) and head architecture
  are otherwise identical to the original.
- Comparing the two head's test-set R² isolates how much of the original OOD drop is
  distribution shift vs. viral sequences just being harder / less represented.

## Model & method
- **Backbone:** frozen IDP-ESM2 (8M and 150M variants), embeddings extracted once via
  mean pooling over the attention mask, no fine-tuning
- **Head (`GeoHead`):** `Linear(hidden_dim → 128) → SiLU → Dropout(0.1) → Linear(128 → 5)`
- **Targets:** 5 geometric BENDER targets (Rg, Ree, ν, Δ, A₀), z-score normalized using
  the *viral train split's* own mean/std
- **Optimizer:** AdamW, lr 3e-3, weight decay 1e-4, cosine annealing over 200 epochs,
  early stopping (patience 20, on a 3-epoch smoothed val loss)

## Data & splits
- Input: BENDER `merged.csv`, filtered to `kingdom == "Viruses"`
- 80/10/10 train/val/test, cluster-aware split (`GroupShuffleSplit` on `cluster_id`
  when present, random otherwise) — same splitting logic as the non-viral baseline,
  just applied within the viral subset
- Sequences run through both `8M` and `150M` IDP-ESM2 backbones independently

## How it was run
Sequential runs across 3 seeds:

```bash
bash idp_esm_trainedvirus_run.sh
```

which calls:

```bash
python3 idp_esm2_viral_only_train.py \
    --bender merged.csv \
    --out    esm2_viral_only_seed<SEED>/ \
    --models 8M 150M \
    --seed   <SEED>
```
for seeds 42, 67, 93.

**Compute:** 1× GPU, 12GB RAM, 24h wall-time limit (Gaivi cluster, `general` partition).
Checkpoints are saved with a `_viral` suffix so they never overwrite the original
non-viral-trained heads.

## Outputs (per seed, per model)
- `{model}_viral_geohead_best.pt` — best checkpoint (state dict + target mean/std)
- `{model}_viral_test_preds.csv` — per-sequence predictions vs. ground truth on the
  viral test split
- `viral_only_r2_summary.csv` — per-target R² for each model, viral-train/viral-test
