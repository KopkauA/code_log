# IDP-ESM2 Baseline — Non-Viral Training, Viral OOD

## What this is
The original IDP-ESM2 baseline: trains a lightweight `GeoHead` MLP on top of a
**frozen IDP-ESM2 backbone**, on non-viral BENDER sequences, then evaluates on
held-out **viral sequences as a genuine OOD test** — matching the GeoGraph
evaluation protocol. This is the baseline that the viral-only retraining script
and the multi-temperature OOD evaluation both build on.

- Runs both IDP-ESM2 model sizes (8M and 150M) in a single invocation, each with
  its own `GeoHead` trained independently.
- Checkpoints produced here (`IDP-ESM2-<size>_geohead_best.pt`) are the ones later
  reloaded by the multi-temperature OOD evaluation script (for the 8M model).

## Model & method
- **Backbone:** frozen IDP-ESM2 (`InstaDeepAI/IDP-ESM2-8M` / `-150M`), embeddings
  extracted once via mean pooling over the attention mask (non-padding tokens
  only), no fine-tuning
- **Head (`GeoHead`):** `Linear(hidden_dim → 128) → SiLU → Dropout(0.1) → Linear(128 → 5)`
  — architecture matches GeoGraph's `FeaturesHead`
- **Targets:** 5 geometric BENDER targets (Rg, Ree, ν, Δ, A₀), z-score normalized
  using the training split's own mean/std
- **Optimizer:** AdamW, lr 3e-3 (matching GeoGraph's GeoHead training), weight
  decay 1e-4, cosine annealing over 200 epochs, early stopping (patience 20, on a
  3-epoch smoothed val loss)
- Sequences truncated to 256 aa, matching the GeoGraph evaluation protocol

## Data & splits
- Input: BENDER `merged.csv`
- Viral sequences (`kingdom == "Viruses"`) separated out entirely as the OOD set
- Remaining non-viral sequences: 80/10/10 train/val/test, cluster-aware
  (`cluster_id`-based) when available, random otherwise
- Both splits run through each IDP-ESM2 backbone (8M, 150M) independently

## How it was run
```bash
python run_idp_esm2_on_bender.py \
    --bender merged.csv \
    --out    idp_esm2_results/ \
    --models 8M 150M \
    --seed   42
```
(script filename on disk: `idp_esm2_virus.py`)

## Outputs
- `{model}_geohead_best.pt` — best checkpoint (state dict + target mean/std) per
  model size
- `{model}_results.csv` — per-target R² on the BENDER test split and the viral
  OOD split, per model
- `all_results.csv` — combined results across both model sizes and both splits

## Results
*(fill in after the run completes — R² per target for 8M and 150M on BENDER-test
vs. OOD-Viruses, e.g. the ν R² summary table the script prints)*
