# Multi-Temperature Viral OOD Evaluation

## What this is
Re-evaluates the already-trained KESTREL and IDP-ESM2-8M checkpoints (trained on
the 5 non-viral kingdoms, with Viruses held out entirely) against viral ground-truth
labels computed at **5 different CALVADOS-2 simulation temperatures**: 278K, 288K,
300K, 310K, 320K.

- The model and its predictions on the 1,025 viral sequences never change across
  temperatures — only the CALVADOS-2 ground-truth labels being compared against do.
  So predictions are computed **once per seed** and reused for all 5 temperature
  comparisons, rather than re-run per temperature.
- Since Viruses were never part of either model's training data at any point, all
  5 temperature comparisons are genuine OOD tests (unlike a related
  `run_reverse_ood_eval.py` script, where leakage is a concern — not the case here).
- Confirms the 300K labels match `merged.csv`'s own viral rg/ree/ν/Δ/A₀ and
  graph-metric columns, as a sanity check that the temperature-specific CSVs are
  consistent with the main dataset.

## Method
- **KESTREL:** reloads `kestrel_best<seed>.pt`, reconstructs the *training-time*
  kingdom vocabulary and target normalization stats (mean/std) from the original
  non-viral training split (needed so the kingdom-embedding size and denormalization
  match what the checkpoint was trained with), then predicts on the viral sequences
  and denormalizes to raw physical units.
- **IDP-ESM2-8M:** reloads `IDP-ESM2-8M_geohead_best_seed<seed>.pt`, re-extracts
  frozen embeddings for the viral sequences, predicts with the saved `GeoHead`, and
  denormalizes using the checkpoint's own stored mean/std.
- For each seed, the one fixed set of raw predictions is scored (R²) against each
  temperature's own ground-truth labels independently.
- Outputs a per-model per-seed per-temperature R² table, a combined long-format
  table, and a seed-averaged summary (mean/std of R² across seeds, by model ×
  temperature × target).

## Data
- `merged.csv` — supplies the `sequence` column (joined on `protein_name` ==
  `UniProt_ID`) and is used to reconstruct KESTREL's training-time kingdom vocab
- 5 temperature-specific CSVs (`bender_<T>K.csv` for T in 278/288/300/310/320),
  each filtered to `kingdom == "Viruses"` and joined to sequences via the same
  UniProt ID mapping; the `A0` column is renamed to lowercase `a0` to match the
  rest of the codebase

## How it was run
```bash
sbatch run_multitemp_ood_eval.slurm
```
which calls:
```bash
python run_multitemp_ood_eval.py \
    --bender      merged.csv \
    --temp_csvs   278:bender_278K.csv 288:bender_288K.csv 300:bender_300K.csv \
                  310:bender_310K.csv 320:bender_320K.csv \
    --code_dir    scripts/ \
    --kestrel_ckpt_dir checkpoints/ \
    --esm2_ckpt_dir    checkpoints/ \
    --seeds 42 67 93 \
    --out results/multitemp_ood
```
`--code_dir` must point at the folder containing `kestrel_ood_virus.py` and
`idp_esm2_virus_fixed.py` (the training-script modules these imports come from),
so they're importable at eval time.

**Compute:** 1× GPU, 4 CPUs, 32GB RAM, 4h wall-time limit (Gaivi cluster, `general`
partition), `seqdance` conda environment.

## Outputs
- `kestrel_multitemp_r2.csv`, `idp_esm2_8m_multitemp_r2.csv` — per-model,
  per-seed, per-temperature R² per target
- `combined_multitemp_r2_long.csv` — both models combined, long format
  (model, seed, temperature, target, r2)
- `combined_multitemp_r2_seed_averaged.csv` — mean/std of R² across seeds,
  grouped by model × temperature × target

## Results
*(fill in after the run completes — how R² for each target shifts across the 5
simulation temperatures for each model, and whether one model is more sensitive
to the temperature at which ground truth was computed)*
