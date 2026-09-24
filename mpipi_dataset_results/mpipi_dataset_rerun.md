# mpipi Dataset Re-Run — KESTREL, IDP-ESM2, ProtT5-XL

## What this is
All three sequence-based baselines (KESTREL, IDP-ESM2, ProtT5-XL) re-run on a
**new dataset** (`mpipi_results_raw.csv`, from the Mpipi force field) instead of
the original `merged.csv`. Same overall protocol as before — non-viral training,
Viruses held out entirely as OOD, 3 seeds each — but two things changed across
all three scripts to accommodate the new data:

1. **Target columns are renamed.** `mpipi_results_raw.csv` uses different target
   column names than `merged.csv`; all three scripts apply the same rename before
   anything else:

   | mpipi column     | renamed to |
   |-------------------|------------|
   | `rg_nm_mpipi`      | `rg`       |
   | `ree_nm_mpipi`     | `ree`      |
   | `nu_mpipi`         | `nu`       |
   | `delta_mpipi`      | `delta`    |
   | `A0_mpipi`         | `a0`       |

   Note: `rg` is mapped from the *fit-derived* `rg_nm_mpipi`, not the directly
   measured `rg_mean_nm` — done for consistency with `ree`, which only has a
   fit-derived form in this dataset. Flag if a different Rg convention is wanted.

2. **KESTREL's graph-topology head was removed.** The 5 graph-topological targets
   (global efficiency, fragmentation index, avg. clustering, transitivity, degree
   assortativity) aren't present in `mpipi_results_raw.csv` and aren't computable
   from it, so this KESTREL variant (`kestrel_geo_only.py`) predicts only the 5
   geometric targets. IDP-ESM2 and ProtT5-XL were already geometric-only, so they
   needed no equivalent change beyond the rename.

## Models re-run

| Model | Script | Backbone | Targets |
|---|---|---|---|
| KESTREL (geo-only) | `kestrel_geo_only.py` | transformer over one-hot sequence, per-kingdom embedding | 5 geometric |
| IDP-ESM2 (8M, 150M) | `idp_esm2_virus_fixed_newdata.py` | frozen IDP-ESM2, GeoHead MLP | 5 geometric |
| ProtT5-XL | `run_prott5_geohead_on_bender_newdata.py` | frozen ProtT5-XL (`Rostlab/prot_t5_xl_uniref50`), GeoHead MLP | 5 geometric |

All three keep the same splitting logic: Viruses separated out entirely as OOD,
remaining kingdoms split 80/10/10 (cluster-aware via `cluster_id` when present),
and all three are run across seeds **42, 67, 93**.

## How each was run

**KESTREL:**
```bash
bash kestrel_ood_virus_run.sh
```
which runs (not multi-seed aware internally, so one invocation per seed):
```bash
python3 kestrel_geo_only.py --data mpipi_results_raw.csv --out kestrel_seed<SEED>_mpipi/ --seed <SEED>
```
Compute: 1× GPU, 12GB RAM, 24h limit, `calvados_env`.

**IDP-ESM2:**
```bash
bash idpesm2_run.sh
```
which runs (both model sizes per invocation):
```bash
python3 idp_esm2_virus_fixed_newdata.py --bender mpipi_results_raw.csv --out esm2_seed<SEED>_mpipi/ --models 8M 150M --seed <SEED>
```
Compute: 1× GPU, 12GB RAM, 24h limit, `esm2_env`.

**ProtT5-XL:**
```bash
bash submit_prott5_geohead.sh
```
This submits a 3-job SLURM chain instead of one script per seed, since ProtT5-XL
embedding extraction is the expensive step and is shared across seeds:
1. **extract** — runs the script once with `--extract-only`, caching embeddings
   for the non-OOD and OOD (viral) splits to disk (skipped if the cache already
   exists, or via `--skip-extract`)
2. **train array** — SLURM array `[1-3]`, one task per seed (task 1→42, 2→67,
   3→93), each running `--seeds <SEED>` against the cached embeddings, with a
   `--dependency=afterok` on the extract job
3. **aggregate** — runs `--aggregate-only` once all 3 array tasks finish, to
   compute the across-seed mean/std summary

Compute: extract — 1× GPU, 4 CPUs, 48GB RAM; train — 1× GPU, 4 CPUs, 24GB RAM per
task; aggregate — 1 CPU, 4GB RAM. All on `general` partition, `seqdance` conda env.

## Outputs
- **KESTREL** (per seed): `kingdom_vocab.txt`, `{model}_best.pt`,
  `{model}_history.csv`, `target_mean.csv`/`target_std.csv`,
  `{model}_{test,ood}_r2.csv`, `{model}_{test,ood}_kingdom_r2.csv`,
  `{model}_{test,ood}_preds.csv` (for `kestrel` and `physchem_mlp`)
- **IDP-ESM2** (per seed, per model size): `{model}_geohead_best.pt`,
  `{model}_results.csv`, `{model}_{split}_preds.csv`, `all_results.csv`
- **ProtT5-XL** (per seed): `prott5xl_seed<seed>_results.csv`,
  `prott5xl_seed<seed>_{bender_test,ood_viruses}_predictions.csv`; after all
  seeds: `all_seeds_results.csv`, `aggregated_results.csv` (mean ± std per
  target, per split)
