# Experiment Log

Index of every training/evaluation run in this repo, with a link to its
detailed log (method, exact commands, compute used, and results).

## Baselines — original BENDER dataset (`merged.csv`)

| Log | Model | What it tests |
|---|---|---|
| [`geograph_bender_baseline.md`](./geograph_bender_baseline.md) | GeoGraph | Sequence+graph baseline, 3-seed viral OOD |
| [`kestrel_ood_virus_baseline.md`](./kestrel_ood_virus_baseline.md) | KESTREL + PhyschemMLP | All-kingdom training, viral OOD (main baseline) |
| [`idp_esm2_baseline.md`](./idp_esm2_baseline.md) | IDP-ESM2 (8M, 150M) | Frozen-backbone + GeoHead, viral OOD |
| [`prott5_baseline.md`](./prott5_baseline.md) | ProtT5-XL | Frozen-backbone + GeoHead, viral OOD |
| [`albatross_baseline.md`](./albatross_baseline.md) | ALBATROSS (per-target BRNN) | `BENDER_BIO.csv`, per-target retrain |

## KESTREL variants & ablations

| Log | What it tests |
|---|---|
| [`kestrel_bacteria_only_baseline.md`](./kestrel_bacteria_only_baseline.md) | Training restricted to Bacteria only vs. all-kingdom pool |
| [`idp_esm2_viral_only_baseline.md`](./idp_esm2_viral_only_baseline.md) | GeoHead retrained on viral sequences only (vs. non-viral→viral OOD) |
| [`kestrel_taxonomy_confound_control.md`](./kestrel_taxonomy_confound_control.md) | Appendix E control: isolates taxonomic diversity vs. sample size |
| [`kestrel_saturation_sweep.md`](./kestrel_saturation_sweep.md) | OOD R² vs. training-set size (n = 500 / 1000 / 2000 / 5000) |

## Cross-model comparisons

| Log | What it tests |
|---|---|
| [`multitemp_ood_eval.md`](./multitemp_ood_eval.md) | KESTREL & IDP-ESM2-8M fixed predictions vs. labels at 5 CALVADOS-2 temperatures |
| [`reverse_ood_eval.md`](./reverse_ood_eval.md) | KESTREL vs. IDP-ESM2-8M evaluated on the 5 *non-viral* kingdoms |
| [`seqdance_kestrel_comparison.md`](./seqdance_kestrel_comparison.md) | SeqDance embeddings + linear regression vs. KESTREL, viral OOD (ES_OOD ratio) |

## New dataset — mpipi force field (`mpipi_results_raw.csv`)

| Log | What it tests |
|---|---|
| [`mpipi_dataset_rerun.md`](./mpipi_dataset_rerun.md) | KESTREL (geo-only), IDP-ESM2, and ProtT5-XL re-run on the mpipi dataset |

## Shared conventions across all runs

- **Splits:** `kingdom == "Viruses"` held out entirely as an out-of-distribution
  (OOD) test set; remaining kingdoms split 80/10/10 train/val/test,
  cluster-aware (via `cluster_id`) where available, random otherwise.
- **Seeds:** 42, 67, 93 for every multi-seed run.
- **Sequence length cap:** 256 residues, unless a log notes otherwise.
- **Targets:** 5 geometric (Rg, Ree, ν, Δ, A₀) and, where applicable, 5
  graph-topological (global efficiency, fragmentation index, avg. clustering,
  transitivity, degree assortativity) properties.

## How to read a log

Each file follows the same structure: what the experiment is and why, the
model/method, data and splits, the exact commands used to run it, output file
names, and a results section with the final numbers.
