# SeqDance vs. KESTREL — Viral OOD Comparison

## What this is
Compares **SeqDance** (a protein language model pretrained on molecular
dynamics and normal mode analysis data) against KESTREL's own OOD
generalization result, on the same fixed viral holdout, for 4 shared targets
(ν, Δ, A₀, Rg).

Rather than KESTREL's specialist heads, SeqDance's frozen embeddings are fed
into a **simple linear regression** — the goal is to isolate how much OOD
signal is already present in SeqDance's embeddings on their own, not to give
SeqDance a comparably tuned head. Computes:

```
ES_OOD = R²_SeqDance / R²_KESTREL
```
for each target, so a value near 1.0 means SeqDance's raw embeddings recover
about as much OOD signal as KESTREL's full trained model; well below 1.0 means
KESTREL's task-specific training is doing real work beyond what's already
linearly present in SeqDance's representations.

## Method — two-step pipeline
**Step 1 — `extract_bender_embeddings.py`:** extracts mean-pooled SeqDance
embeddings for every sequence in the merged BENDER/KESTREL CSV, from a
**local** `.safetensors` checkpoint (not downloaded from the Hugging Face
Hub) using SeqDance's own `ESMwrap` model class. Loads the ESM2-35M tokenizer,
builds the model shell, loads the checkpoint weights with `strict=False`
(warns on any missing/unexpected keys — expected in this pretrained-weights
setup). Splits the dataset before extracting:
- in-distribution training pool: `kingdom != "Viruses"`
- OOD holdout: `kingdom == "Viruses"`

Embeddings are saved as two pickles (`<prefix>_train_emb.pkl`,
`<prefix>_viral_emb.pkl`), each a dict keyed by `UniProt_ID`.

`--model_select` can be `seqdance` (randomly initialized, dynamics-only —
the one actually used here) or `esmdance` (freezes ESM2 weights underneath);
default and actual choice for this run is `seqdance`.

**Step 2 — `eval_bender_ood.py`:** loads both embedding pickles, optionally
reduces to `--pca_dim` dimensions via PCA (200 dims, used here), fits an
independent `LinearRegression` per target on the training-pool embeddings,
predicts on the viral holdout, and reports R² per target. KESTREL's own R² on
the same viral holdout is passed in manually via `--kestrel_r2` (not
recomputed here) to compute the ES_OOD ratio.

## Data
- Input: BENDER/KESTREL merged CSV (`merged.csv`)
- Length filter: 0–1024 residues (`--min_len`/`--max_len`, wider than the
  256-aa cap used elsewhere since SeqDance's embeddings aren't truncated the
  same way)
- Targets evaluated: ν, Δ, A₀, Rg

## How it was run
```bash
sbatch run_bender_ood2.slurm
```
which runs the two steps in sequence:
```bash
python extract_bender_embeddings.py \
    --input merged.csv \
    --checkpoint model.safetensors \
    --output_prefix bender \
    --model_select seqdance \
    --max_len 1024

python eval_bender_ood.py \
    --input merged.csv \
    --train_emb bender_train_emb.pkl \
    --viral_emb bender_viral_emb.pkl \
    --pca_dim 200 \
    --kestrel_r2 "nu=0.8915,delta=0.8590,a0=0.7774,rg=0.9417" \
    --output bender_ood_results.csv
```
The `--kestrel_r2` values are KESTREL's own previously-computed R² on this
same viral holdout, hardcoded into the submission script rather than
recomputed live — update these if KESTREL is retrained.

**Compute:** 1× GPU, 4 CPUs, 32GB RAM, 4h wall-time limit, `general`
partition, `seqdance` conda env.

## Outputs
- `<prefix>_train_emb.pkl`, `<prefix>_viral_emb.pkl` — mean-pooled SeqDance
  embeddings, keyed by UniProt ID
- `bender_ood_results.csv` — per-target `R2_SeqDance`, `R2_KESTREL`,
  `ES_OOD`
