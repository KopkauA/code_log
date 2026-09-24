"""
Taxonomy Confound Control for Appendix E Bacteria Ablation
===========================================================
Addresses the sample-size confound raised by reviewers: the existing ablation
compares KESTREL trained on Bacteria-only (n≈2,849) against KESTREL trained on
full BENDER (n>9,200), confounding taxonomic diversity with sample size.

This script runs three conditions on the fixed OOD viral test set (n=1,025):

  A) bacteria_only        – all Bacteria sequences, 1 taxon    (n≈2,849)
  B) matched_multitaxon   – proportional sample across all 5
                            non-viral taxa, same n as condition A  [NEW CONTROL]
  C) full_bender          – all non-viral sequences, 5 taxa    (n≈9,247)

If the A→C gain is driven by taxonomic diversity, B ≈ C >> A.
If the gain is driven by sample size,              B ≈ A << C.

Usage
-----
  python kestrel_taxonomy_confound_control.py \\
      --data_csv /path/to/bender.csv \\
      --out_dir  taxonomy_confound_output/ \\
      --seeds 42 67 93 \\
      --epochs 100 --patience 15
"""

import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from collections import deque

# ── import shared infrastructure from sibling script ────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from kestrel_ood_virus import (
    AA_VOCAB, AA_TO_IDX, VOCAB_SIZE,
    GEO_TARGETS, GRAPH_TARGETS, ALL_TARGETS,
    KINGDOM_DIM, OOD_KINGDOM,
    IDPDataset, KESTREL,
    build_kingdom_vocab,
    get_device, r2_scores,
    train_model, evaluate,
)

# ─────────────────────────────────────────────
# SAMPLING HELPERS
# ─────────────────────────────────────────────

def sample_proportional(pool_df: pd.DataFrame,
                        n: int,
                        seed: int,
                        kingdom_col: str = "kingdom") -> pd.DataFrame:
    """
    Draw n sequences from pool_df, stratified proportionally by kingdom.

    Each kingdom k contributes floor(n * w_k) sequences where
    w_k = |k| / |pool|. Remainder is filled by random sampling without
    replacement from the pool to hit exactly n.

    Parameters
    ----------
    pool_df     : non-viral sequences (all kingdoms)
    n           : target sample size
    seed        : random seed for reproducibility
    kingdom_col : column that contains kingdom labels

    Returns
    -------
    DataFrame of exactly n rows with diverse kingdom representation.
    """
    rng = np.random.default_rng(seed)
    kingdoms = pool_df[kingdom_col].values
    unique_k, counts = np.unique(kingdoms, return_counts=True)
    weights = counts / counts.sum()

    alloc = (weights * n).astype(int)          # floor allocation
    remainder = n - alloc.sum()                 # leftover seats

    # distribute remainder to kingdoms with largest fractional parts
    fracs = (weights * n) - alloc
    top_k = np.argsort(-fracs)[:remainder]
    alloc[top_k] += 1

    assert alloc.sum() == n, f"allocation bug: {alloc.sum()} != {n}"

    parts = []
    for k, a in zip(unique_k, alloc):
        sub = pool_df[pool_df[kingdom_col] == k]
        if a > len(sub):
            # sample with replacement if a taxon is smaller than its quota
            sampled = sub.sample(n=a, replace=True, random_state=int(seed))
        else:
            sampled = sub.sample(n=a, replace=False, random_state=int(seed))
        parts.append(sampled)

    result = pd.concat(parts).sample(frac=1, random_state=int(seed))  # shuffle
    return result.reset_index(drop=True)


def split_train_val(df: pd.DataFrame,
                    val_frac: float = 0.10,
                    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple random 90/10 train/val split (no held-out test within conditions)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = max(1, int(val_frac * len(df)))
    val_df   = df.iloc[idx[:n_val]].reset_index(drop=True)
    train_df = df.iloc[idx[n_val:]].reset_index(drop=True)
    return train_df, val_df


# ─────────────────────────────────────────────
# CONDITION RUNNER
# ─────────────────────────────────────────────

def run_condition(name: str,
                  train_df: pd.DataFrame,
                  val_df: pd.DataFrame,
                  ood_df: pd.DataFrame,
                  seed: int,
                  out_dir: str,
                  epochs: int,
                  patience: int,
                  device) -> dict:
    """
    Train one KESTREL instance on (train_df, val_df), evaluate OOD on ood_df.

    Returns a dict with keys:
        condition, seed, n_train, n_kingdoms, + one key per target.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # normalization stats from train split only
    stats = {
        "mean": train_df[ALL_TARGETS].mean(),
        "std":  train_df[ALL_TARGETS].std().clip(lower=1e-6),
    }

    kingdom_vocab = build_kingdom_vocab(train_df)
    n_kingdoms    = len(kingdom_vocab)

    bs = min(256, max(16, len(train_df) // 8))
    kw = dict(num_workers=0, pin_memory=False)

    def make_loader(df_, shuffle):
        ds = IDPDataset(df_, max_len=256,
                        target_stats=stats,
                        kingdom_vocab=kingdom_vocab)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, **kw)

    train_dl = make_loader(train_df, True)
    val_dl   = make_loader(val_df,   False)
    ood_dl   = make_loader(ood_df,   False)

    run_tag = f"{name}_seed{seed}"
    run_out = os.path.join(out_dir, run_tag)
    os.makedirs(run_out, exist_ok=True)

    model = train_model(
        KESTREL(n_kingdoms=n_kingdoms, kingdom_dim=KINGDOM_DIM),
        train_dl, val_dl, run_out, f"kestrel_{run_tag}",
        epochs=epochs, patience=patience, lr=5e-4,
        device=device, is_kestrel=True, smooth_window=3,
    )

    _, ood_preds, ood_tgts, _ = evaluate(model, ood_dl, device)
    r2 = r2_scores(ood_preds, ood_tgts)

    # clean checkpoint to save disk
    ckpt = os.path.join(run_out, f"kestrel_{run_tag}_best.pt")
    if os.path.exists(ckpt):
        os.remove(ckpt)

    row = {
        "condition":  name,
        "seed":       seed,
        "n_train":    len(train_df),
        "n_kingdoms": n_kingdoms,
    }
    row.update(r2)
    return row


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_confound_control(data_csv: str,
                         out_dir: str = "taxonomy_confound_output/",
                         seeds: list[int] | None = None,
                         epochs: int = 100,
                         patience: int = 15):
    """
    Run all three conditions × seeds and write results.

    Conditions
    ----------
    A  bacteria_only       : only Bacteria sequences
    B  matched_multitaxon  : same n as A, proportionally sampled across all taxa
    C  full_bender         : all non-viral sequences
    """
    seeds = seeds or [42, 67, 93]
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    # ── load dataset ─────────────────────────────────────────────────────────
    df = pd.read_csv(data_csv)
    df = df.dropna(subset=["rg", "ree", "nu", "sequence"])
    for col in GRAPH_TARGETS:
        if col not in df.columns:
            df[col] = np.nan

    ood_df  = df[df["kingdom"] == OOD_KINGDOM].copy().reset_index(drop=True)
    pool_df = df[df["kingdom"] != OOD_KINGDOM].copy().reset_index(drop=True)

    # count bacteria sequences (defines matched n)
    bacteria_df = pool_df[pool_df["kingdom"] == "Bacteria"].copy()
    n_bacteria  = len(bacteria_df)

    print(f"\nDataset summary")
    print(f"  Total sequences      : {len(df):,}")
    print(f"  OOD viral (fixed)    : {len(ood_df):,}")
    print(f"  Non-viral pool       : {len(pool_df):,}")
    print(f"  Bacteria (condition A): {n_bacteria:,}")
    print(f"\nKingdom breakdown (non-viral):")
    for k, cnt in pool_df["kingdom"].value_counts().items():
        print(f"  {k:20s}: {cnt:,}")

    records = []

    for seed in seeds:
        print(f"\n{'='*65}")
        print(f"SEED {seed}")
        print(f"{'='*65}")

        # ── Condition A: Bacteria-only ────────────────────────────────────────
        print(f"\n[A] bacteria_only  (n={n_bacteria:,}, 1 kingdom)")
        train_a, val_a = split_train_val(bacteria_df, val_frac=0.10, seed=seed)
        row = run_condition(
            "A_bacteria_only", train_a, val_a, ood_df,
            seed, out_dir, epochs, patience, device,
        )
        records.append(row)
        _print_key_r2(row)

        # ── Condition B: Size-matched multi-taxon (NEW CONTROL) ───────────────
        print(f"\n[B] matched_multitaxon  (n={n_bacteria:,}, proportional across all taxa)")
        matched_df = sample_proportional(pool_df, n_bacteria, seed)
        print(f"    Kingdom distribution in matched sample:")
        for k, cnt in matched_df["kingdom"].value_counts().items():
            print(f"      {k:20s}: {cnt:,}")
        train_b, val_b = split_train_val(matched_df, val_frac=0.10, seed=seed)
        row = run_condition(
            "B_matched_multitaxon", train_b, val_b, ood_df,
            seed, out_dir, epochs, patience, device,
        )
        records.append(row)
        _print_key_r2(row)

        # ── Condition C: Full BENDER ──────────────────────────────────────────
        n_full = len(pool_df)
        print(f"\n[C] full_bender  (n={n_full:,}, all taxa)")
        train_c, val_c = split_train_val(pool_df, val_frac=0.10, seed=seed)
        row = run_condition(
            "C_full_bender", train_c, val_c, ood_df,
            seed, out_dir, epochs, patience, device,
        )
        records.append(row)
        _print_key_r2(row)

    # ── save raw results ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, "taxonomy_confound_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nRaw results → {out_csv}")

    # ── summary table ─────────────────────────────────────────────────────────
    key_targets = ["rg", "nu", "delta", "a0"]
    print(f"\n{'='*65}")
    print("SUMMARY  (OOD R²,  mean ± std across seeds)")
    print(f"{'='*65}")
    header = f"{'Condition':<25}" + "".join(f"  {t:>8}" for t in key_targets)
    print(header)
    print("─" * len(header))
    for cond, grp in results_df.groupby("condition", sort=False):
        row_str = f"{cond:<25}"
        for t in key_targets:
            m = grp[t].mean()
            s = grp[t].std(ddof=1) if len(grp) > 1 else float("nan")
            row_str += f"  {m:.3f}±{s:.3f}"
        print(row_str)

    # ── ES_OOD proxy using full BENDER as reference ───────────────────────────
    ref = results_df[results_df["condition"] == "C_full_bender"].groupby("condition")
    ref_mean = results_df[results_df["condition"] == "C_full_bender"][key_targets].mean()

    print(f"\n{'='*65}")
    print("ES_OOD PROXY  (R²_condition / R²_full_BENDER, mean across seeds)")
    print("  Interpretation: 1.0 = matches full BENDER; <1.0 = below ceiling")
    print(f"{'='*65}")
    print(f"{'Condition':<25}" + "".join(f"  {t:>8}" for t in key_targets))
    print("─" * len(header))
    for cond, grp in results_df.groupby("condition", sort=False):
        row_str = f"{cond:<25}"
        for t in key_targets:
            ratio = grp[t].mean() / (ref_mean[t] + 1e-10)
            row_str += f"  {ratio:>8.3f}"
        print(row_str)

    print(f"\n{'='*65}")
    print("INTERPRETATION GUIDE")
    print("  B ≈ C >> A  →  taxonomic diversity drives the gain (not sample size)")
    print("  B ≈ A << C  →  sample size drives the gain (not diversity)")
    print("  A < B < C   →  both contribute independently")
    print(f"{'='*65}")

    return results_df


def _print_key_r2(row: dict):
    for t in ["rg", "nu", "delta", "a0"]:
        if t in row:
            print(f"    OOD R²({t:5s}) = {row[t]:.4f}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Taxonomy confound control for BENDER bacteria ablation")
    p.add_argument("--data_csv",  required=True,
                   help="Path to BENDER CSV (with 'kingdom' and 'sequence' columns)")
    p.add_argument("--out_dir",   default="taxonomy_confound_output/",
                   help="Output directory for checkpoints and results")
    p.add_argument("--seeds",     nargs="+", type=int, default=[42, 67, 93])
    p.add_argument("--epochs",    type=int,  default=100)
    p.add_argument("--patience",  type=int,  default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_confound_control(
        data_csv=args.data_csv,
        out_dir=args.out_dir,
        seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
    )
