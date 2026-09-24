"""
aggregate_seeds.py
Collects summary_r2.csv from each seed run; prints and saves mean ± std R².

Usage:
    python aggregate_seeds.py --out_root output/ --seeds 42 123 456 789 1337
"""

import argparse
import os
import pandas as pd


def main(out_root, seeds):
    frames = []
    missing = []
    for seed in seeds:
        fp = os.path.join(out_root, f"seed_{seed}", "summary_r2.csv")
        if not os.path.exists(fp):
            missing.append(seed)
            continue
        df = pd.read_csv(fp, index_col="target")
        df["seed"] = seed
        frames.append(df)

    if missing:
        print(f"WARNING: missing results for seeds: {missing}")
    if not frames:
        print("No results found — exiting.")
        return

    combined = pd.concat(frames)

    for split in ("test_r2", "ood_r2"):
        if split not in combined.columns:
            continue
        pivot = combined.pivot_table(
            values=split, index="target", aggfunc=["mean", "std"])
        pivot.columns = ["mean", "std"]
        pivot = pivot.sort_index()

        print(f"\n{'='*52}")
        print(f"{split.upper()}  (n={len(frames)} seeds)")
        print(f"{'='*52}")
        print(f"{'Target':30s}  {'Mean':>7}  {'Std':>7}")
        print("─" * 46)
        for tgt, row in pivot.iterrows():
            print(f"{tgt:30s}  {row['mean']:>7.4f}  {row['std']:>7.4f}")

        out_path = os.path.join(out_root, f"aggregate_{split}.csv")
        pivot.to_csv(out_path)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    a = p.parse_args()
    main(a.out_root, a.seeds)
