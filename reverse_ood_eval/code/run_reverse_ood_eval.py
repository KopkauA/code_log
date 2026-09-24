"""
run_reverse_ood_eval.py

Loads the already-trained KESTREL checkpoints and the already-trained
IDP-ESM2-8M GeoHead checkpoints -- BOTH trained on the 5 non-viral kingdoms
(Bacteria, Plants, Fungi, Mammals, Protists) with "Viruses" held out
entirely as an OOD test set, using the same cluster-aware 80/10/10 split
logic in both kestrel_ood_virus.py and idp_esm2_virus_fixed.py -- then runs
inference with each on the 5 non-viral taxa and reports R^2 per taxon per
model.

Requires kestrel_ood_virus.py and idp_esm2_virus_fixed.py to be importable
(place them in the same folder as this script, or point --code_dir at
wherever they live).

--- Leakage handling, read before trusting the numbers ---

Since BOTH models were trained on ~90% of the non-viral data (80% train +
10% val, cluster-aware split), by default this script follows the literal
instruction "run inference on the rest of the 5 taxa" and evaluates BOTH
models on the FULL non-viral set. This means BOTH models' R^2 numbers here
include data they've already seen during training, which will inflate
both relative to a true held-out test -- this is a fair, apples-to-apples
comparison between the two models, just not a genuine generalization test
for either.

Pass --test_split_only to instead evaluate BOTH models only on their
original held-out test split (reconstructed via the same seed + clustering
logic used in each training script) -- a cleaner, leak-free comparison, at
the cost of a smaller sample size per kingdom and not matching the literal
"all 5 taxa" instruction.

Usage:
    python run_reverse_ood_eval.py \
        --bender ./data/merged.csv \
        --code_dir ./scripts \
        --kestrel_ckpt_dir ./checkpoints \
        --esm2_ckpt_dir ./checkpoints \
        --seeds 42 67 93 \
        --out ./results/reverse_ood
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bender", required=True, help="Path to merged.csv (full BENDER dataset)")
    p.add_argument("--code_dir", required=True,
                    help="Folder containing kestrel_ood_virus.py and idp_esm2_virus_fixed.py")
    p.add_argument("--kestrel_ckpt_dir", required=True, help="Folder containing kestrel_best<seed>.pt files")
    p.add_argument("--esm2_ckpt_dir", required=True,
                    help="Folder containing IDP-ESM2-8M_geohead_best<seed>.pt files")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 67, 93])
    p.add_argument("--out", required=True, help="Output directory for result CSVs")
    p.add_argument("--test_split_only", action="store_true",
                    help="Evaluate BOTH models only on their held-out test split (avoids data "
                         "leakage, since both were trained on the non-viral kingdoms) instead of "
                         "the full non-viral set. Off by default, to match the literal instruction "
                         "'run inference on all 5 taxa'.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(device_str):
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# KESTREL evaluation
# ---------------------------------------------------------------------------

def eval_kestrel_all_seeds(df, code_dir, ckpt_dir, seeds, device, test_split_only):
    from kestrel_ood_virus import (
        KESTREL, IDPDataset, make_splits, build_kingdom_vocab,
        ALL_TARGETS, r2_by_kingdom,
    )

    rows = []
    for seed in seeds:
        print(f"\n=== KESTREL seed {seed} ===")
        ckpt_path = os.path.join(ckpt_dir, f"kestrel_best{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[warn] checkpoint not found: {ckpt_path}, skipping")
            continue

        # Reconstruct the EXACT split used during training (deterministic given the seed)
        splits = make_splits(df, seed=seed)
        kingdom_vocab = build_kingdom_vocab(splits["train"])
        stats = {
            "mean": splits["train"][ALL_TARGETS].mean(),
            "std": splits["train"][ALL_TARGETS].std().clip(lower=1e-6),
        }

        eval_df = splits["test"].reset_index(drop=True) if test_split_only \
            else df[df["kingdom"] != "Viruses"].reset_index(drop=True)
        print(f"[info] evaluating on {len(eval_df)} sequences "
              f"({'held-out test split only' if test_split_only else 'full non-viral set'})")

        model = KESTREL(n_kingdoms=len(kingdom_vocab))
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        ds = IDPDataset(eval_df, max_len=256, target_stats=stats, kingdom_vocab=kingdom_vocab)
        dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

        all_preds, all_targets, all_kingdoms = [], [], []
        with torch.no_grad():
            for b in dl:
                preds = model(b["sequence"].to(device), b["mask"].to(device), b["kingdom_idx"].to(device))
                all_preds.append(preds.cpu())
                all_targets.append(b["targets"])
                all_kingdoms.extend(b["kingdom"])
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        kr = r2_by_kingdom(all_preds, all_targets, all_kingdoms)
        print(kr)

        kr = kr.reset_index()
        kr["model"] = "KESTREL"
        kr["seed"] = seed
        rows.append(kr)

    if not rows:
        print("[warn] no KESTREL seeds evaluated")
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# IDP-ESM2-8M (viral geohead) evaluation
# ---------------------------------------------------------------------------

def r2_scores_generic(preds, targets, target_names):
    out = {}
    for i, name in enumerate(target_names):
        y, yh = targets[:, i].numpy(), preds[:, i].numpy()
        ss_res = ((y - yh) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum() + 1e-10
        out[name] = float(1 - ss_res / ss_tot)
    return out


def r2_by_kingdom_generic(preds, targets, kingdoms, target_names):
    kingdoms = np.array(kingdoms)
    rows = []
    for k in sorted(set(kingdoms)):
        idx = np.where(kingdoms == k)[0]
        r2 = r2_scores_generic(preds[idx], targets[idx], target_names)
        r2["kingdom"] = k
        r2["n"] = len(idx)
        rows.append(r2)
    return pd.DataFrame(rows).set_index("kingdom")


def eval_esm2_all_seeds(df, code_dir, ckpt_dir, seeds, device, test_split_only):
    """
    IDP-ESM2-8M's GeoHead (per idp_esm2_virus_fixed.py / run_idp_esm2_on_bender.py)
    was trained on the 5 non-viral kingdoms with the SAME cluster-aware
    80/10/10 split logic as KESTREL, virus held out entirely as OOD -- the
    exact same leakage situation as KESTREL.
    """
    from idp_esm2_virus_fixed import (
        GeoHead, load_esm2, extract_embeddings, BENDER_TARGETS,
        MAX_SEQ_LEN, make_splits,
    )

    # Match idp_esm2_virus_fixed.py's own filtering exactly: drop missing
    # targets/sequence, then apply the same max-length cutoff it uses.
    filtered_df = df.dropna(subset=["sequence"] + BENDER_TARGETS)
    filtered_df = filtered_df[filtered_df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    nonviral_df = filtered_df[filtered_df["kingdom"] != "Viruses"].reset_index(drop=True)

    # embeddings only depend on the frozen ESM2 backbone, not the seed -- compute once
    tokenizer, esm_model, hidden_dim = load_esm2("8M", device)

    rows = []
    for seed in seeds:
        print(f"\n=== IDP-ESM2-8M seed {seed} ===")
        ckpt_path = os.path.join(ckpt_dir, f"IDP-ESM2-8M_geohead_best{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[warn] checkpoint not found: {ckpt_path}, skipping")
            continue

        if test_split_only:
            # reconstruct the exact split this seed's checkpoint was trained/validated with
            _, _, eval_df = make_splits(nonviral_df, seed=seed)
        else:
            eval_df = nonviral_df
        eval_df = eval_df.reset_index(drop=True)
        print(f"[info] evaluating on {len(eval_df)} sequences "
              f"({'held-out test split only' if test_split_only else 'full non-viral set'})")

        embeddings = extract_embeddings(eval_df["sequence"].tolist(), tokenizer, esm_model, device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = GeoHead(hidden_dim).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        t_mean = torch.tensor(ckpt["t_mean"], dtype=torch.float32)
        t_std = torch.tensor(ckpt["t_std"], dtype=torch.float32)

        preds = []
        with torch.no_grad():
            for i in range(0, len(embeddings), 128):
                batch = embeddings[i:i + 128].to(device)
                pred = model(batch).cpu()
                preds.append(pred * t_std + t_mean)
        preds = torch.cat(preds)

        targets = torch.tensor(eval_df[BENDER_TARGETS].values.astype(float), dtype=torch.float32)
        kr = r2_by_kingdom_generic(preds, targets, eval_df["kingdom"].tolist(), BENDER_TARGETS)
        print(kr)

        kr = kr.reset_index()
        kr["model"] = "IDP-ESM2-8M"
        kr["seed"] = seed
        rows.append(kr)

    esm_model.cpu()
    torch.cuda.empty_cache()

    if not rows:
        print("[warn] no IDP-ESM2-8M seeds evaluated")
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.code_dir)

    device = get_device(args.device)
    print(f"[info] using device: {device}")

    df = pd.read_csv(args.bender)
    print(f"[info] loaded {len(df)} rows from {args.bender}")

    kestrel_results = eval_kestrel_all_seeds(
        df, args.code_dir, args.kestrel_ckpt_dir, args.seeds, device, args.test_split_only)
    if not kestrel_results.empty:
        kestrel_out = os.path.join(args.out, "kestrel_reverse_ood_by_kingdom.csv")
        kestrel_results.to_csv(kestrel_out, index=False)
        print(f"[info] wrote {kestrel_out}")

    esm2_results = eval_esm2_all_seeds(
        df, args.code_dir, args.esm2_ckpt_dir, args.seeds, device, args.test_split_only)
    if not esm2_results.empty:
        esm2_out = os.path.join(args.out, "idp_esm2_8m_reverse_ood_by_kingdom.csv")
        esm2_results.to_csv(esm2_out, index=False)
        print(f"[info] wrote {esm2_out}")

    # combined long-format summary: model, seed, kingdom, target, r2, n
    combined_rows = []
    for results_df, target_names in [
        (kestrel_results, None),  # KESTREL's columns already include all its targets
        (esm2_results, None),
    ]:
        if results_df.empty:
            continue
        id_cols = ["model", "seed", "kingdom", "n"]
        target_cols = [c for c in results_df.columns if c not in id_cols]
        for _, row in results_df.iterrows():
            for tgt in target_cols:
                combined_rows.append({
                    "model": row["model"], "seed": row["seed"],
                    "kingdom": row["kingdom"], "n": row["n"],
                    "target": tgt, "r2": row[tgt],
                })

    if combined_rows:
        combined_df = pd.DataFrame(combined_rows)
        combined_out = os.path.join(args.out, "combined_r2_long.csv")
        combined_df.to_csv(combined_out, index=False)
        print(f"[info] wrote combined long-format results to {combined_out}")

        # also print a seed-averaged summary for a quick read
        summary = combined_df.groupby(["model", "kingdom", "target"])["r2"].agg(["mean", "std"]).reset_index()
        summary_out = os.path.join(args.out, "combined_r2_seed_averaged.csv")
        summary.to_csv(summary_out, index=False)
        print(f"[info] wrote seed-averaged summary to {summary_out}")
        print("\n=== Seed-averaged R^2 summary ===")
        print(summary.to_string())


if __name__ == "__main__":
    main()
