"""
run_multitemp_ood_eval.py

Both KESTREL and IDP-ESM2-8M's GeoHead were trained on the 5 non-viral
kingdoms, with "Viruses" held out entirely as a genuine OOD test set, using
CALVADOS-2 labels computed at 300K (confirmed identical to merged.csv's own
rg/ree/nu/delta/a0/graph-metric columns for viral sequences).

This script re-uses each model's already-trained checkpoint to make ONE set
of predictions per seed on the 1,025 viral OOD sequences (the model/input
never changes across temperatures -- only the labels do), then compares
those same fixed predictions against CALVADOS-2 ground truth computed at
5 different simulation temperatures: 278K, 288K, 300K, 310K, 320K.

Unlike run_reverse_ood_eval.py, there is NO leakage concern here: virus was
never part of either model's training data at any point, so every one of
these 5 temperature evaluations is a genuine out-of-distribution test.

Requires kestrel_ood_virus.py and idp_esm2_virus_fixed.py to be importable
(point --code_dir at wherever they live).

Usage:
    python run_multitemp_ood_eval.py \
        --bender ./merged.csv \
        --temp_csvs 278:bender_278K.csv 288:bender_288K.csv 300:bender_300K.csv \
                    310:bender_310K.csv 320:bender_320K.csv \
        --code_dir ./scripts \
        --kestrel_ckpt_dir ./checkpoints \
        --esm2_ckpt_dir ./checkpoints \
        --seeds 42 67 93 \
        --out ./results/multitemp_ood
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# CALVADOS-2 temperature files use "A0" (uppercase); everything else in
# this codebase uses "a0" (lowercase) -- normalize on load.
TEMP_COL_RENAME = {"A0": "a0"}

# Set from --bender in main(); used inside eval_kestrel_multitemp to
# reconstruct each seed's original kingdom vocabulary.
FULL_BENDER_PATH = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bender", required=True,
                    help="Path to merged.csv -- used to supply the 'sequence' column (joined on "
                         "protein_name == UniProt_ID) and to reconstruct KESTREL's kingdom vocab")
    p.add_argument("--temp_csvs", nargs="+", required=True,
                    help="List of TEMP:path pairs, e.g. 278:bender_278K.csv 300:bender_300K.csv ...")
    p.add_argument("--code_dir", required=True,
                    help="Folder containing kestrel_ood_virus.py and idp_esm2_virus_fixed.py")
    p.add_argument("--kestrel_ckpt_dir", required=True, help="Folder containing kestrel_best<seed>.pt files")
    p.add_argument("--esm2_ckpt_dir", required=True,
                    help="Folder containing IDP-ESM2-8M_geohead_best_seed<seed>.pt files")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 67, 93])
    p.add_argument("--out", required=True, help="Output directory for result CSVs")
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_device(device_str):
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_temp_csvs(temp_csv_args):
    """Turns ['278:path1', '300:path2', ...] into {278: 'path1', 300: 'path2', ...}"""
    out = {}
    for item in temp_csv_args:
        temp_str, path = item.split(":", 1)
        out[int(temp_str)] = path
    return out


def load_viral_labels_by_temp(temp_csv_map, id_to_seq):
    """
    For each temperature, load its CSV, keep only viral rows, rename A0->a0,
    attach the sequence column via the UniProt_ID -> sequence mapping, and
    keep only rows where a sequence was actually found.
    """
    labels_by_temp = {}
    for temp, path in sorted(temp_csv_map.items()):
        df = pd.read_csv(path)
        df = df.rename(columns=TEMP_COL_RENAME)
        df = df[df["kingdom"] == "Viruses"].reset_index(drop=True)
        df["sequence"] = df["protein_name"].map(id_to_seq)
        n_before = len(df)
        df = df.dropna(subset=["sequence"]).reset_index(drop=True)
        if len(df) < n_before:
            print(f"[warn] {temp}K: {n_before - len(df)} viral rows had no matching sequence, dropped")
        print(f"[info] {temp}K: {len(df)} viral sequences with labels + sequence")
        labels_by_temp[temp] = df
    return labels_by_temp


def r2_scores_generic(preds, targets, target_names):
    out = {}
    for i, name in enumerate(target_names):
        y, yh = targets[:, i].numpy(), preds[:, i].numpy()
        ss_res = ((y - yh) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum() + 1e-10
        out[name] = float(1 - ss_res / ss_tot)
    return out


# ---------------------------------------------------------------------------
# KESTREL: predict once per seed, evaluate against every temperature's labels
# ---------------------------------------------------------------------------

def eval_kestrel_multitemp(labels_by_temp, code_dir, ckpt_dir, seeds, device):
    from kestrel_ood_virus.code.kestrel_ood_virus import KESTREL, IDPDataset, make_splits, build_kingdom_vocab, ALL_TARGETS

    rows = []
    for seed in seeds:
        print(f"\n=== KESTREL seed {seed} ===")
        ckpt_path = os.path.join(ckpt_dir, f"kestrel_best{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[warn] checkpoint not found: {ckpt_path}, skipping")
            continue

        # Predictions are identical across all 5 temperatures (same sequences,
        # same frozen model) -- compute once, reuse for every temperature.
        picked_temp = sorted(labels_by_temp.keys())[0]
        eval_df = labels_by_temp[picked_temp].copy()

        # IDPDataset needs ALL_TARGETS columns to exist, but we only ever use
        # its "sequence"/"mask"/"kingdom_idx" output below -- we compare raw
        # predictions against each temperature's own real values separately.
        for t in ALL_TARGETS:
            if t not in eval_df.columns:
                eval_df[t] = 0.0

        # kingdom_vocab / n_kingdoms is a property of TRAINING (the non-viral
        # split), not of the viral eval data -- reconstruct it exactly as
        # during training so the checkpoint's kingdom-embedding size matches.
        full_df = pd.read_csv(FULL_BENDER_PATH)
        splits = make_splits(full_df, seed=seed)
        kingdom_vocab = build_kingdom_vocab(splits["train"])
        train_mean = splits["train"][ALL_TARGETS].mean()
        train_std = splits["train"][ALL_TARGETS].std().clip(lower=1e-6)

        model = KESTREL(n_kingdoms=len(kingdom_vocab))
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        stats = {"mean": train_mean, "std": train_std}
        ds = IDPDataset(eval_df, max_len=256, target_stats=stats, kingdom_vocab=kingdom_vocab)
        dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

        all_preds_normed = []
        with torch.no_grad():
            for b in dl:
                preds = model(b["sequence"].to(device), b["mask"].to(device), b["kingdom_idx"].to(device))
                all_preds_normed.append(preds.cpu())
        all_preds_normed = torch.cat(all_preds_normed)

        # denormalize back to physical (raw) scale using TRAIN stats, so
        # predictions are comparable to each temperature's real physical labels
        preds_raw = all_preds_normed * torch.tensor(train_std.values, dtype=torch.float32) \
            + torch.tensor(train_mean.values, dtype=torch.float32)

        # now compare this ONE set of raw predictions against each temperature's real labels
        for temp, temp_df in sorted(labels_by_temp.items()):
            targets = torch.tensor(temp_df[ALL_TARGETS].values.astype(float), dtype=torch.float32)
            r2 = r2_scores_generic(preds_raw, targets, ALL_TARGETS)
            r2["model"] = "KESTREL"
            r2["seed"] = seed
            r2["temperature_K"] = temp
            r2["n"] = len(temp_df)
            rows.append(r2)
            print(f"  T={temp}K: " + ", ".join(f"{k}={v:.4f}" for k, v in r2.items() if k in ALL_TARGETS))

    if not rows:
        print("[warn] no KESTREL seeds evaluated")
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# IDP-ESM2-8M: predict once per seed, evaluate against every temperature's labels
# ---------------------------------------------------------------------------

def eval_esm2_multitemp(labels_by_temp, code_dir, ckpt_dir, seeds, device):
    from idp_esm2_virus_fixed import GeoHead, load_esm2, extract_embeddings, BENDER_TARGETS

    picked_temp = sorted(labels_by_temp.keys())[0]
    eval_df = labels_by_temp[picked_temp].copy().reset_index(drop=True)

    tokenizer, esm_model, hidden_dim = load_esm2("8M", device)
    embeddings = extract_embeddings(eval_df["sequence"].tolist(), tokenizer, esm_model, device)

    rows = []
    for seed in seeds:
        print(f"\n=== IDP-ESM2-8M seed {seed} ===")
        ckpt_path = os.path.join(ckpt_dir, f"IDP-ESM2-8M_geohead_best_seed{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[warn] checkpoint not found: {ckpt_path}, skipping")
            continue

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
                preds.append(pred * t_std + t_mean)  # denormalize to raw physical scale
        preds_raw = torch.cat(preds)

        for temp, temp_df in sorted(labels_by_temp.items()):
            targets = torch.tensor(temp_df[BENDER_TARGETS].values.astype(float), dtype=torch.float32)
            r2 = r2_scores_generic(preds_raw, targets, BENDER_TARGETS)
            r2["model"] = "IDP-ESM2-8M"
            r2["seed"] = seed
            r2["temperature_K"] = temp
            r2["n"] = len(temp_df)
            rows.append(r2)
            print(f"  T={temp}K: " + ", ".join(f"{k}={v:.4f}" for k, v in r2.items() if k in BENDER_TARGETS))

    esm_model.cpu()
    torch.cuda.empty_cache()

    if not rows:
        print("[warn] no IDP-ESM2-8M seeds evaluated")
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global FULL_BENDER_PATH
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.code_dir)
    FULL_BENDER_PATH = args.bender

    device = get_device(args.device)
    print(f"[info] using device: {device}")

    bender_df = pd.read_csv(args.bender)
    id_to_seq = dict(zip(bender_df["UniProt_ID"], bender_df["sequence"]))
    print(f"[info] loaded {len(bender_df)} rows from {args.bender} for sequence lookup")

    temp_csv_map = parse_temp_csvs(args.temp_csvs)
    labels_by_temp = load_viral_labels_by_temp(temp_csv_map, id_to_seq)

    kestrel_results = eval_kestrel_multitemp(labels_by_temp, args.code_dir, args.kestrel_ckpt_dir, args.seeds, device)
    if not kestrel_results.empty:
        out_path = os.path.join(args.out, "kestrel_multitemp_r2.csv")
        kestrel_results.to_csv(out_path, index=False)
        print(f"[info] wrote {out_path}")

    esm2_results = eval_esm2_multitemp(labels_by_temp, args.code_dir, args.esm2_ckpt_dir, args.seeds, device)
    if not esm2_results.empty:
        out_path = os.path.join(args.out, "idp_esm2_8m_multitemp_r2.csv")
        esm2_results.to_csv(out_path, index=False)
        print(f"[info] wrote {out_path}")

    # combined long-format + seed-averaged summary
    combined_rows = []
    for results_df in [kestrel_results, esm2_results]:
        if results_df.empty:
            continue
        id_cols = ["model", "seed", "temperature_K", "n"]
        target_cols = [c for c in results_df.columns if c not in id_cols]
        for _, row in results_df.iterrows():
            for tgt in target_cols:
                combined_rows.append({
                    "model": row["model"], "seed": row["seed"],
                    "temperature_K": row["temperature_K"], "n": row["n"],
                    "target": tgt, "r2": row[tgt],
                })

    if combined_rows:
        combined_df = pd.DataFrame(combined_rows)
        combined_out = os.path.join(args.out, "combined_multitemp_r2_long.csv")
        combined_df.to_csv(combined_out, index=False)
        print(f"[info] wrote {combined_out}")

        summary = combined_df.groupby(["model", "temperature_K", "target"])["r2"].agg(["mean", "std"]).reset_index()
        summary_out = os.path.join(args.out, "combined_multitemp_r2_seed_averaged.csv")
        summary.to_csv(summary_out, index=False)
        print(f"[info] wrote {summary_out}")
        print("\n=== Seed-averaged R^2 by temperature ===")
        print(summary.to_string())


if __name__ == "__main__":
    main()
