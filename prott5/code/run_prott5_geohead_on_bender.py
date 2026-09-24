"""
run_prott5_geohead_on_bender.py

Trains and evaluates a GeoHead MLP on frozen ProtT5-XL embeddings,
predicting all 5 BENDER geometric targets including a0.

HPC workflow — embeddings extracted once, 3 seeds trained in parallel:
    python run_prott5_geohead_on_bender.py --extract-only --emb-cache <dir>
    python run_prott5_geohead_on_bender.py --seeds 42  --emb-cache <dir>
    python run_prott5_geohead_on_bender.py --seeds 67  --emb-cache <dir>
    python run_prott5_geohead_on_bender.py --seeds 93  --emb-cache <dir>
    python run_prott5_geohead_on_bender.py --aggregate-only

Local single-run (all seeds sequentially):
    python run_prott5_geohead_on_bender.py

Dependencies:
    pip install transformers sentencepiece torch pandas numpy tqdm

Outputs (per seed, in --out):
    prott5xl_seed<seed>_results.csv                 R2 summary (bender_test, ood_viruses)
    prott5xl_seed<seed>_bender_test_predictions.csv  per-sequence true/pred/residual, test split
    prott5xl_seed<seed>_ood_viruses_predictions.csv  per-sequence true/pred/residual, OOD viruses
"""

import os
import re
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# All 5 BENDER geometric targets (a0 is BENDER-only, absent from IDRome)
BENDER_TARGETS = ["rg", "ree", "nu", "delta", "a0"]

PROTT5_HF_ID  = "Rostlab/prot_t5_xl_uniref50"
PROTT5_HIDDEN = 1024

MAX_SEQ_LEN   = 256
BATCH_SIZE    = 16
EMBED_BATCH   = 8
HEAD_EPOCHS   = 200
HEAD_PATIENCE = 20
HEAD_LR       = 3e-3
HEAD_DROPOUT  = 0.1

DEFAULT_SEEDS = [42, 67, 93]

# --- Gaivi paths (edited from the original macOS /Volumes/... paths) ---
_BENDER_CSV = "./merged.csv"
_OUT_DIR    = "./results/prott5/"
_EMB_CACHE  = "./prott5_emb_cache/"

# ─────────────────────────────────────────────
# GeoHead MLP  (identical to GeoGraph's FeaturesHead)
# ─────────────────────────────────────────────

class GeoHead(nn.Module):
    """
    Shallow MLP prediction head.
    Input:  mean-pooled ProtT5-XL embeddings (1024,)
    Output: len(BENDER_TARGETS) = 5 geometric properties
    """
    def __init__(self, hidden_dim=PROTT5_HIDDEN, n_out=len(BENDER_TARGETS),
                 dropout=HEAD_DROPOUT):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_out),
        )

    def forward(self, x):
        return self.head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_bender(path, max_seq_len=MAX_SEQ_LEN):
    df = pd.read_csv(path)
    # require all 5 targets; rows missing a0 are dropped
    df = df.dropna(subset=["sequence"] + BENDER_TARGETS)
    n_before = len(df)
    df = df[df["sequence"].str.len() <= max_seq_len].reset_index(drop=True)
    print(f"BENDER: {n_before} → {len(df)} sequences (≤{max_seq_len} aa, all 5 targets present)")
    return df


def make_splits(df, train_frac=0.80, val_frac=0.10, seed=42):
    """Cluster-aware split if cluster_id present, else random."""
    if "cluster_id" in df.columns:
        rng      = np.random.default_rng(seed)
        clusters = df["cluster_id"].unique()
        rng.shuffle(clusters)
        n  = len(clusters)
        n1 = int(train_frac * n)
        n2 = int(val_frac * n)
        tc = set(clusters[:n1])
        vc = set(clusters[n1:n1+n2])
        ec = set(clusters[n1+n2:])
        train = df[df["cluster_id"].isin(tc)].reset_index(drop=True)
        val   = df[df["cluster_id"].isin(vc)].reset_index(drop=True)
        test  = df[df["cluster_id"].isin(ec)].reset_index(drop=True)
        print(f"Cluster-aware split: Train:{len(train)} Val:{len(val)} Test:{len(test)}")
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n   = len(idx)
        n1  = int(train_frac * n)
        n2  = int(val_frac * n)
        train = df.iloc[idx[:n1]].reset_index(drop=True)
        val   = df.iloc[idx[n1:n1+n2]].reset_index(drop=True)
        test  = df.iloc[idx[n1+n2:]].reset_index(drop=True)
        print(f"Random split: Train:{len(train)} Val:{len(val)} Test:{len(test)}")
    return train, val, test

# ─────────────────────────────────────────────
# PROTT5 HELPERS
# ─────────────────────────────────────────────

def prep_seq(seq):
    """Space-separate residues and replace non-standard AAs for ProtT5."""
    seq = re.sub(r"[UZOB]", "X", seq.upper())
    return " ".join(list(seq))


def load_prott5(device):
    print(f"\nLoading ProtT5-XL from {PROTT5_HF_ID} …")
    tokenizer = T5Tokenizer.from_pretrained(PROTT5_HF_ID, do_lower_case=False)
    model     = T5EncoderModel.from_pretrained(PROTT5_HF_ID)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in model.parameters())
    print(f"  Hidden dim: {PROTT5_HIDDEN}  Params: {n:,}")
    return tokenizer, model


@torch.no_grad()
def extract_embeddings(sequences, tokenizer, prott5, device,
                       batch_size=EMBED_BATCH, max_len=MAX_SEQ_LEN):
    """
    Mean-pool ProtT5-XL encoder hidden states over non-padding positions.
    Each residue maps to one token; EOS is included (one token out of L+1,
    standard practice for ProtT5 embeddings).
    """
    all_embs = []
    for i in tqdm(range(0, len(sequences), batch_size),
                  desc="Extracting embeddings"):
        batch  = [prep_seq(s[:max_len]) for s in sequences[i:i+batch_size]]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len + 2,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = prott5(**inputs).last_hidden_state       # (B, L, 1024)

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb  = (hidden * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)                    # (N, 1024)

# ─────────────────────────────────────────────
# EMBEDDING DATASET
# ─────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets_df, target_mean, target_std):
        self.emb     = embeddings
        self.tgt_raw = torch.tensor(
            targets_df[BENDER_TARGETS].values.astype(float), dtype=torch.float32)
        mean     = torch.tensor(target_mean, dtype=torch.float32)
        std      = torch.tensor(target_std,  dtype=torch.float32)
        self.tgt = torch.nan_to_num((self.tgt_raw - mean) / std, nan=0.0)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return {"emb": self.emb[idx],
                "targets":     self.tgt[idx],
                "targets_raw": self.tgt_raw[idx]}

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_geohead(train_emb, val_emb, train_df, val_df,
                  out_dir, seed,
                  epochs=HEAD_EPOCHS, patience=HEAD_PATIENCE,
                  lr=HEAD_LR, device=None):
    """Train GeoHead MLP on frozen ProtT5-XL embeddings."""
    device = device or torch.device("cpu")
    torch.manual_seed(seed)

    t_mean = train_df[BENDER_TARGETS].mean().values
    t_std  = train_df[BENDER_TARGETS].std().clip(lower=1e-6).values

    train_ds = EmbeddingDataset(train_emb, train_df, t_mean, t_std)
    val_ds   = EmbeddingDataset(val_emb,   val_df,   t_mean, t_std)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = GeoHead().to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    best  = float("inf")
    wait  = 0
    ckpt  = os.path.join(out_dir, f"prott5xl_geohead_best_seed{seed}.pt")
    val_window = deque(maxlen=3)

    print(f"\nTraining GeoHead for ProtT5-XL seed={seed} ({model.n_params():,} params)")
    print(f"{'Ep':>4} {'Trn':>8} {'Val':>8} {'nu_R2':>8}")
    print("-" * 36)

    for ep in range(epochs):
        model.train()
        tl = 0
        for b in train_dl:
            pred = model(b["emb"].to(device))
            loss = F.mse_loss(pred, b["targets"].to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        tl /= len(train_dl)

        model.eval()
        vl, vp, vt_raw = 0, [], []
        with torch.no_grad():
            for b in val_dl:
                pred = model(b["emb"].to(device))
                vl  += F.mse_loss(pred, b["targets"].to(device)).item()
                p_raw = pred.cpu() * torch.tensor(t_std) + torch.tensor(t_mean)
                vp.append(p_raw)
                vt_raw.append(b["targets_raw"])
        vl /= len(val_dl)
        vp  = torch.cat(vp); vt_raw = torch.cat(vt_raw)

        nu_idx = BENDER_TARGETS.index("nu")
        y, yh  = vt_raw[:, nu_idx].numpy(), vp[:, nu_idx].numpy()
        nu_r2  = 1 - ((y - yh) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-10)

        sched.step()
        print(f"{ep+1:>4} {tl:>8.4f} {vl:>8.4f} {nu_r2:>8.3f}")

        val_window.append(vl)
        smoothed = sum(val_window) / len(val_window)
        if smoothed < best:
            best = smoothed; wait = 0
            torch.save({"state_dict": model.state_dict(),
                        "t_mean": t_mean, "t_std": t_std}, ckpt)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop @ ep {ep+1}"); break

    ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["state_dict"])
    return model, ckpt_data["t_mean"], ckpt_data["t_std"]


def r2_scores(preds, targets, target_names):
    out = {}
    for i, name in enumerate(target_names):
        y, yh  = targets[:, i].numpy(), preds[:, i].numpy()
        ss_res = ((y - yh) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum() + 1e-10
        out[name] = float(1 - ss_res / ss_tot)
    return out


@torch.no_grad()
def evaluate_on_df(model, embeddings, df, t_mean, t_std, split_name, device,
                   pred_csv_path=None):
    model.eval()
    t_mean_t = torch.tensor(t_mean, dtype=torch.float32)
    t_std_t  = torch.tensor(t_std,  dtype=torch.float32)

    preds = []
    for i in range(0, len(embeddings), BATCH_SIZE):
        emb  = embeddings[i:i + BATCH_SIZE].to(device)
        pred = model(emb).cpu()
        preds.append(pred * t_std_t + t_mean_t)
    preds = torch.cat(preds)   # (N, 5)

    # evaluate only on targets present in this split's dataframe
    available = [t for t in BENDER_TARGETS if t in df.columns]
    avail_idx = [BENDER_TARGETS.index(t) for t in available]
    targets   = torch.tensor(df[available].values.astype(float),
                             dtype=torch.float32)
    r2 = r2_scores(preds[:, avail_idx], targets, available)

    print(f"\n-- {split_name} ({len(df)} sequences) ----------------")
    print(f"  {'Target':<12} {'R2':>6}")
    print(f"  {'--':<12} {'--':>6}")
    for tgt, score in r2.items():
        print(f"  {tgt:<12} {score:>6.4f}")

    # ── per-sequence predictions ────────────────────────────────────
    id_cols = [c for c in ["id", "seq_id", "name", "uniprot_id"]
               if c in df.columns]
    out = pd.DataFrame(index=df.index)
    for c in id_cols:
        out[c] = df[c].values
    if "sequence" in df.columns:
        out["sequence"] = df["sequence"].values
    if "kingdom" in df.columns:
        out["kingdom"] = df["kingdom"].values

    for j, tgt in enumerate(BENDER_TARGETS):
        pred_col = preds[:, j].numpy()
        out[f"{tgt}_pred"] = pred_col
        if tgt in df.columns:
            true_col = df[tgt].values.astype(float)
            out[f"{tgt}_true"] = true_col
            out[f"{tgt}_resid"] = true_col - pred_col

    if pred_csv_path:
        out.to_csv(pred_csv_path, index=False)
        print(f"  Saved per-sequence predictions → {pred_csv_path}")

    return r2

# ─────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────

def aggregate_results(out_dir, seeds=DEFAULT_SEEDS):
    """Read per-seed result CSVs from disk and print mean ± std."""
    print(f"\n{'='*70}")
    print("AGGREGATING results across seeds …")
    rows = []
    for seed in seeds:
        csv_path = os.path.join(out_dir, f"prott5xl_seed{seed}_results.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found — skipping seed {seed}")
            continue
        df = pd.read_csv(csv_path)
        df["seed"] = seed
        rows.append(df)
    if not rows:
        print("No per-seed result files found."); return

    df_all = pd.concat(rows, ignore_index=True)
    df_all.to_csv(os.path.join(out_dir, "all_seeds_results.csv"), index=False)

    for split_name in ["bender_test", "ood_viruses"]:
        subset = df_all[df_all["split"] == split_name]
        if subset.empty:
            continue
        print(f"\n  {split_name}")
        print(f"  {'Target':<12} {'mean':>8} {'std':>8}")
        print(f"  {'--':<12} {'--':>8} {'--':>8}")
        for tgt in BENDER_TARGETS:
            if tgt in subset.columns:
                vals = subset[tgt].dropna().values
                if len(vals):
                    print(f"  {tgt:<12} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}")

    agg_cols = [t for t in BENDER_TARGETS if t in df_all.columns]
    agg = df_all.groupby("split")[agg_cols].agg(["mean", "std"])
    agg.to_csv(os.path.join(out_dir, "aggregated_results.csv"))
    print(f"\nSaved aggregated_results.csv → {out_dir}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(bender_csv, out_dir, seeds=DEFAULT_SEEDS, device_str=None,
        emb_cache=None, extract_only=False, aggregate_only=False):

    os.makedirs(out_dir, exist_ok=True)

    if aggregate_only:
        aggregate_results(out_dir, seeds)
        return

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda"); print("Device: CUDA")
    else:
        device = torch.device("cpu");  print("Device: CPU")

    # ── load and split raw data ───────────────────────────────────
    print("\n=== Loading BENDER data ===")
    bender_df      = load_bender(bender_csv)
    bender_ood     = bender_df[bender_df["kingdom"] == "Viruses"].copy().reset_index(drop=True)
    bender_non_ood = bender_df[bender_df["kingdom"] != "Viruses"].copy().reset_index(drop=True)
    # __pos__ lets seed jobs index into the precomputed embedding tensor
    # after make_splits resets the dataframe index
    bender_non_ood["__pos__"] = range(len(bender_non_ood))

    print(f"BENDER non-OOD:       {len(bender_non_ood)} sequences")
    print(f"BENDER OOD (Viruses): {len(bender_ood)} sequences")

    # ── embeddings: load from cache or extract ────────────────────
    if emb_cache:
        non_ood_emb_pt = os.path.join(emb_cache, "non_ood_emb.pt")
        ood_emb_pt     = os.path.join(emb_cache, "ood_emb.pt")
        non_ood_df_csv = os.path.join(emb_cache, "non_ood_df.csv")
        ood_df_csv     = os.path.join(emb_cache, "ood_df.csv")
        cache_ready    = all(os.path.exists(f) for f in
                             [non_ood_emb_pt, ood_emb_pt, non_ood_df_csv, ood_df_csv])
    else:
        cache_ready = False

    if cache_ready:
        print(f"\n=== Loading cached embeddings from {emb_cache} ===")
        all_non_ood_emb = torch.load(non_ood_emb_pt, weights_only=True)
        ood_emb         = torch.load(ood_emb_pt,     weights_only=True)
        bender_non_ood  = pd.read_csv(non_ood_df_csv)
        bender_ood      = pd.read_csv(ood_df_csv)
        print(f"  non-OOD emb: {tuple(all_non_ood_emb.shape)}")
        print(f"  OOD emb:     {tuple(ood_emb.shape)}")
    else:
        print("\n=== Loading ProtT5-XL ===")
        tokenizer, prott5 = load_prott5(device)

        print("\n=== Extracting embeddings ===")
        all_non_ood_emb = extract_embeddings(
            bender_non_ood["sequence"].tolist(), tokenizer, prott5, device)
        ood_emb = extract_embeddings(
            bender_ood["sequence"].tolist(), tokenizer, prott5, device)

        prott5.cpu(); torch.cuda.empty_cache()
        print(f"Backbone offloaded — "
              f"non-OOD {tuple(all_non_ood_emb.shape)}, "
              f"OOD {tuple(ood_emb.shape)}")

        if emb_cache:
            os.makedirs(emb_cache, exist_ok=True)
            print(f"Saving embeddings to {emb_cache} …")
            torch.save(all_non_ood_emb, non_ood_emb_pt)
            torch.save(ood_emb,         ood_emb_pt)
            bender_non_ood.to_csv(non_ood_df_csv, index=False)
            bender_ood.to_csv(ood_df_csv,         index=False)
            print("  Saved.")

    if extract_only:
        print("\n--extract-only: embeddings cached. Exiting.")
        return

    # ── multi-seed training ───────────────────────────────────────
    seed_results = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        train_df, val_df, test_df = make_splits(bender_non_ood, seed=seed)

        train_emb = all_non_ood_emb[train_df["__pos__"].values]
        val_emb   = all_non_ood_emb[val_df["__pos__"].values]
        test_emb  = all_non_ood_emb[test_df["__pos__"].values]

        for df in (train_df, val_df, test_df):
            df.drop(columns=["__pos__"], inplace=True, errors="ignore")

        model, t_mean, t_std = train_geohead(
            train_emb, val_emb, train_df, val_df,
            out_dir, seed=seed, device=device)

        results = {
            "bender_test": evaluate_on_df(
                model, test_emb, test_df, t_mean, t_std,
                f"ProtT5-XL seed={seed} -- BENDER TEST", device,
                pred_csv_path=os.path.join(
                    out_dir, f"prott5xl_seed{seed}_bender_test_predictions.csv")),
            "ood_viruses": evaluate_on_df(
                model, ood_emb, bender_ood, t_mean, t_std,
                f"ProtT5-XL seed={seed} -- OOD Viruses", device,
                pred_csv_path=os.path.join(
                    out_dir, f"prott5xl_seed{seed}_ood_viruses_predictions.csv")),
        }
        seed_results[seed] = results

        rows = [{"seed": seed, "split": split, **r2}
                for split, r2 in results.items()]
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f"prott5xl_seed{seed}_results.csv"),
            index=False)

    if len(seeds) > 1:
        aggregate_results(out_dir, seeds)

    print(f"\nAll done. Results in {out_dir}")

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ProtT5-XL + GeoHead on BENDER — multi-seed evaluation")
    p.add_argument("--bender",
                   default=_BENDER_CSV,
                   help="Path to BENDER merged CSV")
    p.add_argument("--out",
                   default=_OUT_DIR,
                   help="Output directory")
    p.add_argument("--seeds",  nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="Random seeds (default: 42 67 93)")
    p.add_argument("--device", default=None,
                   help="Force device (cuda/cpu)")
    p.add_argument("--emb-cache", default=_EMB_CACHE,
                   help="Dir to save/load precomputed ProtT5 embeddings "
                        "(shared across seeds; safe to reuse across runs)")
    p.add_argument("--extract-only", action="store_true",
                   help="Extract & cache embeddings then exit — no training")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Aggregate existing per-seed result CSVs then exit")
    a = p.parse_args()
    run(a.bender, a.out, a.seeds, a.device,
        emb_cache=a.emb_cache,
        extract_only=a.extract_only,
        aggregate_only=a.aggregate_only)
