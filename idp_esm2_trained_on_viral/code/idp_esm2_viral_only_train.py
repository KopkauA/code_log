"""
idp_esm2_viral_only_train.py

Retrain the GeoHead MLP using ONLY viral BENDER sequences (kingdom == "Viruses"),
with its own train/val/test split carved out of the viral subset -- as opposed
to the original script, which trains on non-viral sequences and only ever
*evaluates* on viral sequences as an OOD test.

This lets you compare:
  - head trained on non-viral, tested on viral   (your original OOD result)
  - head trained on viral,     tested on viral   (this script)

## Usage

python3 idp_esm2_viral_only_train.py \
    --bender merged.csv \
    --out    esm2_viral_seed42/ \
    --models 8M 150M \
    --seed   42
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from collections import deque
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

BENDER_TARGETS = ["rg", "ree", "nu", "delta", "a0"]
MAX_SEQ_LEN    = 256
BATCH_SIZE     = 32
EMBED_BATCH    = 32
HEAD_EPOCHS    = 200
HEAD_PATIENCE  = 20
HEAD_LR        = 3e-3
HEAD_DROPOUT   = 0.1

ESM2_MODELS = {
    "8M": {
        "hf_id":      "InstaDeepAI/IDP-ESM2-8M",
        "tokenizer":  "facebook/esm2_t6_8M_UR50D",
        "hidden_dim": 320,
    },
    "150M": {
        "hf_id":      "InstaDeepAI/IDP-ESM2-150M",
        "tokenizer":  "facebook/esm2_t12_35M_UR50D",
        "hidden_dim": 640,
    },
}


class GeoHead(nn.Module):
    def __init__(self, hidden_dim, n_out=len(BENDER_TARGETS), dropout=HEAD_DROPOUT):
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


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, targets_df, target_mean, target_std,
                 targets=BENDER_TARGETS):
        self.emb     = embeddings
        self.tgt_raw = torch.tensor(targets_df[targets].values.astype(float),
                                     dtype=torch.float32)
        self.mean    = torch.tensor(target_mean, dtype=torch.float32)
        self.std     = torch.tensor(target_std,  dtype=torch.float32)
        self.tgt     = torch.nan_to_num((self.tgt_raw - self.mean) / self.std, nan=0.0)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return {"emb": self.emb[idx], "targets": self.tgt[idx],
                "targets_raw": self.tgt_raw[idx]}


def load_bender(path, max_seq_len=MAX_SEQ_LEN):
    df = pd.read_csv(path)
    df = df.dropna(subset=["sequence"] + BENDER_TARGETS)
    n_before = len(df)
    df = df[df["sequence"].str.len() <= max_seq_len].reset_index(drop=True)
    print(f"BENDER: {n_before} -> {len(df)} sequences (<= {max_seq_len} aa)")
    return df


def make_splits(df, train_frac=0.80, val_frac=0.10, seed=42):
    """Cluster-aware split if cluster_id present, else random."""
    if "cluster_id" in df.columns:
        rng = np.random.default_rng(seed)
        clusters = df["cluster_id"].unique()
        rng.shuffle(clusters)
        n  = len(clusters)
        n1 = int(train_frac * n)
        n2 = int(val_frac * n)
        tc, vc, ec = set(clusters[:n1]), set(clusters[n1:n1+n2]), set(clusters[n1+n2:])
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


def load_esm2(model_key, device):
    cfg = ESM2_MODELS[model_key]
    print(f"\nLoading IDP-ESM2-{model_key} backbone from {cfg['hf_id']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    model     = AutoModel.from_pretrained(cfg["hf_id"])
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Hidden dim: {cfg['hidden_dim']}  "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model, cfg["hidden_dim"]


@torch.no_grad()
def extract_embeddings(sequences, tokenizer, esm_model, device,
                        batch_size=EMBED_BATCH, max_len=MAX_SEQ_LEN):
    all_embs = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch = [s[:max_len] for s in sequences[i:i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len + 2)
        inputs  = {k: v.to(device) for k, v in inputs.items()}
        outputs = esm_model(**inputs)
        hidden  = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb  = (hidden * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def train_geohead(hidden_dim, train_emb, val_emb, train_df, val_df,
                   out_dir, name, epochs=HEAD_EPOCHS, patience=HEAD_PATIENCE,
                   lr=HEAD_LR, device=None):
    device = device or torch.device("cpu")

    t_mean = train_df[BENDER_TARGETS].mean().values
    t_std  = train_df[BENDER_TARGETS].std().clip(lower=1e-6).values

    train_ds = EmbeddingDataset(train_emb, train_df, t_mean, t_std)
    val_ds   = EmbeddingDataset(val_emb,   val_df,   t_mean, t_std)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = GeoHead(hidden_dim).to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    best  = float("inf")
    wait  = 0
    # NOTE: "_viral" suffix so this never overwrites your original checkpoint
    ckpt  = os.path.join(out_dir, f"{name}_viral_geohead_best.pt")
    val_window = deque(maxlen=3)

    print(f"\nTraining GeoHead (viral-only) for {name} ({model.n_params():,} params)")
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


def r2_scores(preds, targets_tensor, target_names):
    out = {}
    for i, name in enumerate(target_names):
        y, yh  = targets_tensor[:, i].numpy(), preds[:, i].numpy()
        ss_res = ((y - yh) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum() + 1e-10
        out[name] = float(1 - ss_res / ss_tot)
    return out


@torch.no_grad()
def evaluate_on_df(model, embeddings, df, t_mean, t_std, split_name, device):
    model.eval()
    t_mean_t = torch.tensor(t_mean, dtype=torch.float32)
    t_std_t  = torch.tensor(t_std,  dtype=torch.float32)

    preds = []
    for i in range(0, len(embeddings), BATCH_SIZE):
        emb  = embeddings[i:i + BATCH_SIZE].to(device)
        pred = model(emb).cpu()
        preds.append(pred * t_std_t + t_mean_t)
    preds = torch.cat(preds)

    available = [t for t in BENDER_TARGETS if t in df.columns]
    targets   = torch.tensor(df[available].values.astype(float), dtype=torch.float32)
    r2 = r2_scores(preds, targets, available)

    print(f"\n-- {split_name} ({len(df)} sequences) ----------------")
    for tgt, score in r2.items():
        print(f"  {tgt:<12} {score:>6.4f}")
    return r2, preds


def run(bender_csv, out_dir, model_keys, seed=42, device_str=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda"); print("Device: CUDA")
    else:
        device = torch.device("cpu");  print("Device: CPU")

    print("\n=== Loading data ===")
    bender_df = load_bender(bender_csv)

    # KEY DIFFERENCE from the original script: split WITHIN the viral
    # subset itself, instead of training on non-viral and holding out
    # viral sequences entirely.
    viral_df = bender_df[bender_df["kingdom"] == "Viruses"].reset_index(drop=True)
    print(f"\nViral BENDER sequences: {len(viral_df)}")
    viral_train, viral_val, viral_test = make_splits(viral_df, seed=seed)

    all_results = {}

    for model_key in model_keys:
        print(f"\n{'='*60}\nMODEL: IDP-ESM2-{model_key}\n{'='*60}")
        name = f"IDP-ESM2-{model_key}"

        tokenizer, esm_model, hidden_dim = load_esm2(model_key, device)

        print("\nExtracting viral train/val/test embeddings...")
        train_emb = extract_embeddings(viral_train["sequence"].tolist(), tokenizer, esm_model, device)
        val_emb   = extract_embeddings(viral_val["sequence"].tolist(),   tokenizer, esm_model, device)
        test_emb  = extract_embeddings(viral_test["sequence"].tolist(),  tokenizer, esm_model, device)

        esm_model.cpu()
        torch.cuda.empty_cache()

        model, t_mean, t_std = train_geohead(
            hidden_dim, train_emb, val_emb, viral_train, viral_val,
            out_dir, name, device=device)

        r2, preds = evaluate_on_df(
            model, test_emb, viral_test, t_mean, t_std,
            f"{name} -- Viral-only TEST", device)

        all_results[name] = r2

        pred_df = viral_test[["sequence"] + BENDER_TARGETS].copy()
        for i, tgt in enumerate(BENDER_TARGETS):
            pred_df[f"{tgt}_pred"] = preds[:, i].numpy()
        pred_df.to_csv(os.path.join(out_dir, f"{name}_viral_test_preds.csv"), index=False)

    rows = [{"model": name, **r2} for name, r2 in all_results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "viral_only_r2_summary.csv"), index=False)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Retrain GeoHead MLP on viral-only BENDER sequences")
    p.add_argument("--bender", required=True, help="Path to BENDER merged.csv")
    p.add_argument("--out",    default="viral_only_results/", help="Output directory")
    p.add_argument("--models", nargs="+", default=["8M", "150M"], choices=["8M", "150M"])
    p.add_argument("--seed",   default=42, type=int)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    run(a.bender, a.out, a.models, a.seed, a.device)
