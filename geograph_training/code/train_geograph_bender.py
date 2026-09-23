"""Train GeoGraph on BENDER dataset.

Splits:
  - OOD test: all Viruses (held out entirely)
  - Remaining kingdoms: 80/10/10 train/val/test by cluster_id

Usage:
    python train_geograph_bender.py \
        --csv ./data/bender_calvados_complete.csv \
        --residue-dir ./residue_features \
        --output-dir ./geograph_training \
        --max-seq-len 256 \
        --epochs 100 \
        --batch-size 512 \
        --lr 5e-4 \
        --device cuda
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Feature definitions (matching GeoGraph paper exactly) ────────────────────

GEOMETRIC_FEATURES = [
    "end_to_end_distance",
    "radius_of_gyration",
    "asphericity",
    "scaling_law_exponent",
    "scaling_law_prefactor",
]

GRAPH_SEQUENCE_FEATURES = [
    "fragmentation_index",
    "avg_shortest_path_length",
    "global_efficiency",
    "avg_clustering",
    "transitivity",
    "degree_assortativity",
    "charge_assortativity",
    "hydrophobicity_assortativity",
]

SEQUENCE_FEATURES = GEOMETRIC_FEATURES + GRAPH_SEQUENCE_FEATURES

RESIDUE_FEATURES = [
    "degree_centrality",
    "betweenness_centrality",
    "harmonic_centrality",
    "pagerank",
    "core_number",
    "local_clustering_coeff",
    "in_lcc",
]

# Map from BENDER CSV column names to GeoGraph feature names
CSV_TO_GEOGRAPH = {
    "ree": "end_to_end_distance",
    "rg": "radius_of_gyration",
    "delta": "asphericity",
    "nu": "scaling_law_exponent",
    "A0": "scaling_law_prefactor",
    "fragmentation_index": "fragmentation_index",
    "avg_shortest_path_length": "avg_shortest_path_length",
    "global_efficiency": "global_efficiency",
    "avg_clustering": "avg_clustering",
    "transitivity": "transitivity",
    "degree_assortativity": "degree_assortativity",
    "charge_assortativity": "charge_assortativity",
    "hydrophobicity_assortativity": "hydrophobicity_assortativity",
}


# ── Dataset ──────────────────────────────────────────────────────────────────

class BenderDataset(Dataset):
    def __init__(self, df, residue_dir, seq_mean, seq_std, res_mean, res_std, max_seq_len):
        self.sequences = df["sequence"].tolist()
        self.uniprot_ids = df["UniProt_ID"].tolist()
        self.residue_dir = Path(residue_dir)
        self.max_seq_len = max_seq_len

        # Sequence-level targets (already mapped to GeoGraph names)
        self.seq_targets = torch.tensor(
            df[SEQUENCE_FEATURES].values, dtype=torch.float32
        )
        # Normalise
        self.seq_targets = (self.seq_targets - seq_mean) / seq_std

        self.res_mean = res_mean
        self.res_std = res_std

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        uid = self.uniprot_ids[idx]
        seq_target = self.seq_targets[idx]

        # Load residue-level targets
        npz_path = self.residue_dir / f"{uid}.npz"
        try:
            if npz_path.exists():
                data = np.load(npz_path)
                res_target = np.stack([data[f] for f in RESIDUE_FEATURES], axis=-1)  # (L, 7)
                res_target = torch.tensor(res_target, dtype=torch.float32)
                res_target = (res_target - self.res_mean) / self.res_std
            else:
                res_target = torch.zeros(len(seq), len(RESIDUE_FEATURES))
        except (EOFError, ValueError, OSError):
            res_target = torch.zeros(len(seq), len(RESIDUE_FEATURES))

        return seq, seq_target, res_target


def collate_fn(batch):
    sequences, seq_targets, res_targets = zip(*batch)
    seq_targets = torch.stack(seq_targets)
    # Pad residue targets to max length in batch
    max_len = max(r.shape[0] for r in res_targets)
    n_res_feats = res_targets[0].shape[1]
    padded_res = torch.zeros(len(res_targets), max_len, n_res_feats)
    res_mask = torch.zeros(len(res_targets), max_len)
    for i, r in enumerate(res_targets):
        padded_res[i, :r.shape[0], :] = r
        res_mask[i, :r.shape[0]] = 1.0
    return list(sequences), seq_targets, padded_res, res_mask


# ── Splits ───────────────────────────────────────────────────────────────────

def create_splits(df, seed=42):
    """Split data: Viruses = OOD, rest = 80/10/10 by cluster_id."""
    ood_df = df[df["kingdom"] == "Viruses"].copy()
    non_virus = df[df["kingdom"] != "Viruses"].copy()

    # Fill NaN cluster_ids with unique values so GroupShuffleSplit doesn't choke
    nan_mask = non_virus["cluster_id"].isna()
    if nan_mask.any():
        max_id = non_virus["cluster_id"].max()
        if pd.isna(max_id):
            max_id = 0
        fill_ids = range(int(max_id) + 1, int(max_id) + 1 + nan_mask.sum())
        non_virus.loc[nan_mask, "cluster_id"] = list(fill_ids)

    # First split: 80% train vs 20% rest, grouped by cluster
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, rest_idx = next(gss1.split(non_virus, groups=non_virus["cluster_id"]))

    train_df = non_virus.iloc[train_idx].copy()
    rest_df = non_virus.iloc[rest_idx].copy()

    # Second split: 50/50 val/test from the 20% rest
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["cluster_id"]))

    val_df = rest_df.iloc[val_idx].copy()
    test_df = rest_df.iloc[test_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    ood_df["split"] = "ood_virus"

    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, ood_virus={len(ood_df)}")

    return train_df, val_df, test_df, ood_df


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_seq_loss = 0
    total_res_loss = 0
    n_batches = 0

    for sequences, seq_targets, res_targets, res_mask in tqdm(
        loader, desc=f"Epoch {epoch}", leave=False
    ):
        inputs = model.tokenize(sequences)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_targets = seq_targets.to(device)
        res_targets = res_targets.to(device)
        res_mask = res_mask.to(device)

        # Forward
        embeddings = model.backbone(input_ids=input_ids, attention_mask=attention_mask)[
            "last_hidden_state"
        ]

        # Sequence-level: mean pool -> sequence head
        masked_emb = embeddings * attention_mask.unsqueeze(-1)
        mean_emb = masked_emb.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        seq_pred = model.sequence_head(mean_emb)  # (B, n_seq_feats)

        # Residue-level: per-token -> residue head
        res_pred = model.residue_head(embeddings)  # (B, L_tok, n_res_feats)

        # Align residue predictions with targets
        # The tokenizer may add special tokens — trim to sequence length
        if model.add_special_tokens:
            res_pred = res_pred[:, 1:, :]  # skip [CLS]

        min_len = min(res_pred.shape[1], res_targets.shape[1])
        res_pred = res_pred[:, :min_len, :]
        res_targets_trimmed = res_targets[:, :min_len, :]
        res_mask_trimmed = res_mask[:, :min_len]

        # Losses
        seq_loss = nn.functional.mse_loss(seq_pred, seq_targets)

        if res_mask_trimmed.sum() > 0:
            res_diff = (res_pred - res_targets_trimmed) ** 2
            res_loss = (res_diff * res_mask_trimmed.unsqueeze(-1)).sum() / (
                res_mask_trimmed.sum() * res_pred.shape[-1]
            )
        else:
            res_loss = torch.tensor(0.0, device=device)

        loss = seq_loss + res_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_seq_loss += seq_loss.item()
        total_res_loss += res_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "seq_loss": total_seq_loss / n_batches,
        "res_loss": total_res_loss / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device, split_name="val"):
    model.eval()
    total_loss = 0
    total_seq_loss = 0
    total_res_loss = 0
    n_batches = 0

    all_seq_preds = []
    all_seq_targets = []

    for sequences, seq_targets, res_targets, res_mask in loader:
        inputs = model.tokenize(sequences)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_targets = seq_targets.to(device)
        res_targets = res_targets.to(device)
        res_mask = res_mask.to(device)

        embeddings = model.backbone(input_ids=input_ids, attention_mask=attention_mask)[
            "last_hidden_state"
        ]

        masked_emb = embeddings * attention_mask.unsqueeze(-1)
        mean_emb = masked_emb.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        seq_pred = model.sequence_head(mean_emb)

        res_pred = model.residue_head(embeddings)
        if model.add_special_tokens:
            res_pred = res_pred[:, 1:, :]

        min_len = min(res_pred.shape[1], res_targets.shape[1])
        res_pred = res_pred[:, :min_len, :]
        res_targets_trimmed = res_targets[:, :min_len, :]
        res_mask_trimmed = res_mask[:, :min_len]

        seq_loss = nn.functional.mse_loss(seq_pred, seq_targets)

        if res_mask_trimmed.sum() > 0:
            res_diff = (res_pred - res_targets_trimmed) ** 2
            res_loss = (res_diff * res_mask_trimmed.unsqueeze(-1)).sum() / (
                res_mask_trimmed.sum() * res_pred.shape[-1]
            )
        else:
            res_loss = torch.tensor(0.0, device=device)

        total_loss += (seq_loss + res_loss).item()
        total_seq_loss += seq_loss.item()
        total_res_loss += res_loss.item()
        n_batches += 1

        all_seq_preds.append(seq_pred.cpu())
        all_seq_targets.append(seq_targets.cpu())

    all_seq_preds = torch.cat(all_seq_preds, dim=0)
    all_seq_targets = torch.cat(all_seq_targets, dim=0)

    # Per-feature R² on normalised targets
    r2_scores = {}
    for i, feat_name in enumerate(SEQUENCE_FEATURES):
        y_true = all_seq_targets[:, i]
        y_pred = all_seq_preds[:, i]
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        r2_scores[feat_name] = r2.item()

    return {
        "loss": total_loss / n_batches,
        "seq_loss": total_seq_loss / n_batches,
        "res_loss": total_res_loss / n_batches,
        "r2": r2_scores,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GeoGraph on BENDER")
    parser.add_argument("--csv", required=True, help="Path to bender_calvados_complete.csv")
    parser.add_argument("--residue-dir", required=True, help="Directory with per-protein .npz residue features")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and prepare data ──
    print("Loading data...")
    df = pd.read_csv(args.csv)

    # Rename CSV columns to GeoGraph feature names
    rename_map = {csv_col: geo_col for csv_col, geo_col in CSV_TO_GEOGRAPH.items() if csv_col in df.columns}
    df = df.rename(columns=rename_map)

    # Filter by max sequence length
    df = df[df["sequence"].str.len() <= args.max_seq_len]
    print(f"After length filter (<= {args.max_seq_len}): {len(df)} proteins")

    # Drop rows with NaN in any target feature
    available_seq_features = [f for f in SEQUENCE_FEATURES if f in df.columns]
    missing_features = [f for f in SEQUENCE_FEATURES if f not in df.columns]
    if missing_features:
        print(f"WARNING: Missing sequence features (will be zeroed): {missing_features}")
        for f in missing_features:
            df[f] = 0.0
        available_seq_features = SEQUENCE_FEATURES

    df = df.dropna(subset=available_seq_features)
    print(f"After dropping NaN targets: {len(df)} proteins")

    # ── Splits ──
    print("\nCreating splits...")
    train_df, val_df, test_df, ood_df = create_splits(df, seed=args.seed)

    # Save splits
    all_splits = pd.concat([train_df, val_df, test_df, ood_df])
    all_splits.to_csv(output_dir / "splits.csv", index=False)

    # ── Compute normalisation stats from training set ──
    seq_values = torch.tensor(train_df[SEQUENCE_FEATURES].values, dtype=torch.float32)
    seq_mean = seq_values.mean(dim=0)
    seq_std = seq_values.std(dim=0).clamp(min=1e-6)

    # Residue-level stats: sample from training set
    print("Computing residue feature stats from training set...")
    res_accum = []
    residue_dir = Path(args.residue_dir)
    for uid in train_df["UniProt_ID"].values[:2000]:  # sample for speed
        npz_path = residue_dir / f"{uid}.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            vals = np.stack([data[f] for f in RESIDUE_FEATURES], axis=-1)  # (L, 7)
            res_accum.append(vals)
    if res_accum:
        all_res = np.concatenate(res_accum, axis=0)
        res_mean = torch.tensor(all_res.mean(axis=0), dtype=torch.float32)
        res_std = torch.tensor(all_res.std(axis=0), dtype=torch.float32).clamp(min=1e-6)
    else:
        res_mean = torch.zeros(len(RESIDUE_FEATURES))
        res_std = torch.ones(len(RESIDUE_FEATURES))

    # ── Datasets and loaders ──
    train_ds = BenderDataset(train_df, args.residue_dir, seq_mean, seq_std, res_mean, res_std, args.max_seq_len)
    val_ds = BenderDataset(val_df, args.residue_dir, seq_mean, seq_std, res_mean, res_std, args.max_seq_len)
    test_ds = BenderDataset(test_df, args.residue_dir, seq_mean, seq_std, res_mean, res_std, args.max_seq_len)
    ood_ds = BenderDataset(ood_df, args.residue_dir, seq_mean, seq_std, res_mean, res_std, args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── Model config (matching paper: 4-layer, hidden=256, heads=4, RoPE) ──
    cfg = OmegaConf.create({
        "tokenizer": {
            "model_name": "facebook/esm2_t6_8M_UR50D",
            "add_special_tokens": True,
            "max_length": args.max_seq_len + 2,  # +2 for [CLS] and [EOS]
        },
        "backbone": {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "expansion_factor": 2,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "position_embedding_type": "rotary",
        },
        "head_hidden_factor": 0.5,
    })

    features_info = {
        "sequence_features": SEQUENCE_FEATURES,
        "residue_features": RESIDUE_FEATURES,
        "geometric_features": GEOMETRIC_FEATURES,
    }

    features_stats = {
        "sequence_features": {"mean": seq_mean, "std": seq_std},
        "residue_features": {"mean": res_mean, "std": res_std},
    }

    # ── Build model ──
    print("\nBuilding model...")
    model = GeoGraphTrainable(cfg, features_info, features_stats)
    model = model.to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer and scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    # ── Training loop ──
    print(f"\nTraining for {args.epochs} epochs on {args.device}")
    print(f"Total steps: {total_steps}, warmup: {args.warmup_steps}")

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, args.device, epoch)
        val_metrics = evaluate(model, val_loader, args.device, "val")

        geo_r2 = {k: v for k, v in val_metrics["r2"].items() if k in GEOMETRIC_FEATURES}
        mean_geo_r2 = np.mean(list(geo_r2.values()))

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_geo_R²={mean_geo_r2:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_seq_loss": train_metrics["seq_loss"],
            "train_res_loss": train_metrics["res_loss"],
            "val_loss": val_metrics["loss"],
            "val_seq_loss": val_metrics["seq_loss"],
            "val_res_loss": val_metrics["res_loss"],
            "val_r2": val_metrics["r2"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, cfg, features_info, features_stats, output_dir / "best_model.ckpt")
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(model, cfg, features_info, features_stats, output_dir / f"epoch_{epoch}.ckpt")

    # ── Final evaluation ──
    print("\n" + "=" * 60)
    print("Final evaluation on test and OOD sets")
    print("=" * 60)

    # Reload best model
    ckpt = torch.load(output_dir / "best_model.ckpt", map_location=args.device, weights_only=False)
    model_eval = GeoGraphTrainable(cfg, features_info, features_stats)
    model_eval.load_state_dict(ckpt["state_dict"])
    model_eval = model_eval.to(args.device)

    for name, loader in [("test", test_loader), ("ood_virus", ood_loader)]:
        metrics = evaluate(model_eval, loader, args.device, name)
        print(f"\n{name.upper()} set:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  R² per feature:")
        for feat, r2 in metrics["r2"].items():
            label = "GEO" if feat in GEOMETRIC_FEATURES else "GRF"
            print(f"    [{label}] {feat}: {r2:.4f}")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nAll outputs saved to {output_dir}")


# ── Model wrapper (adds training forward pass) ──────────────────────────────

class GeoGraphTrainable(nn.Module):
    """GeoGraph with a training-compatible forward pass."""

    def __init__(self, cfg, features_info, features_stats=None):
        super().__init__()
        from transformers import AutoTokenizer, EsmConfig, EsmModel

        self.sequence_features = features_info["sequence_features"]
        self.residue_features = features_info["residue_features"]
        self.geometric_features = features_info["geometric_features"]
        self.num_sequence_features = len(self.sequence_features)
        self.num_residue_features = len(self.residue_features)

        # Register normalisation stats
        if features_stats is not None:
            self.register_buffer("sequence_features_mean", features_stats["sequence_features"]["mean"])
            self.register_buffer("sequence_features_std", features_stats["sequence_features"]["std"])
            self.register_buffer("residue_features_mean", features_stats["residue_features"]["mean"])
            self.register_buffer("residue_features_std", features_stats["residue_features"]["std"])
        else:
            self.register_buffer("sequence_features_mean", torch.zeros(self.num_sequence_features))
            self.register_buffer("sequence_features_std", torch.ones(self.num_sequence_features))
            self.register_buffer("residue_features_mean", torch.zeros(self.num_residue_features))
            self.register_buffer("residue_features_std", torch.ones(self.num_residue_features))

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.model_name)
        self.add_special_tokens = cfg.tokenizer.add_special_tokens
        self.max_sequence_length = cfg.tokenizer.max_length

        # Backbone
        config = EsmConfig(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            max_position_embeddings=self.max_sequence_length,
            num_hidden_layers=cfg.backbone.num_hidden_layers,
            num_attention_heads=cfg.backbone.num_attention_heads,
            hidden_size=cfg.backbone.hidden_size,
            intermediate_size=cfg.backbone.expansion_factor * cfg.backbone.hidden_size,
            hidden_dropout_prob=cfg.backbone.hidden_dropout_prob,
            attention_probs_dropout_prob=cfg.backbone.attention_probs_dropout_prob,
            position_embedding_type=cfg.backbone.position_embedding_type,
        )
        self.backbone = EsmModel(config, add_pooling_layer=False)

        # Heads
        hidden_dim = int(cfg.backbone.hidden_size * cfg.head_hidden_factor)
        self.sequence_head = nn.Sequential(
            nn.Linear(cfg.backbone.hidden_size, hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.backbone.hidden_dropout_prob),
            nn.Linear(hidden_dim, self.num_sequence_features),
        )
        self.residue_head = nn.Sequential(
            nn.Linear(cfg.backbone.hidden_size, hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.backbone.hidden_dropout_prob),
            nn.Linear(hidden_dim, self.num_residue_features),
        )

    def tokenize(self, sequences):
        return self.tokenizer(
            sequences,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def forward(self, input_ids, attention_mask, return_geometric_features=True, return_embeddings=False):
        embeddings = self.backbone(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
        outputs = {}
        if return_embeddings:
            outputs["embeddings"] = embeddings
        if return_geometric_features:
            masked = embeddings * attention_mask.unsqueeze(-1)
            mean_emb = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            seq_pred = self.sequence_head(mean_emb)
            seq_pred = seq_pred * self.sequence_features_std + self.sequence_features_mean
            geo = {
                name: seq_pred[:, i]
                for i, name in enumerate(self.sequence_features)
                if name in self.geometric_features
            }
            outputs["geometric_features"] = geo
        return outputs


def save_checkpoint(model, cfg, features_info, features_stats, path):
    torch.save({
        "state_dict": model.state_dict(),
        "hyper_parameters": {
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "features_info": features_info,
            "features_stats": {
                "sequence_features": {
                    "mean": features_stats["sequence_features"]["mean"],
                    "std": features_stats["sequence_features"]["std"],
                },
                "residue_features": {
                    "mean": features_stats["residue_features"]["mean"],
                    "std": features_stats["residue_features"]["std"],
                },
            },
        },
    }, path)


if __name__ == "__main__":
    main()
