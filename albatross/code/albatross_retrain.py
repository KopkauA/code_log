"""
ALBATROSS retrain on BENDER_BIO.csv
────────────────────────────────────
Architecture : PARROT BRNN_MtO  (bidirectional LSTM → linear head)
               One model per target; weights saved as
               <out>/<target>/network.pt  (SPARROW-compatible layout)
Split        : cluster-aware 80/10/10; Viruses held out as OOD
               (mirrors kestrel_ood_virus.py)
Encoding     : PARROT one-hot (20 canonical AAs, input_size=20)
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# PARROT one-hot order (alphabetical by AA letter)
AA_VOCAB   = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX  = {aa: i for i, aa in enumerate(AA_VOCAB)}
INPUT_SIZE = 20

GEO_TARGETS = ["rg", "ree", "nu", "delta", "a0"]
GRAPH_TARGETS = [
    "global_efficiency", "fragmentation_index",
    "avg_clustering", "transitivity", "degree_assortativity",
]
ALL_TARGETS = GEO_TARGETS + GRAPH_TARGETS

OOD_KINGDOM = "Viruses"

# Per-target BRNN config.
# Geo targets use the same hidden_size / num_layers as the published SPARROW
# v2 weights (inferred from blob shapes in sparrow/data/networks/).
# Graph targets are new — defaults to (64, 2); pass --hidden_size / --num_layers
# on the CLI to override the graph-target fallback.
TARGET_CONFIG = {
    # target       : (hidden_size, num_layers)
    "rg"           : (45, 1),
    "ree"          : (45, 1),   # SPARROW calls this "re"
    "nu"           : (35, 2),   # SPARROW: scaling_exponent
    "delta"        : (55, 2),   # SPARROW: asphericity
    "a0"           : (70, 1),   # SPARROW: prefactor
    # graph targets — no published ALBATROSS analogue; use tunable defaults
    "global_efficiency"      : (64, 2),
    "fragmentation_index"    : (64, 2),
    "avg_clustering"         : (64, 2),
    "transitivity"           : (64, 2),
    "degree_assortativity"   : (64, 2),
}


# ─────────────────────────────────────────────
# ARCHITECTURE  (PARROT BRNN_MtO, inlined)
# ─────────────────────────────────────────────

class BRNN_MtO(nn.Module):
    """PARROT BRNN_MtO: bidirectional LSTM → linear head (many-to-one)."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc   = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size, device=x.device)
        _, (h_n, _) = self.lstm(x, (h0, c0))
        # h_n[-2]: final hidden of forward direction
        # h_n[-1]: final hidden of backward direction
        final = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        return self.fc(final)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# SPLITS  (cluster-aware 80/10/10 + OOD)
# ─────────────────────────────────────────────

def make_splits(df, ood_kingdom=OOD_KINGDOM,
                train_frac=0.80, val_frac=0.10, seed=42):
    rng    = np.random.default_rng(seed)
    ood_df = df[df["kingdom"] == ood_kingdom].copy()
    main   = df[df["kingdom"] != ood_kingdom].copy()

    print(f"OOD ({ood_kingdom}): {len(ood_df)}")
    print(f"Main: {len(main)}")

    if "cluster_id" in main.columns:
        clusters = main["cluster_id"].unique()
        rng.shuffle(clusters)
        n  = len(clusters)
        n1 = int(train_frac * n)
        n2 = int(val_frac   * n)
        tc = set(clusters[:n1])
        vc = set(clusters[n1:n1+n2])
        ec = set(clusters[n1+n2:])
        tr = main[main["cluster_id"].isin(tc)]
        va = main[main["cluster_id"].isin(vc)]
        te = main[main["cluster_id"].isin(ec)]
    else:
        idx = rng.permutation(len(main))
        n   = len(idx)
        n1  = int(train_frac * n)
        n2  = int(val_frac   * n)
        tr  = main.iloc[idx[:n1]]
        va  = main.iloc[idx[n1:n1+n2]]
        te  = main.iloc[idx[n1+n2:]]

    print(f"Train:{len(tr)}  Val:{len(va)}  Test:{len(te)}  OOD:{len(ood_df)}")
    return {"train": tr, "val": va, "test": te, "ood": ood_df}


# ─────────────────────────────────────────────
# DATASET + COLLATE
# ─────────────────────────────────────────────

class TargetDataset(Dataset):
    def __init__(self, df, target, mean, std, max_len=256):
        self.df      = df.dropna(subset=[target]).reset_index(drop=True)
        self.target  = target
        self.mean    = mean
        self.std     = std
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["sequence"])[:self.max_len]
        L   = len(seq)
        enc = torch.zeros(L, INPUT_SIZE)
        for i, aa in enumerate(seq):
            j = AA_TO_IDX.get(aa, -1)
            if j >= 0:
                enc[i, j] = 1.0
        val  = (float(row[self.target]) - self.mean) / self.std
        return enc, torch.tensor([val], dtype=torch.float32), L


def pad_collate(batch):
    encs, tgts, lengths = zip(*batch)
    max_len = max(lengths)
    padded  = torch.zeros(len(encs), max_len, INPUT_SIZE)
    for i, (e, l) in enumerate(zip(encs, lengths)):
        padded[i, :l] = e
    return padded, torch.stack(tgts)


# ─────────────────────────────────────────────
# TRAINING + EVALUATION
# ─────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("Device: MPS"); return torch.device("mps")
    if torch.cuda.is_available():
        print("Device: CUDA"); return torch.device("cuda")
    print("Device: CPU");      return torch.device("cpu")


def r2_score(preds, targets):
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum() + 1e-10
    return float(1 - ss_res / ss_tot)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.L1Loss(reduction="sum")
    all_p, all_t = [], []
    total_loss = 0.0
    for x, t in loader:
        p = model(x.to(device)).cpu()
        total_loss += criterion(p, t).item()
        all_p.append(p); all_t.append(t)
    p = torch.cat(all_p); t = torch.cat(all_t)
    return total_loss, r2_score(p, t)


def train_one(target, splits, mean, std,
              hidden_size, num_layers, epochs,
              batch_size, lr, out_dir, device,
              max_len=256, default_hidden=64, default_layers=2):

    pin = (str(device) != "mps")
    kw  = dict(collate_fn=pad_collate, num_workers=0, pin_memory=pin)

    def loader(split_df, shuffle):
        ds = TargetDataset(split_df, target, mean, std, max_len=max_len)
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kw)

    train_dl = loader(splits["train"], True)
    val_dl   = loader(splits["val"],   False)
    test_dl  = loader(splits["test"],  False)
    ood_dl   = loader(splits["ood"],   False) if len(splits["ood"]) else None

    if train_dl is None or val_dl is None:
        print(f"  Skipping {target}: insufficient data")
        return None, {"target": target}

    # use per-target config if available, then CLI override, then fallback
    hs, nl = TARGET_CONFIG.get(target, (default_hidden, default_layers))
    if hidden_size is not None:
        hs = hidden_size
    if num_layers is not None:
        nl = num_layers
    model = BRNN_MtO(INPUT_SIZE, hs, nl, 1).to(device)
    opt   = Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss(reduction="sum")

    tgt_dir = os.path.join(out_dir, target)
    os.makedirs(tgt_dir, exist_ok=True)
    ckpt = os.path.join(tgt_dir, "network.pt")

    best_val  = float("inf")
    val_hist  = []   # for PARROT auto-stop
    hist      = []
    MAX_EPOCHS = 5000

    print(f"\n── {target}  ({model.n_params():,} params) "
          f"  train={len(train_dl.dataset)}  val={len(val_dl.dataset)}")
    print(f"{'Ep':>4} {'Trn':>12} {'Val':>12} {'valR²':>7}")
    print("─" * 40)

    for ep in range(MAX_EPOCHS):
        model.train()
        tl = 0.0
        for x, t in train_dl:
            p    = model(x.to(device))
            loss = criterion(p, t.to(device))
            opt.zero_grad(); loss.backward()
            opt.step(); tl += loss.item()

        vl, vr2 = evaluate(model, val_dl, device)
        hist.append({"epoch": ep+1, "train": tl, "val": vl, "val_r2": vr2})
        print(f"{ep+1:>4} {tl:>12.2f} {vl:>12.2f} {vr2:>7.3f}")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt)

        # PARROT auto-stop: after `epochs` minimum epochs, check whether
        # val loss failed to improve by >0.5% for `epochs` consecutive epochs
        val_hist.append(vl)
        if ep + 1 >= epochs and len(val_hist) >= epochs:
            window = val_hist[-epochs:]
            if all(window[i] >= window[i-1] * 0.995
                   for i in range(1, len(window))):
                print(f"  Auto-stop @ ep {ep+1}"); break

    pd.DataFrame(hist).to_csv(
        os.path.join(tgt_dir, "history.csv"), index=False)
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=True))

    row = {"target": target}
    for split_name, dl in [("test", test_dl), ("ood", ood_dl)]:
        if dl is None:
            continue
        _, r2 = evaluate(model, dl, device)
        row[f"{split_name}_r2"] = r2
        print(f"  {split_name.upper():4s} R²: {r2:.4f}")

    return model, row


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(data_csv, out_dir, targets,
        hidden_size, num_layers,
        epochs, batch_size, lr, seed, max_len=256):

    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    df = pd.read_csv(data_csv)
    df = df.dropna(subset=["sequence"])
    # ensure graph targets exist (fill NaN if absent)
    for col in GRAPH_TARGETS:
        if col not in df.columns:
            df[col] = np.nan
    print(f"Loaded {len(df)} sequences | {df['kingdom'].nunique()} kingdoms")

    splits = make_splits(df, seed=seed)

    # persist split row-indices for reproducibility
    for name, sdf in splits.items():
        sdf.index.to_series().to_csv(
            os.path.join(out_dir, f"split_{name}.csv"), index=False)

    results = []
    for tgt in targets:
        if tgt not in df.columns:
            print(f"Skipping {tgt}: column absent"); continue

        train_vals = splits["train"][tgt].dropna()
        if len(train_vals) < 10:
            print(f"Skipping {tgt}: too few training rows"); continue

        mean = float(train_vals.mean())
        std  = float(max(train_vals.std(), 1e-6))

        # write norm stats before train_one creates the dir
        tgt_dir = os.path.join(out_dir, tgt)
        os.makedirs(tgt_dir, exist_ok=True)
        pd.Series({"mean": mean, "std": std}).to_csv(
            os.path.join(tgt_dir, "norm_stats.csv"), header=False)

        _, row = train_one(
            tgt, splits, mean, std,
            hidden_size, num_layers, epochs,
            batch_size, lr, out_dir, device,
            max_len=max_len, default_hidden=64, default_layers=2)
        if row:
            results.append(row)

    if results:
        summary = pd.DataFrame(results).set_index("target")
        summary.to_csv(os.path.join(out_dir, "summary_r2.csv"))
        print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
        print(summary.to_string())


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Retrain ALBATROSS-style BRNN_MtO models on BENDER_BIO.csv")
    p.add_argument("--data",
                   default="/Volumes/User Homes/Nu Project/BENDER-BIO/BENDER_BIO.csv")
    p.add_argument("--out",         default="albatross_retrain_output/")
    p.add_argument("--targets",     nargs="+", default=ALL_TARGETS,
                   help="Subset of targets to train (default: all 10)")
    p.add_argument("--hidden_size", default=None, type=int,
                   help="Override LSTM hidden units (default: per TARGET_CONFIG)")
    p.add_argument("--num_layers",  default=None, type=int,
                   help="Override stacked LSTM layers (default: per TARGET_CONFIG)")
    p.add_argument("--epochs",      default=25,   type=int,
                   help="Min epochs before auto-stop; also the check window (PARROT default 25)")
    p.add_argument("--batch_size",  default=64,   type=int)
    p.add_argument("--lr",          default=1e-3, type=float)
    p.add_argument("--seed",        default=42,   type=int)
    p.add_argument("--max_len",     default=256,  type=int,
                   help="Truncate sequences to this length (default 256)")
    a = p.parse_args()
    run(a.data, a.out, a.targets,
        a.hidden_size, a.num_layers,
        a.epochs, a.batch_size, a.lr, a.seed,
        max_len=a.max_len)

# ─────────────────────────────────────────────
# QUICK START
# ─────────────────────────────────────────────
#
#   python albatross_retrain.py
#
# or for a single target:
#
#   python albatross_retrain.py --targets rg --epochs 50
#
# Outputs per target (SPARROW-compatible layout):
#   albatross_retrain_output/<target>/network.pt      ← weights
#   albatross_retrain_output/<target>/norm_stats.csv  ← mean / std
#   albatross_retrain_output/<target>/history.csv     ← loss curve
#   albatross_retrain_output/split_{train,val,test,ood}.csv
#   albatross_retrain_output/summary_r2.csv
#
# To load a saved model later:
#
#   import torch
#   from albatross_retrain import BRNN_MtO, INPUT_SIZE
#   model = BRNN_MtO(INPUT_SIZE, hidden_size=256, num_layers=2, num_classes=1)
#   model.load_state_dict(torch.load("albatross_retrain_output/rg/network.pt"))
#
# ─────────────────────────────────────────────
