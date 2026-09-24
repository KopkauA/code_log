"""
Train a linear regression from SeqDance embeddings -> BENDER targets
(nu, delta, a0, rg) using the in-distribution training pool, then evaluate
R^2 on the OOD viral holdout. Finally compute ES_OOD = R2_SeqDance / R2_KESTREL
for each target, using KESTREL's own R^2 values that you already have.

This is the SeqDance-side equivalent of KESTREL's OOD generalization test --
same targets, same viral holdout, just using SeqDance embeddings + a simple
linear regression instead of KESTREL's own specialist heads.

Usage:
    python eval_bender_ood.py \
        --input merged.csv \
        --train_emb bender_train_emb.pkl \
        --viral_emb bender_viral_emb.pkl \
        --kestrel_r2 nu=0.71,delta=0.68,a0=0.64,rg=0.80 \
        --output ood_results.csv
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# The four targets we want to evaluate on
TARGETS = ["nu", "delta", "a0", "rg"]

# This function parses command-line arguments and returns them as a Namespace object
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to the merged BENDER/KESTREL CSV")
    p.add_argument("--train_emb", required=True, help="Pickle of in-distribution training embeddings")
    p.add_argument("--viral_emb", required=True, help="Pickle of OOD viral embeddings")
    p.add_argument("--pca_dim", type=int, default=200, help="Set to 0 to skip PCA")
    p.add_argument("--kestrel_r2", default=None,
                    help="Comma-separated target=value pairs of KESTREL's own R^2 on the same viral "
                         "holdout, e.g. 'nu=0.71,delta=0.68,a0=0.64,rg=0.80'. If omitted, ES_OOD is "
                         "left blank and only R2_SeqDance is reported.")
    p.add_argument("--output", default=None)
    return p.parse_args()

# This function parses a string of comma-separated key=value pairs into a dictionary
def parse_kestrel_r2(arg_str):
    if not arg_str:
        return {}
    out = {}
    for pair in arg_str.split(","):
        k, v = pair.split("=")
        out[k.strip()] = float(v.strip())
    return out

# The main function orchestrates the entire evaluation process:
def main():
    args = parse_args() # parse command-line arguments into a Namespace object
    kestrel_r2 = parse_kestrel_r2(args.kestrel_r2) # parse the KESTREL R^2 values from the command-line argument into a dictionary

    df = pd.read_csv(args.input)

    # Load the embeddings from the pickle files
    with open(args.train_emb, "rb") as f:
        train_emb = pickle.load(f)
    with open(args.viral_emb, "rb") as f:
        viral_emb = pickle.load(f)

    # Filter the dataframe to only include sequences for which we have embeddings
    train_df = df[df["UniProt_ID"].isin(train_emb.keys())].reset_index(drop=True)
    viral_df = df[df["UniProt_ID"].isin(viral_emb.keys())].reset_index(drop=True)
    print(f"[info] {len(train_df)} training sequences, {len(viral_df)} viral OOD sequences with embeddings")

    # Concatenate the embeddings into matrices for training and viral holdout
    X_train_full = np.concatenate([train_emb[i] for i in train_df["UniProt_ID"]])
    X_viral_full = np.concatenate([viral_emb[i] for i in viral_df["UniProt_ID"]])

    pca = None # pca means "principal component analysis" -- a technique for reducing the dimensionality of the embeddings while preserving as much variance as possible
    if args.pca_dim:
        pca = PCA(n_components=args.pca_dim)
        X_train_full = pca.fit_transform(X_train_full)
        X_viral_full = pca.transform(X_viral_full)
        print(f"[info] reduced embeddings to {args.pca_dim} dims via PCA")

    results = []
    for target in TARGETS:
        y_train = train_df[target].to_numpy() # convert the target values for the training set into a numpy array
        y_viral = viral_df[target].to_numpy() # convert the target values for the viral holdout into a numpy array

        model = LinearRegression() # create a new linear regression model
        model.fit(X_train_full, y_train) # fit the linear regression model to the training embeddings and target values
        y_pred = model.predict(X_viral_full) # use the fitted model to predict the target values for the viral holdout embeddings

        r2_seqdance = r2_score(y_viral, y_pred) # compute the R^2 score for the SeqDance embeddings on the viral holdout
        r2_kestrel = kestrel_r2.get(target) # get the corresponding R^2 score for KESTREL on the same target, if provided
        es_ood = (r2_seqdance / r2_kestrel) if r2_kestrel else None # compute the OOD generalization score (ES_OOD) as the ratio of SeqDance R^2 to KESTREL R^2, if KESTREL's R^2 is available

        results.append({
            "target": target,
            "R2_SeqDance": r2_seqdance,
            "R2_KESTREL": r2_kestrel,
            "ES_OOD": es_ood,
        })
        r2_kestrel_str = f"{r2_kestrel:.4f}" if r2_kestrel is not None else "N/A"
        es_ood_str = f"{es_ood:.4f}" if es_ood is not None else "N/A"
        print(f"[info] {target}: R2_SeqDance={r2_seqdance:.4f}, R2_KESTREL={r2_kestrel_str}, ES_OOD={es_ood_str}")

    results_df = pd.DataFrame(results)
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"[info] wrote results to {args.output}")


if __name__ == "__main__":
    main()
