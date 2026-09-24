"""
Extract mean-pooled SeqDance embeddings for the BENDER/KESTREL dataset,
loading weights from a LOCAL .safetensors checkpoint (not Hugging Face Hub).

Splits the dataset into:
    - in-distribution training pool (kingdom != 'Viruses')
    - OOD viral holdout (kingdom == 'Viruses')
and saves embeddings for both, keyed by UniProt_ID.

Usage:
    python extract_bender_embeddings.py \
        --input merged.csv \
        --checkpoint /path/to/model.safetensors \
        --output_prefix ./data/bender \
        --model_select seqdance
"""

import argparse # lets the script accept command-line flags like --input, --checkpoint
import pickle # for saving the embeddings to disk
import sys # for manipulating the Python path to import the model
import time # for timing how long the embedding extraction takes

import pandas as pd # for reading the CSV file
import torch # for tensor computations and model inference
from safetensors.torch import load_file # for loading the model checkpoint in safetensors format
from transformers import AutoTokenizer # for tokenizing protein sequences

SEQDANCE_MODEL_DIR = "./model" # path to the directory containing the model code
sys.path.insert(0, SEQDANCE_MODEL_DIR) # add the model directory to the Python path so we can import the model
from model import ESMwrap  # noqa: E402

"""
This function parses command-line arguments and returns them as a Namespace object
- a namespace object is a simple class that allows you to access the arguments as attributes
"""
def parse_args():
    p = argparse.ArgumentParser() # create a new argument parser
    p.add_argument("--input", required=True, help="Path to the merged BENDER/KESTREL CSV")
    p.add_argument("--checkpoint", required=True, help="Path to local model.safetensors")
    p.add_argument("--output_prefix", required=True,
                    help="Prefix for output pickle files -- writes <prefix>_train_emb.pkl and "
                         "<prefix>_viral_emb.pkl")
    p.add_argument("--model_select", choices=["seqdance", "esmdance"], default="seqdance",
                    help="SeqDance was randomly initialized (dynamics-only); ESMDance freezes ESM2 "
                         "weights underneath. Use 'seqdance' unless you specifically downloaded the "
                         "ESMDance checkpoint.")
    p.add_argument("--esm2_select", default="model_35M")
    p.add_argument("--min_len", type=int, default=0)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--device", default=None)
    return p.parse_args()

"""
This function loads the model architecture and weights from a local checkpoint file
"""
def load_model_from_checkpoint(esm2_select, model_select, checkpoint_path, device):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D") # load the ESM2 tokenizer from Hugging Face Hub

    # Build the empty model shell first (architecture only, random/ESM2-pretrained weights)
    model = ESMwrap(esm2_select, model_select)

    # Load the actual trained weights from the local safetensors file
    state_dict = load_file(checkpoint_path) # load the weights from the .safetensors file into a state dictionary
    missing, unexpected = model.load_state_dict(state_dict, strict=False) 
    if missing:
        print(f"[warn] {len(missing)} missing keys when loading checkpoint (showing first 5): {missing[:5]}")
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys in checkpoint (showing first 5): {unexpected[:5]}")

    model = model.to(device) # move the model to the specified device (CPU or GPU)
    model.eval() # set the model to evaluation mode (disables dropout, etc.)
    return tokenizer, model # return the tokenizer and model for use in embedding extraction

# This function takes a protein sequence, tokenizes it, passes it through the model, and returns the mean-pooled embedding
@torch.no_grad()
def get_mean_pooled_embedding(tokenizer, model, seq, device, max_len):
    inputs = tokenizer([seq], return_tensors="pt", max_length=max_len, truncation=True).to(device) # tokenize the sequence and move to device; return_tensors="pt" means we want PyTorch tensors
    output = model(inputs, return_res_emb=True, return_attention_map=False,
                    return_res_pred=False, return_pair_pred=False) # pass the tokenized input through the model to get the embeddings
    return output["res_emb"].mean(dim=1).cpu().numpy() # mean-pool the residue embeddings across the sequence length dimension, move to CPU, and convert to numpy array

# This function extracts embeddings for a given subset of the dataset (train or viral) and returns a dictionary of embeddings keyed by UniProt_ID
def extract_for_subset(df, tokenizer, model, device, max_len, label):
    embeddings = {}
    start = time.time()
    for i, row in df.iterrows():
        seq_id = row["UniProt_ID"]
        seq = row["sequence"]
        embeddings[seq_id] = get_mean_pooled_embedding(tokenizer, model, seq, device, max_len) # store the mean-pooled embedding for this sequence in the embeddings dictionary
        if (i + 1) % 100 == 0: # print progress every 100 sequences
            elapsed = time.time() - start
            print(f"[info] [{label}] {i + 1}/{len(df)} embeddings done ({elapsed:.1f}s elapsed)")
    return embeddings

# The main function orchestrates the entire embedding extraction process:
def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[info] using device: {device}")

    print(f"[info] loading {args.model_select} from local checkpoint: {args.checkpoint}")
    tokenizer, model = load_model_from_checkpoint(args.esm2_select, args.model_select, args.checkpoint, device)

    df = pd.read_csv(args.input)
    print(f"[info] loaded {len(df)} rows from {args.input}")

    df = df[(df["length"] >= args.min_len) & (df["length"] <= args.max_len)].reset_index(drop=True)
    print(f"[info] {len(df)} rows remain after length filtering ({args.min_len}-{args.max_len})")

    train_df = df[df["kingdom"] != "Viruses"].reset_index(drop=True) # filter for in-distribution training sequences (kingdom != 'Viruses')
    viral_df = df[df["kingdom"] == "Viruses"].reset_index(drop=True) # filter for out-of-distribution viral sequences
    print(f"[info] in-distribution training pool: {len(train_df)} sequences")
    print(f"[info] OOD viral holdout: {len(viral_df)} sequences")

    train_emb = extract_for_subset(train_df, tokenizer, model, device, args.max_len, "train") # extract embeddings for the training pool
    with open(f"{args.output_prefix}_train_emb.pkl", "wb") as f:
        pickle.dump(train_emb, f)
    print(f"[info] wrote {len(train_emb)} training embeddings to {args.output_prefix}_train_emb.pkl") # save the training embeddings to a pickle file

    viral_emb = extract_for_subset(viral_df, tokenizer, model, device, args.max_len, "viral") # extract embeddings for the viral holdout
    with open(f"{args.output_prefix}_viral_emb.pkl", "wb") as f:
        pickle.dump(viral_emb, f) # .dump means "serialize" the embeddings dictionary and write it to the file; serialize means convert the Python object into a byte stream that can be saved to disk
    print(f"[info] wrote {len(viral_emb)} viral embeddings to {args.output_prefix}_viral_emb.pkl") # save the viral embeddings to a pickle file


if __name__ == "__main__":
    main()
