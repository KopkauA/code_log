#!/bin/bash
#SBATCH --job-name=idp_esm2_run_viral_only
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00

# REQUIRED on this cluster
export LD_LIBRARY_PATH=/apps/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
source ~/esm2_env/bin/activate

cd ~/kestrel_se

python3 idp_esm2_viral_only_train.py --bender merged.csv --out esm2_viral_only_seed42/ --models 8M 150M --seed 42
python3 idp_esm2_viral_only_train.py --bender merged.csv --out esm2_viral_only_seed67/ --models 8M 150M --seed 67
python3 idp_esm2_viral_only_train.py --bender merged.csv --out esm2_viral_only_seed93/ --models 8M 150M --seed 93