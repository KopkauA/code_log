#!/bin/bash
#SBATCH --job-name=kestrel_bacteria_only
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00

# REQUIRED on this cluster
export LD_LIBRARY_PATH=/apps/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
source ~/calvados_env/bin/activate

cd ~/kestrel_se

# kestrel_bacteria_only.py isn't multi-seed aware internally, so each
# seed gets its own separate command / --out dir here.
python3 kestrel_bacteria_only.py --data merged.csv --out kestrel_bact_seed42/ --seed 42
python3 kestrel_bacteria_only.py --data merged.csv --out kestrel_bact_seed67/ --seed 67
python3 kestrel_bacteria_only.py --data merged.csv --out kestrel_bact_seed93/ --seed 93
