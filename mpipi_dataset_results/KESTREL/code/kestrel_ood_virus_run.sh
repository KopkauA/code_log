#!/bin/bash
#SBATCH --job-name=kestrel_ood_virus
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00

# REQUIRED on this cluster
export LD_LIBRARY_PATH=/apps/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
source ~/calvados_env/bin/activate

cd ~/kestrel_se

# kestrel_ood_virus.py isn't multi-seed aware internally, so each seed gets its own separate
python3 kestrel_geo_only.py --data mpipi_results_raw.csv --out kestrel_seed42_mpipi/ --seed 42
python3 kestrel_geo_only.py --data mpipi_results_raw.csv --out kestrel_seed67_mpipi/ --seed 67
python3 kestrel_geo_only.py --data mpipi_results_raw.csv --out kestrel_seed93_mpipi/ --seed 93
