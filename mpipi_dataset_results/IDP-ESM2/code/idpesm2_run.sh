#!/bin/bash
#SBATCH --job-name=idp_esm2_run
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00

# REQUIRED on this cluster
export LD_LIBRARY_PATH=/apps/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH
source ~/esm2_env/bin/activate

cd ~/kestrel_se

python3 idp_esm2_virus_fixed_newdata.py --bender mpipi_results_raw.csv --out esm2_seed42_mpipi/ --models 8M 150M --seed 42
python3 idp_esm2_virus_fixed_newdata.py --bender mpipi_results_raw.csv --out esm2_seed67_mpipi/ --models 8M 150M --seed 67
python3 idp_esm2_virus_fixed_newdata.py --bender mpipi_results_raw.csv --out esm2_seed93_mpipi/ --models 8M 150M --seed 93