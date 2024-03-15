#!/bin/bash

#SBATCH --job-name=seed3           # Submit a job named "example"
#SBATCH --nodes=1                    # Using 1 node
#SBATCH --gres=gpu:1                 # Using 1 gpu
#SBATCH --time=0-12:00:00            # 1 hour time limit
#SBATCH --mem=10000MB                # Using 10GB CPU Memory
#SBATCH --output=./S-%x.%j.out       # Make a log file

eval "$(conda shell.bash hook)"
conda activate dassl

srun sh scripts/rpo_prime/base2new_generalization_main_seed3.sh 
