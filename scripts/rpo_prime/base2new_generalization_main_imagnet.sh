#!/bin/bash
#SBATCH --job-name=rpo_attn
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=UNLIMITED
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0311_promptattn_imagenet.out
#SBATCH --partition=laal2
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=0
SHOT=16

for dataset in imagenet
do
    for seed in 1 2 3
    do
    sh scripts/rpo_prime/base2new_train.sh ${dataset} ${seed} ${GPU} main_final_imagenet ${SHOT}
    #sh scripts/rpo_prime/base2new_test.sh ${dataset} ${seed} ${GPU} main_9_9 ${SHOT} base
    sh scripts/rpo_prime/base2new_test.sh ${dataset} ${seed} ${GPU} main_final_imagenet ${SHOT} 15 new
    done
done

