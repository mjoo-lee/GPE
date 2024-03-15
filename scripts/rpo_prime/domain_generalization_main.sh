#!/bin/bash
#SBATCH --job-name=rpo_prime_dg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0312_dg_attn_15.out
#SBATCH --partition=P1
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=0
SHOT=16
EPOCH=15
TRAINER=RPO_prime

for seed in 1 2 3
do
    #sh scripts/rpo_prime/dg_train.sh imagenet ${seed} ${GPU} main_final_imagenet ${SHOT} ${TRAINER}
    for dataset in imagenet imagenet_a imagenet_r imagenet_sketch imagenetv2 
    do 
        sh scripts/rpo_prime/dg_test.sh imagenet ${dataset} ${seed} ${GPU} main_final_imagenet ${SHOT} ${EPOCH} ${TRAINER}
    done
done
