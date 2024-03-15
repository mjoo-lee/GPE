#!/bin/bash
#SBATCH --job-name=rpo_996
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0227_rpo996_softmaxmean3.out
#SBATCH --partition=P2
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=0
SHOT=16

for dataset in caltech101 sun397 imagenet
#for dataset in caltech101 imagenet
do
    for seed in 1 2 3
    do
    sh scripts/rpo_prime/base2new_train.sh ${dataset} ${seed} ${GPU} main_tmp ${SHOT}
    #sh scripts/rpo_prime/base2new_test.sh ${dataset} ${seed} ${GPU} main_9_9 ${SHOT} base
    sh scripts/rpo_prime/base2new_test.sh ${dataset} ${seed} ${GPU} main_tmp ${SHOT} new
    done
done

