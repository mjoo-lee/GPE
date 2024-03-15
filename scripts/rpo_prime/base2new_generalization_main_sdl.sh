#!/bin/bash
#SBATCH --job-name=rpo_sdl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0229_sdl.out
#SBATCH --partition=P1
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=0
SHOT=16

for dataset in eurosat dtd fgvc_aircraft oxford_flowers
do
    for seed in 1 2 3
    do
    sh scripts/rpo_prime/base2new_train_sdl.sh ${dataset} ${seed} ${GPU} main_tmp1_0.1sdl ${SHOT}
    #sh scripts/rpo_prime/base2new_test.sh ${dataset} ${seed} ${GPU} main_9_9 ${SHOT} base
    sh scripts/rpo_prime/base2new_test_sdl.sh ${dataset} ${seed} ${GPU} main_tmp1_0.1sdl ${SHOT} new
    done
done
