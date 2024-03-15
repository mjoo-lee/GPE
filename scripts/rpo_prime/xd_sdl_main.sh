#!/bin/bash
#SBATCH --job-name=rpo_996xd_sdl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0229_fin_xd_sdl_oxford_pets_test.out
#SBATCH --partition=P1
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=0
SHOT=16
EPOCH=30
TRAINER=RPO_prime_sdl

# for seed in 1 2 3
#  do
#      #training
#      sh scripts/rpo_prime/xd_sdl_train.sh eurosat ${seed} ${GPU} main_final_sdl ${SHOT} ${TRAINER}
#  done         

for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 ucf101 caltech101 sun397 imagenet  
do
    for seed in 1 2 3
    do
        # evaluation
        sh scripts/rpo_prime/xd_sdl_test.sh oxford_pets ${dataset} ${seed} ${GPU} main_final_sdl ${SHOT} ${EPOCH} ${TRAINER}
    done
done
