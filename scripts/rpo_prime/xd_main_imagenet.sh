#!/bin/bash
#SBATCH --job-name=rpo_996xd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=UNLIMITED
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0229_fin_xd_imagenet3_test.out
#SBATCH --partition=laal2
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=0
SHOT=16
EPOCH=15
TRAINER=RPO_prime

# for seed in 2
#  do
#      #training
#      sh scripts/rpo_prime/xd_train.sh imagenet ${seed} ${GPU} main_final_imagenet ${SHOT} ${TRAINER}
#  done         

for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 ucf101 caltech101 sun397 imagenet  
do
    for seed in 3
    do
        # evaluation
        sh scripts/rpo_prime/xd_test.sh imagenet ${dataset} ${seed} ${GPU} main_final_imagenet ${SHOT} ${EPOCH} ${TRAINER}
    done
done
