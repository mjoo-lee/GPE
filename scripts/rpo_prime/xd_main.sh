#!/bin/bash
#SBATCH --job-name=rpo_1212_xd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0302_1212xd_oxford_pets.out
#SBATCH --partition=P1
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=0
SHOT=16
EPOCH=30
TRAINER=RPO_prime

for seed in 1 2 3
 do
     #training
     sh scripts/rpo_prime/xd_train.sh oxford_pets ${seed} ${GPU} main_final1212 ${SHOT} ${TRAINER}
 done         

for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 ucf101 caltech101 sun397 imagenet  
do
    for seed in 1 2 3
    do
        # evaluation
        sh scripts/rpo_prime/xd_test.sh oxford_pets ${dataset}  ${seed} ${GPU} main_final1212 ${SHOT} ${EPOCH} ${TRAINER}
    done
done
