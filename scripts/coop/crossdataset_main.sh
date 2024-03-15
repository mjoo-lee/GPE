#!/bin/bash
#SBATCH --job-name=coop_xd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0228_coopxd_air.out
#SBATCH --partition=P2
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=0
SHOT=16
# EPOCH=50
# cfg=vit_b16_ep50_ctxv1
EPOCH=200
cfg=vit_b16_ctxv1
TRAINER=CoOp


# for seed in 1 2 3
#  do
#      #training
#      sh scripts/coop/crossdataset_train.sh fgvc_aircraft ${seed} ${GPU} ${cfg} ${SHOT} ${TRAINER}
#  done         

for dataset in eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 ucf101 caltech101 sun397 imagenet  
do
    for seed in 1 2 3
    do
        # evaluation
        sh scripts/coop/crossdataset_test.sh ${dataset} fgvc_aircraft ${seed} ${GPU} ${cfg} ${SHOT} ${EPOCH} ${TRAINER}
    done
done