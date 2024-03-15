#!/bin/bash
#SBATCH --job-name=rpo_every
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=output/0221_rpo_everyepoch_alldataset_test2.out
#SBATCH --partition=P2
set -v
set -e
set -x

eval "$(conda shell.bash hook)"
conda activate dassl

GPU=$1
SHOT=16
EPOCH=15

##Train on food101##
#for seed in 1 2 3
#do
    # for cfg in main_K24
    # do
    #     sh scripts/rpo/base2new_train.sh food101 3 ${GPU} ${cfg} ${SHOT}
    # done
#done
####################

for seed in 1 2 3; do
    for cfg in main_K24; do
        for dataset in dtd eurosat fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 sun397 ucf101 caltech101; do
            #for ((epoch=1; epoch<=${EPOCH}; epoch++)) do
                sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} main_K24 ${SHOT} 10 base
                sh scripts/rpo/base2new_test.sh ${dataset} ${seed} ${GPU} ${cfg} ${SHOT} 10 new
            #done
        done
    done
done
# sh scripts/rpo/base2new_test.sh food101 1 ${GPU} main_K24 ${SHOT} 1 base

# for seed in 1 2 3
# do
#     for cfg in main_K24
#     do
#             for EPOCH in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#             do
#                 sh scripts/rpo/base2new_test.sh food101 ${seed} ${GPU} main_K24 ${SHOT} ${EPOCH} base
#                 sh scripts/rpo/base2new_test.sh food101 ${seed} ${GPU} ${cfg} ${SHOT} ${EPOCH} new
#             done
#     done
# done