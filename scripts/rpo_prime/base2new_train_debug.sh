#!/bin/bash

#cd ../..

# custom config
DATA=/shared/s2/lab01/dataset/clip
TRAINER=RPO_prime

DATASET=$1
SEED=$2
GPU=$3

CFG=$4
SHOTS=$5

DIR=output/debug/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}


CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES base
