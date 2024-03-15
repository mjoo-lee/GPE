#!/bin/bash

#cd ../..

# custom config
DATA=/shared/s2/lab01/dataset/clip
TRAINER=$6

DATASET=$1
SEED=$2
GPU=$3

CFG=$4
SHOTS=$5

DIR=output/coop/crossdataset/train_source/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
#if [ -d "$DIR" ]; then
#    echo "Oops! The results exist at ${DIR} (so skip this job)"
#else
CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES all
#fi