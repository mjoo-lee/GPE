#!/bin/bash

#cd ../..

# custom config
DATA=/shared/s2/lab01/dataset/clip
TRAINER=RPO_prime_sdl

DATASET=$1
SEED=$2
GPU=$3

CFG=$4

SHOTS=$5
SUB=$6


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/rpo_prime/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
DIR=output/rpo_prime/base2new/test_${SUB}/${COMMON_DIR}

#if [ -d "$DIR" ]; then
#    echo "Oops! The results exist at ${DIR} (so skip this job)"
#else
CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/RPO_prime/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
#fi
