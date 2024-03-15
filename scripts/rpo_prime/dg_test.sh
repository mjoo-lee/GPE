#!/bin/bash

#cd ../..

# custom config
DATA=/shared/s2/lab01/dataset/clip

LOAD=$1
DATASET=$2
SEED=$3
GPU=$4

CFG=$5

SHOTS=$6
LOADEP=$7
TRAINER=$8

COMMON_DIR=source_${LOAD}/${DATASET}/seed${SEED}
#MODEL_DIR=output/rpo_prime/crossdataset/train_source/${LOAD}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=/shared/s2/lab01/myungjoo/RPO_v2/output/rpo_prime_attn/base2new/train_base/imagenet/shots_16/RPO_prime/main_final_imagenet/seed${SEED}

DIR=output/rpo_prime/dg_attn/test_target/${COMMON_DIR}

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
--load-epoch ${LOADEP} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES all
#fi