#!/bin/bash

PROCESSES=64
BRUSH=settings/my_simple_brush.myb
ROLLOUT=20
MAX_EPISODE_STEPS=10
N_SAVE_INTERVAL=30
N_SAVE_FINAL_OBS_INTERVAL=10
JIKEI_NPZ=tmp/jikei_dataset_limited.npz
EMNIST_GZ_IMAGES=tmp/emnist_gzip/emnist-letters-train-images-idx3-ubyte.gz
EMNIST_GZ_LABELS=tmp/emnist_gzip/emnist-letters-train-labels-idx1-ubyte.gz
STAYING_PENALTY=10.0
EMPTY_DRAWING_PENALTY=10.0
STAYING_PENALTY=0.0
N_UPDATE=10000000000000000000000000
N_EVAL_INTERVAL=10
LR=0.0001
GAMMA=0.99
BETA=0.001

function train () {
    PROBLEM=$1
    OUTDIR=$2
    pipenv run python train_simple.py \
        $PROCESSES \
        --problem $PROBLEM \
        --brush_info_file $BRUSH \
        --outdir results/${OUTDIR} \
        --lr $LR \
        --n_update $N_UPDATE \
        --n_eval_interval $N_EVAL_INTERVAL \
        --rollout_n $ROLLOUT \
        --gamma $GAMMA \
        --beta $BETA \
        --max_episode_steps $MAX_EPISODE_STEPS \
        --n_save_interval $N_SAVE_INTERVAL \
        --n_save_final_obs_interval $N_SAVE_FINAL_OBS_INTERVAL \
        --jikei_npz $JIKEI_NPZ \
        --emnist_gz_images $EMNIST_GZ_IMAGES \
        --emnist_gz_labels $EMNIST_GZ_LABELS \
        --empty_drawing_penalty $EMPTY_DRAWING_PENALTY \
        --staying_penalty $STAYING_PENALTY
}


PROBLEM=$1
LABEL=$2

train ${PROBLEM} ${LABEL}
