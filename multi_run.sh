#!/bin/bash

# python argparse source for experiments
experiments=(
"--lr 1e-5 --weight-decay 1e-4 --embedding-k 64 --batch-size 64 --pref-update-interval 25"
"--lr 1e-5 --weight-decay 1e-4 --embedding-k 64 --batch-size 128 --pref-update-interval 25"
"--lr 1e-5 --weight-decay 1e-4 --embedding-k 64 --batch-size 256 --pref-update-interval 25"
"--lr 1e-5 --weight-decay 1e-4 --embedding-k 64 --batch-size 512 --pref-update-interval 25"
"--lr 1e-5 --weight-decay 1e-4 --embedding-k 64 --batch-size 1024 --pref-update-interval 25"
"--lr 1e-5 --weight-decay 1e-4 --embedding-k 64 --batch-size 2048 --pref-update-interval 25"
"--lr 1e-5 --weight-decay 1e-4 --embedding-k 64 --batch-size 4096 --pref-update-interval 25"
"--lr 1e-5 --weight-decay 1e-4 --embedding-k 64 --batch-size 8192 --pref-update-interval 25"
# "--dataset-name yahoo_r3"
)

# default prefix of job name
DEFAULT_NAME=rec

# DEVICE SETTING
DEVICES=(
    "--partition=hgx --gres=gpu:hgx:1 "
    "--partition=gpu1 --gres=gpu:rtx3090:1 "
    "--partition=gpu2 --gres=gpu:a10:1 "
    )

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# virutal environment directory
ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3

# file directory of experiment ".py"
EXECUTION_FILE=/home1/wonhyung64/Github/DRS/yahoo_implicit.py                   #OURS
# EXECUTION_FILE=/home1/wonhyung64/Github/DRS/baselines/ncf/yahoo_implicit.py   #NCF
# EXECUTION_FILE=/home1/wonhyung64/Github/DRS/baselines/biser/yahoo_implicit.py   #biser

# data directory for experiments
DATA_DIR=/home1/wonhyung64/Github/DRS/data


for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index ${DEVICES[2]} $RUN_SRC $ENV $EXECUTION_FILE --data-dir $DATA_DIR ${experiments[$index]} 
    sleep 1
done
