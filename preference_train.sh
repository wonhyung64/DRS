#!/bin/bash

# python argparse source for experiments
experiments=(
"--preference-factor-dim=4 --preference-lr=1e-2 "
"--preference-factor-dim=4 --preference-lr=1e-3 "
"--preference-factor-dim=4 --preference-lr=1e-4 "

"--preference-factor-dim=8 --preference-lr=1e-2 "
"--preference-factor-dim=8 --preference-lr=1e-3 "
"--preference-factor-dim=8 --preference-lr=1e-4 "

"--preference-factor-dim=16 --preference-lr=1e-2 "
"--preference-factor-dim=16 --preference-lr=1e-3 "
"--preference-factor-dim=16 --preference-lr=1e-4 "

"--preference-factor-dim=32 --preference-lr=1e-2 "
"--preference-factor-dim=32 --preference-lr=1e-3 "
"--preference-factor-dim=32 --preference-lr=1e-4 "

"--preference-factor-dim=64 --preference-lr=1e-2 "
"--preference-factor-dim=64 --preference-lr=1e-3 "
"--preference-factor-dim=64 --preference-lr=1e-4 "
)

# default prefix of job name
DEFAULT_NAME=preference

# DEVICE SETTING
DEVICES=(
    "--partition=hgx --gres=gpu:hgx:1 "
    "--partition=gpu1 --gres=gpu:rtx3090:1 "
    "--partition=gpu2 --gres=gpu:a10:1 "
    "--partition=gpu4 --gres=gpu:a6000:1 "
    "--partition=gpu5 --gres=gpu:a6000:1 "
    )

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# virutal environment directory
ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python3

# file directory of experiment ".py"
EXECUTION_FILE=/home1/wonhyung64/Github/DRS/preference_train.py

# data directory for experiments
DATA_DIR=/home1/wonhyung64/Github/DRS/data


for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index ${DEVICES[3]} $RUN_SRC $ENV $EXECUTION_FILE --data-dir $DATA_DIR ${experiments[$index]} 
    sleep 1
done
