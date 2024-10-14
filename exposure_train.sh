#!/bin/bash

# python argparse source for experiments
experiments=(
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
"--exposure-factor-dim=4 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

"--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
"--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
"--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

"--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
"--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
#10/14 17:17

# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=4 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=8 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=1 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=2 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=4 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=8 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-2 --exposure-neg-size=16 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=1 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=2 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=4 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=8 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "

# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=1. --lambda-exposed-item-reg=1. --lambda-unexposed-user-reg=1. --lambda-unexposed-item-reg=1. "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=0.5 --lambda-exposed-item-reg=0.5 --lambda-unexposed-user-reg=0.5 --lambda-unexposed-item-reg=0.5 "
# "--exposure-factor-dim=16 exposure-lr=1e-3 --exposure-neg-size=16 --lambda-exposed-user-reg=2. --lambda-exposed-item-reg=2. --lambda-unexposed-user-reg=2. --lambda-unexposed-item-reg=2. "
)

# default prefix of job name
DEFAULT_NAME=exposure

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
EXECUTION_FILE=/home1/wonhyung64/Github/DRS/baselines/mf/exposure_train.py

# data directory for experiments
DATA_DIR=/home1/wonhyung64/Github/DRS/data


for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index ${DEVICES[3]} $RUN_SRC $ENV $EXECUTION_FILE --data-dir $DATA_DIR ${experiments[$index]} 
    sleep 1
done
