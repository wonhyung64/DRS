#!/bin/bash

# python argparse source for experiments
experiments=(
"--lr 1e-3 --weight-decay 1e-3 --batch-size 1024"
"--lr 1e-3 --weight-decay 1e-4 --batch-size 1024"
"--lr 1e-4 --weight-decay 1e-3 --batch-size 1024"
"--lr 1e-4 --weight-decay 1e-4 --batch-size 1024"
"--lr 1e-3 --weight-decay 1e-3 --batch-size 512"
"--lr 1e-3 --weight-decay 1e-4 --batch-size 512"
"--lr 1e-4 --weight-decay 1e-3 --batch-size 512"
"--lr 1e-4 --weight-decay 1e-4 --batch-size 512"
)

# default prefix of job name
DEFAULT_NAME=rec

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# virutal environment directory
ENV=/Users/wonhyung64/miniforge3/envs/rank/bin/python

# file directory of experiment ".py"
EXECUTION_FILE=/Users/wonhyung64/Github/DRS/yahoo_implicit.py

# data directory for experiments
DATA_DIR=./assets/data/v.1.2.5/initial_data_type1.json


for index in ${!experiments[*]}; do
    echo --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE --data-dir $DATA_DIR ${experiments[$index]} 
    # sbatch --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE --data-dir $DATA_DIR ${experiments[$index]}
    sleep 1
done
