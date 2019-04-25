#!/bin/bash

source /root/miniconda/bin/activate iris_env_36

export PYTHONPATH=`pwd`:$PYTHONPATH

DATA_DIR=/tmp/translate_ende
VOCAB_FILE=${DATA_DIR}/vocab.ende.32768

FLAGS="--data_dir=${DATA_DIR}
       --vocab_file=${VOCAB_FILE}
       --param_set=base
       --batch_size=2048
       --num_gpus=1"

python transformer_main.py ${FLAGS}