#!/bin/bash

source /root/miniconda/bin/activate iris_env_36

export PYTHONPATH=`pwd`:$PYTHONPATH

PARAM_SET=big

NUM_GPUS=8

DATA_DIR=wmt14_data
MODEL_DIR=model_${PARAM_SET}_run_${NUM_GPUS}
VOCAB_FILE=${DATA_DIR}/vocab.ende.32768

BATCH_SIZE_EACH=2048
BATCH_SIZE_TOTAL=$((${BATCH_SIZE_EACH} * ${NUM_GPUS}))

FLAGS="--param_set=${PARAM_SET}
       --model_dir=${MODEL_DIR}
       --data_dir=${DATA_DIR}
       --vocab_file=${VOCAB_FILE}
       --batch_size=${BATCH_SIZE_TOTAL}
       --num_gpus=${NUM_GPUS}
       --bleu_source=test_data/newstest2014.en
       --bleu_ref=test_data/newstest2014.de
       --all_reduce_alg=hierarchical_copy"
#       --hooks=profilerhook"

# python data_download.py --data_dir=${DATA_DIR}

python transformer_main.py ${FLAGS}