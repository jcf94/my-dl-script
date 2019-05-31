#!/bin/bash

source /root/miniconda/bin/activate iris_env_36

# Prepare Env

python get_environment_mix.py

TF_WORKER_HOSTS=$(cat __tf_worker_hosts)
TF_TASK_INDEX=$(cat __tf_task_index)

ibstat

export RDMA_DEVICE=mlx4_0
export RDMA_DEVICE_PORT=2

export PYTHONPATH=`pwd`:$PYTHONPATH

# Ready to Run

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
       --bleu_ref=test_data/newstest2014.de"
#       --hooks=profilerhook"
#       --all_reduce_alg=hierarchical_copy

DIST_FLAGS="--worker_hosts=${TF_WORKER_HOSTS}
            --task_index=${TF_TASK_INDEX}
            --server_protocol=grpc+verbs"

# python data_download.py --data_dir=${DATA_DIR}

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python transformer_main_noestimator.py ${FLAGS} #${DIST_FLAGS}
# nvprof -o 8gpu-nvvp%p.result 
# NCCL_DEBUG=INFO
#python transformer_main.py ${FLAGS}