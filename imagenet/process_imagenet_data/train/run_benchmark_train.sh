#!/bin/bash

# tf 1.12.0 benchmarks

TRAIN_FLAGS="--data_name=imagenet
        --data_dir=$HOME/ImageNet/record-data
        --train_dir=$HOME/ImageNet/train/resnet152-2
        --num_epochs=20
        --summary_verbosity=3
        --save_summaries_steps=1000
        --save_model_steps=2000
        --print_training_accuracy=True"

MODEL_FLAGS="--num_gpus=4
        --model=resnet152
        --optimizer=rmsprop
        --batch_size=64
        --variable_update=replicated
        --all_reduce_spec=nccl"

CUDA_VISIBLE_DEVICES="0,1,2,3" python /home/jcf/benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py ${TRAIN_FLAGS} ${MODEL_FLAGS}
