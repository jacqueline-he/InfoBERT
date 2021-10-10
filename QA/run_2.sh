#!/bin/bash

NUM_GPU=8

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU run_qa.py \
    --model_name_or_path roberta-base \
    --dataset_name squad_adversarial \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./adv-results-1009-rob \
    --per_device_eval_batch_size=8   \
    --per_device_train_batch_size=8   \
    --num_train_epochs 2 