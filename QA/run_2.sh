#!/bin/bash

NUM_GPU=8

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU run_qa.py \
    --model_name_or_path roberta-base \
    --dataset_name squad_adversarial \
    --attack_name AddSent \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir /n/fs/scratch/jh70/robertab-adv-results-addsent-2-epochs \
    --version 6 \
    --per_device_eval_batch_size=2   \
    --per_device_train_batch_size=12   \
    --num_train_epoch 2