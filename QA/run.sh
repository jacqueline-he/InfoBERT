#!/bin/bash

NUM_GPU=2

PORT_ID=$(expr $RANDOM + 1000)

export OMP_NUM_THREADS=4

python -m torch.distributed.launch --nproc_per_node=$NUM_GPU  --master_port=$PORT_ID run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./results \
    --per_device_eval_batch_size=1   \
    --per_device_train_batch_size=1   