#!/bin/bash

python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --learning_rate 3e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir  ./results-3 \
  --max_steps 1000