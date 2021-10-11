#!/bin/bash

python run_qa.py \
  --model_name_or_path roberta-base \
  --dataset_name squad_adversarial \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 2 \
  --learning_rate 3e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir  /n/fs/scratch/jh70/adv-results-rb-aos \
  --version 6 \
  --max_steps 100