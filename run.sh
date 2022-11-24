#!/bin/bash

source ~/anaconda3/bin/activate nlp && python run.py \
--model_name_or_path nghuyong/ernie-3.0-xbase-zh \
--do_train \
--do_eval \
--train_file ./ChID/train_data_1w.json \
--validation_file ./ChID/dev_data.json \
--test_file ./ChID/test_data.json \
--metric_for_best_model eval_accuracy \
--load_best_model_at_end \
--learning_rate 5e-5 \
--evaluation_strategy epoch \
--num_train_epochs 5 \
--output_dir ./ernie/ernie-3.0-xbase-zh/train_1w \
--per_device_eval_batch_size 8 \
--per_device_train_batch_size 8 \
--seed 42 \
--max_seq_length 512 \
--warmup_ratio 0.1 \
--save_strategy epoch \
--overwrite_output
