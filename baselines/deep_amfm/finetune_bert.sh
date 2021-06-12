#!/bin/bash                                                                                                                                                                                                    
data_dir=datasets

python run_language_modeling.py \
	--train_data_file=${data_dir}/train.lm \
	--output_dir=embedding_models/full_am \
	--model_type=bert \
	--model_name_or_path=bert-base-uncased \
	--do_train \
	--do_eval \
	--eval_data_file=${data_dir}/dev.lm \
	--overwrite_output_dir \
	--per_device_train_batch_size=4 \
	--per_device_eval_batch_size=4 \
	--block_size=512 \
	--mlm

