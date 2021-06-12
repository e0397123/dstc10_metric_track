# Deep AM-FM Baseline

## System Requirements

1. python 3.x
2. scipy
3. scikit-learn
4. pytorch=1.6.0
5. transformers=3.5.0

## Instructions to Run the Baseline

This component aims to assess the semantic aspect of system responses.

### 1. Fine-tune BERT for Adequacy Metric
```
#!/bin/bash                                                                                                                                                                                                    
data_dir=/path/to/pretraining/dialogue/dataset

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
```

### 2. Fine-tune GPT-2 for Fluency Metric
```
#!/bin/bash                                                                                                                                                                                                    
data_dir=datasets

python run_language_modeling.py \
	--train_data_file=${data_dir}/train.lm \
	--output_dir=language_models/full_fm \
	--model_type=gpt2 \
	--model_name_or_path=gpt2 \
	--do_train \
	--do_eval \
	--eval_data_file=${data_dir}/dev.lm \
	--overwrite_output_dir \
	--per_device_train_batch_size=4 \
	--per_device_eval_batch_size=4 \
	--block_size=512
```