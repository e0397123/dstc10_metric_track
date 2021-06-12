# Deep AM-FM Baseline

## Adequacy Metric

This component aims to assess the semantic aspect of system responses.

### Run Adequacy Evaluation

#### 1. Fine-tune BERT-base Model (finetune_bert.sh, you may customize your own training dataset)
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

##### 2. Create preprocessed training and validation data with specific training size. This step is to conduct preprocessing on the twitter dialogues.
```bash
python ../../engines/embedding_models/bert/create_raw_data.py \
  --train_file=/path/to/train.txt \
  --train_output=/path/to/processed/train/file \
  --valid_file=/path/to/valid.txt \
  --valid_output=/path/to/processed/valid/file \
  --data_size={size of your data, such as 10000}
```

##### 3. Create tfrecord pretraining data. The tfrecord data is to easier the pretraining and faster loading. 
```bash
python ../../engines/embedding_models/bert/create_pretraining_data.py \
  --input_file=/path/to/processed/train/file \
  --output_file=/path/to/processed/train/tfrecord_file \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=60 \
  --max_predictions_per_seq=9 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

##### 4. Conduct pretraining of bert model
```bash
CUDA_VISIBLE_DEVICES=1 python ../../engines/embedding_models/bert/run_pretraining.py \
  --train_input_file=/path/to/processed/train/tfrecord_file \
  --valid_input_file=/path/to/processed/valid/tfrecord_file \
  --output_dir=/path/to/save/model \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=8 \
  --max_seq_length=60 \
  --max_predictions_per_seq=9 \
  --num_train_steps=5000 \
  --max_eval_steps=100 \
  --num_warmup_steps=100 \
  --learning_rate=2e-5
```

##### 5. Feature extraction. This step is to extract fixed word-level contextualized embedding.
```bash
CUDA_VISIBLE_DEVICES=1 python ../../engines/embedding_models/bert/extract_features.py \
  --input_file=/path/to/processed/hypothesis/file \
  --output_file=/path/to/extracted/hypothesis/json/file \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=/path/to/the/trained/checkpoint \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=60 \
  --batch_size=8
```
```bash
CUDA_VISIBLE_DEVICES=1 python ../../engines/embedding_models/bert/extract_features.py \
  --input_file=/path/to/processed/reference/file \
  --output_file=/path/to/extracted/reference/json/file \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=/path/to/the/trained/checkpoint \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=60 \
  --batch_size=8
```

## Fluency Metric

This component aims to assess the syntactic validity of system responses.

## System Requirements

1. python 3.x
2. scipy
3. scikit-learn
4. pytorch=1.6.0
5. transformers=3.5.0
