# Deep AM-FM Baseline

The baseline is a modified version of the deep AM-FM toolkit presented in the original paper, "Deep AM-FM: Toolkit for Automatic Dialogue Evaluation". 
<br /><br />
The baseline consists of two components, the adequacy metric (AM) and fluency metric (FM). AM evaluates the semantic aspect of system responses while FM assesses the syntactic validity of system responses.

## System Requirements

1. python 3.x
2. scipy
3. scikit-learn
4. pytorch=1.6.0
5. transformers=3.5.0

## Instructions to Run the Baseline

The fine-tuned AM and FM models can be downloaded at https://drive.google.com/file/d/1SIM-On9MEdpr6OFkWs2YmhCgqnLNJAnS/view?usp=sharing

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
data_dir=/path/to/pretraining/dialogue/dataset
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

### 3. Compute Reference-based AM-FM Scores for Turn-level Dataset
```
python compute_wr.py \
    --dataset=${name of evaluation dataset} \
    --device=${cpu or cuda} \
    --am_model_path=embedding_models/full_am \
    --fm_model_path=language_models/full_fm
```

### 4. Compute Reference-free AM-FM Scores for Turn-level Dataset
```
python compute_wor.py \
    --dataset=${name of evaluation dataset} \
    --device=${cpu or cuda} \
    --am_model_path=embedding_models/full_am \
    --fm_model_path=language_models/full_fm
```

### 5. Compute Reference-free AM-FM Scores for Dialogue-level Dataset
```
python compute_dial.py \
    --dataset=${name of evaluation dataset} \
    --device=${cpu or cuda} \
    --am_model_path=embedding_models/full_am \
    --fm_model_path=language_models/full_fm
```
Note that for the reference-free version, we consider the current response w.r.t the previous utterance. It is up to the participants to decide whether they want to incorporate the whole dialogue history, facts or any additional information.

### Reference

```
@Inbook{Zhang2021,
    author="Zhang, Chen and D'Haro, Luis Fernando and Banchs, Rafael E. and Friedrichs, Thomas and Li, Haizhou",
    title="Deep AM-FM: Toolkit for Automatic Dialogue Evaluation",
    bookTitle="Conversational Dialogue Systems for the Next Decade",
    year="2021",
    pages="53--69",
    isbn="978-981-15-8395-7",
    doi="10.1007/978-981-15-8395-7_5"
}
```
