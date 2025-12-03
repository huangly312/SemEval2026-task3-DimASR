#!/bin/bash
python "run_task2&3_trainer_multilingual.py" \
  --task 2 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task2.jsonl \
  --bert_model_type bert-base-multilingual-uncased \
  --mode train \
  --exclude_NULL True \



python "run_task2&3_trainer_multilingual.py" \
  --task 3 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task3.jsonl \
  --bert_model_type bert-base-multilingual-uncased \
  --mode inference \
  --exclude_NULL True \