#!/bin/bash

# DimABSA Task2/Task3 - Example Training and Inference Script

# Activate conda environment (modify as needed)
# conda activate dimabsa

# Set environment variables
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0

# ============================================
# Example 1: Train Task2 on Chinese Restaurant
# ============================================
echo "Training Task2 on Chinese Restaurant dataset..."

python train.py \
  --config config.yaml \
  --task_type task2 \
  --subtask subtask_2 \
  --language zho \
  --domain restaurant \
  --exclude_null

echo "Training completed!"

# ============================================
# Example 2: Run inference with trained model
# ============================================
echo "Running inference..."

python predict.py \
  --config config.yaml \
  --checkpoint checkpoints/subtask_2_zho_restaurant_no_NULL/final_checkpoint \
  --task_type task2 \
  --subtask subtask_2 \
  --language zho \
  --domain restaurant

echo "Inference completed!"

# ============================================
# Example 3: Train Task3 on English Laptop
# ============================================
# echo "Training Task3 on English Laptop dataset..."
# 
# python train.py \
#   --config config.yaml \
#   --task_type task3 \
#   --subtask subtask_3 \
#   --language eng \
#   --domain laptop \
#   --num_train_epochs 3
# 
# echo "Training completed!"
# 
# python predict.py \
#   --config config.yaml \
#   --checkpoint checkpoints/subtask_3_eng_laptop_no_NULL/final_checkpoint \
#   --task_type task3 \
#   --subtask subtask_3 \
#   --language eng \
#   --domain laptop
# 
# echo "Inference completed!"


