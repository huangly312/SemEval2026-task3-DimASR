#!/bin/bash

# Distributed Training Script for DimABSA using HuggingFace Accelerate

# ============================================
# Multi-GPU Training with accelerate
# ============================================

# Number of GPUs to use
NUM_GPUS=2

# Set environment variables
export HF_ENDPOINT="https://hf-mirror.com"

# Run with accelerate for distributed training
accelerate launch \
  --num_processes $NUM_GPUS \
  --multi_gpu \
  --mixed_precision bf16 \
  train.py \
  --config config.yaml \
  --task_type task2 \
  --subtask subtask_2 \
  --language zho \
  --domain restaurant \
  --exclude_null

# ============================================
# Alternative: Use torch.distributed.launch (PyTorch native)
# ============================================
# python -m torch.distributed.launch \
#   --nproc_per_node=$NUM_GPUS \
#   --master_port=29500 \
#   train.py \
#   --config config.yaml \
#   --task_type task2 \
#   --subtask subtask_2 \
#   --language zho \
#   --domain restaurant

# ============================================
# For specific GPU selection
# ============================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --num_processes 4 train.py --config config.yaml


