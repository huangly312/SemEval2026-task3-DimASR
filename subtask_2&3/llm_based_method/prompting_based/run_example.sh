#!/bin/bash

# DimABSA Online LLM Prediction - Batch Run for Chinese Restaurant
# Tests both thinking and no-thinking modes

echo "======================================"
echo "DimABSA Predictions: ZHO Restaurant"
echo "======================================"

# Task 2 - thinking disabled
echo "[Task 2] zho_restaurant (no thinking)"
python main.py --task_type task2 --language zho --domain restaurant

# Task 2 - thinking enabled
echo "[Task 2] zho_restaurant (thinking)"
python main.py --task_type task2 --language zho --domain restaurant --thinking

# Task 3 - thinking disabled
echo "[Task 3] zho_restaurant (no thinking)"
python main.py --task_type task3 --language zho --domain restaurant

# Task 3 - thinking enabled
echo "[Task 3] zho_restaurant (thinking)"
python main.py --task_type task3 --language zho --domain restaurant --thinking

echo "======================================"
echo "All predictions completed!"
echo "Output files:"
echo "  output/subtask_2/pred_Qwen3-32B_zho_restaurant.jsonl"
echo "  output/subtask_2/pred_Qwen3-32B_zho_restaurant_thinking.jsonl"
echo "  output/subtask_3/pred_Qwen3-32B_zho_restaurant.jsonl"
echo "  output/subtask_3/pred_Qwen3-32B_zho_restaurant_thinking.jsonl"
echo "======================================"
