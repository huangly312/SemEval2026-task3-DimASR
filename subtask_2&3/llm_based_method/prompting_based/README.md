# DimABSA Online LLM Prediction

基于在线LLM API的DimABSA预测系统（Task 2 & Task 3）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行预测

```bash
# 测试模式（仅处理前5条，输出到终端，不保存文件）
python main.py --task_type task2 --language zho --domain restaurant --test

# 正式运行（不启用thinking）
python main.py --task_type task2 --language zho --domain restaurant

# 启用thinking模式
python main.py --task_type task2 --language zho --domain restaurant --thinking

# Task 3 示例
python main.py --task_type task3 --language eng --domain laptop --thinking
```

## 参数说明

- `--task_type`: 任务类型，可选 `task2` 或 `task3`（必需）
- `--language`: 语言代码，可选 `eng`, `jpn`, `rus`, `tat`, `ukr`, `zho`（必需）
- `--domain`: 领域，可选 `restaurant`, `laptop`, `hotel`, `finance`（必需）
- `--thinking`: 启用思维链模式（可选）
- `--test`: 测试模式，只处理前5条数据并输出到终端，不保存文件（可选）

## 固定配置

- API地址: `http://10.208.62.156:8002/v1`
- 模型: `Qwen3-32B`
- max_tokens: `4096`
- temperature: `0.7`
- top_p: `0.8`
- top_k: `20`

## 输出

结果保存在 `output/subtask_{2|3}/pred_Qwen3-32B_{language}_{domain}[_thinking].jsonl`

- 不启用thinking: `pred_Qwen3-32B_zho_restaurant.jsonl`
- 启用thinking: `pred_Qwen3-32B_zho_restaurant_thinking.jsonl`

## 批量运行

批量运行中文餐厅的 Task 2 & 3（包括 thinking 和 no-thinking 两种模式）：

```bash
bash run_example.sh
```
