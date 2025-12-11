# DimABSA Task2/Task3 - HuggingFace 实现

基于 HuggingFace Transformers + PEFT 的 DimABSA LoRA 微调实现。


## 快速开始

### 1. 安装环境

```bash
# 创建环境
conda create -n dimabsa python=3.10 -y
conda activate dimabsa

# 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 12.1/12.2 (推荐 A100)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# (可选) 如果是 A100 且想使用 Flash Attention 2 加速
# pip install flash-attn --no-build-isolation
```

**验证 PyTorch 安装**：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# 应该输出: PyTorch: 2.1.0+cu121, CUDA: True
```

**验证安装**（可选）：
```bash
python test_setup.py  # 检查环境和依赖
```

### 2. 配置任务

编辑 `config.yaml`：

```yaml
task:
  subtask: "subtask_2"      # subtask_2 或 subtask_3
  task_type: "task2"        # task2 或 task3
  language: "zho"           # eng, jpn, rus, tat, ukr, zho
  domain: "restaurant"      # restaurant, laptop, hotel, finance
  exclude_null: true        # 是否排除 NULL aspects

model:
  model_id: "Qwen/Qwen3-4B-Instruct-2507"  # 推荐使用 Qwen3

training:
  num_train_epochs: 2
  learning_rate: 1.0e-4
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
```

### 3. 训练

```bash
# 使用配置文件
python train.py --config config.yaml

# 或命令行覆盖参数
python train.py \
  --task_type task2 \
  --language zho \
  --domain restaurant \
  --num_train_epochs 2
```

训练输出保存在：`checkpoints/{model}_{subtask}_{lang}_{domain}_{null_flag}_{date}/`

例如：`checkpoints/qwen3-4b_subtask_2_zho_restaurant_no_NULL_12-09/`

### 4. 推理

```bash
python predict.py \
  --config config.yaml \
  --checkpoint checkpoints/qwen3-4b_subtask_2_zho_restaurant_no_NULL_12-09/final_checkpoint
```

预测结果保存在：`predictions/subtask_2/pred_{model}_{lang}_{domain}_{null_flag}_{date}.jsonl`

例如：`predictions/subtask_2/pred_qwen3-4b_zho_restaurant_no_NULL_12-09.jsonl`

### 一键运行示例

```bash
bash run_example.sh  # 自动完成训练+推理
```

## 主要配置说明

### 任务切换

```yaml
# Task2: 抽取 (Aspect, Opinion, VA) 三元组
task:
  task_type: "task2"
  subtask: "subtask_2"

# Task3: 抽取 (Aspect, Category, Opinion, VA) 四元组
task:
  task_type: "task3"
  subtask: "subtask_3"
```

### 模型选择

推荐模型（Qwen3 系列）：
```yaml
model:
  model_id: "Qwen/Qwen3-4B-Instruct-2507"  # 4B 模型，速度快，效果好
  # model_id: "Qwen/Qwen3-8B"              # 8B 模型，更好的效果
  # model_id: "Qwen/Qwen2.5-7B-Instruct"   # Qwen2.5 7B（较旧版本）
```

**注意**：
- Qwen3-4B 推荐用于快速实验和资源受限场景
- Qwen3-8B 效果更好，但需要更多显存（~16GB 训练）
- 第一次运行会自动下载模型到 `~/.cache/huggingface/`

### 使用本地模型

如果已经下载了模型到本地，可以避免重复下载：

**方式1：配置文件**
```yaml
model:
  use_local_model: true
  local_model_path: "/path/to/Qwen3-4B-Instruct-2507"
```

**方式2：命令行参数**
```bash
python train.py \
  --use_local_model \
  --local_model_path /path/to/Qwen3-4B-Instruct-2507
```

### 量化配置

```yaml
model:
  quantization:
    load_in_4bit: true              # 4bit 量化，节省显存
    bnb_4bit_quant_type: "nf4"      # NF4 量化类型
    bnb_4bit_compute_dtype: "bfloat16"
```

### LoRA 参数

```yaml
model:
  lora:
    r: 16                # LoRA rank，越大效果越好但越慢
    lora_alpha: 16       # 通常等于 r
    lora_dropout: 0.05
```

### 训练参数

```yaml
training:
  num_train_epochs: 2                    # 训练轮数
  per_device_train_batch_size: 1         # Batch size
  gradient_accumulation_steps: 4         # 梯度累积（有效 batch size = 1*4 = 4）
  learning_rate: 1.0e-4                  # 学习率
  bf16: true                             # 使用 BF16 混合精度
  gradient_checkpointing: true           # 节省显存
```

## 测试功能

### 快速测试（不训练）

```bash
# 测试数据加载和模型初始化
python quick_test.py

# 跳过模型加载（更快）
python quick_test.py --skip_model
```

### 小规模测试训练

修改 `config.yaml` 进行快速验证：
```yaml
training:
  num_train_epochs: 1          # 只训练1轮
  save_steps: 50               # 每50步保存
  logging_steps: 10            # 每10步打印日志
```

## 多 GPU 训练(未测试)

编辑 `train_distributed.sh` 设置 GPU 数量：
```bash
NUM_GPUS=2  # 使用2张卡
```

然后运行：
```bash
bash train_distributed.sh
```

或使用 accelerate：
```bash
accelerate launch --num_processes 2 train.py --config config.yaml
```

## 支持的任务和数据

### 任务类型
- **Task 2**: Aspect-Opinion-VA 三元组抽取
  - 输出格式：`(thai food, average to good, 6.75#6.38)`
  
- **Task 3**: Aspect-Category-Opinion-VA 四元组抽取
  - 输出格式：`(thai food, FOOD#QUALITY, average to good, 6.75#6.38)`
  - 包含领域特定的实体和属性标签

### 语言和领域
- **语言**: eng, jpn, rus, tat, ukr, zho
- **领域**: restaurant, laptop, hotel, finance

### 数据格式

输入 JSONL 格式：
```json
{"ID": "1", "Text": "食物很好吃但是服务一般", "Triplet": [...]}
```

输出 JSONL 格式：
```json
{"ID": "1", "Text": "...", "Triplet": [{"Aspect": "食物", "Opinion": "好吃", "VA": "7.5#6.2"}]}
```


## 文件说明

```
hf_based/
├── config.yaml              # 配置文件 ← 主要修改这个
├── train.py                 # 训练脚本
├── predict.py               # 推理脚本
├── test_setup.py            # 环境测试
├── quick_test.py            # 功能测试
├── run_example.sh           # 一键运行示例
├── train_distributed.sh     # 多卡训练脚本
├── requirements.txt         # Python 依赖
├── utils/
│   ├── data_processor.py    # 数据处理
│   └── model_utils.py       # 模型加载
├── prompt/
│   ├── task2_prompt.txt           # Task2 提示词
│   └── task3_prompt_template.txt  # Task3 提示词模板
├── data/                    # 数据目录（已有）
├── checkpoints/             # 训练输出（运行时创建）
└── predictions/             # 推理输出（运行时创建）
```

## 命令速查

```bash
# 环境检查
python test_setup.py

# 快速测试
python quick_test.py --skip_model

# 训练（单卡）
python train.py --config config.yaml

# 训练（多卡）
bash train_distributed.sh

# 推理
python predict.py --checkpoint checkpoints/xxx/final_checkpoint

# 一键运行
bash run_example.sh
```
