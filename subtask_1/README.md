# Subtask 1: Valence-Arousal Regression

## 文件结构

```
subtask_1/
├── config.yaml              # 配置文件(数据集、模型、超参数)
├── main.py                  # 主程序入口
├── train.py                 # 训练模块
├── predict.py               # 预测模块
├── model.py                 # TransformerVARegressor模型定义
├── Dataset.py               # VADataset数据集类
├── evaluation.py            # 评估指标计算
├── utils.py                 # 工具函数(随机种子、实验目录、数据加载等)
├── example_data_processing.py  # 数据预处理示例
├── experiments/             # 实验目录(按模型/语言/领域组织)
└── README.md                # 本文档
```

## 配置说明

### `config.yaml`参数

```yaml
task: task1                  # 任务类型(task1/task2/task3)
subtask: subtask_1           # 子任务目录(subtask_1/subtask_2/subtask_3)
lang: tat                    # 语言代码(eng/zho/jpn/rus/tat/ukr)
domain: restaurant           # 领域(restaurant/laptop/hotel/finance)
model_name: bert-base-multilingual-cased  # HuggingFace模型名称
model_path:                  # (可选)模型本地路径，如果不提供则使用model_name
use_local_only: True         # True=仅使用本地缓存, False=允许联网下载
lr: 1e-5                     # 学习率
epochs: 5                    # 训练轮数
batch_size: 32               # 批次大小

# 运行模式: "train" (仅训练), "predict" (仅预测), "both" (训练+预测)
mode: both

# 预测时使用的模型checkpoint路径 (仅在mode="predict"时需要)
# 如果mode="both"，此字段可选，训练完成后会自动使用训练得到的最优模型
checkpoint_path: ""
```

### 支持的语言-领域组合

- **eng**: restaurant, laptop
- **zho**: restaurant, laptop, finance
- **jpn**: hotel, finance
- **rus**: restaurant
- **tat**: restaurant
- **ukr**: restaurant

## 使用方法

### 1. 配置环境

确保安装以下依赖:
```bash
pip install torch transformers pandas numpy scipy scikit-learn pyyaml tqdm
```

### 2. 修改配置

编辑`config.yaml`,设置目标数据集和模型参数:
```yaml
lang: zho
domain: finance
model_name: bert-base-multilingual-cased
lr: 1e-5
epochs: 5
batch_size: 32
mode: both  # 训练+预测
```

### 3. 运行程序

```bash
cd /path/to/SemEval2026/My_localcopy
python subtask_1/main.py
```

### 4. 运行模式

#### 模式1: 训练+预测 (mode: both)
```yaml
mode: both
```
- 先训练模型，训练完成后自动使用最优模型进行预测
- 训练和预测的结果都保存在同一个实验目录中

#### 模式2: 仅训练 (mode: train)
```yaml
mode: train
```
- 只训练模型，不进行预测
- 可用于训练多个模型后再统一预测

#### 模式3: 仅预测 (mode: predict)
```yaml
mode: predict
checkpoint_path: "subtask_1/experiments/bert-base-multilingual-cased/tat/restaurant/20241215_143022_1e-5_32/best_model.pth"
```
- 使用已训练好的模型进行预测
- 需要提供`checkpoint_path`指定模型路径

### 5. 训练过程

程序会自动:
- **全局日志记录**: 从程序启动开始，所有输出到终端的内容都会同时保存到日志文件
- **创建实验目录**: 在main.py中统一创建实验目录，所有相关文件（training.log, trainlog.json, best_model.pth, 预测结果）都保存在同一目录
- 设置全局随机种子(seed=42)，确保实验可复现
- 加载训练数据并划分验证集(9:1)
- 每个epoch结束后在验证集上评估(PCC_V, PCC_A, RMSE_VA)
- 如果连续3个epoch验证集RMSE无改进，触发Early Stopping停止训练
- 保存RMSE_VA最优的模型
- 记录训练历史到`trainlog.json`(结构化数据)
- 如果mode为"both"，使用最优模型对测试集进行预测

### 6. 实验目录结构

每次训练会自动创建唯一的实验目录，按以下层级组织:

```
subtask_1/
└── experiments/
    └── {model_name}/              # 模型名称
        └── {lang}/                # 语言代码
            └── {domain}/          # 领域
                └── {timestamp}_{lr}_{batch_size}/  # 时间戳_学习率_批次大小
                    ├── trainlog.json          # 训练日志(JSON格式，包含配置和指标)
                    ├── training.log          # 完整运行日志(所有终端输出，包括配置加载、训练、预测等)
                    ├── best_model.pth         # 最优模型
                    └── pred_{lang}_{domain}.jsonl  # 预测结果(如果执行了预测)

**注意**: 如果mode为"predict"（仅预测），日志文件保存在:
```
subtask_1/logs/{lang}_{domain}/predict_{timestamp}.log
```
```

**示例目录**:
```
subtask_1/experiments/bert-base-multilingual-cased/tat/restaurant/20241215_143022_1e-5_32/
```

### 7. 输出文件说明

#### 训练日志: `trainlog.json`
结构化的训练日志，包含配置、最优模型信息和每个epoch的详细指标：
```json
{
  "config": {
    "model_name": "bert-base-multilingual-cased",
    "lang": "tat",
    "domain": "restaurant",
    "lr": 1e-5,
    "epochs": 5,
    "batch_size": 32
  },
  "best_score_config": {
    "best_score": 0.123456,
    "best_epoch": 3
  },
  "early_stopping": {
    "enabled": true,
    "patience": 3,
    "triggered": false,
    "actual_epochs": 5
  },
  "training_history": [
    {
      "epoch": 1,
      "train_loss": 2.345678,
      "val_loss": 1.987654,
      "PCC_V": 0.654321,
      "PCC_A": 0.543210,
      "RMSE_VA": 0.123456
    },
    ...
  ]
}
```

#### 训练输出日志: `training.log`
包含训练过程中所有输出到终端的内容，便于查看完整的训练过程：
- 随机种子设置信息
- 数据加载和统计信息
- 每个epoch的训练和验证指标
- Early Stopping触发信息
- 模型保存信息
- 训练完成总结

示例内容：
```
Global seed set to: 42
  - CUDA deterministic mode: enabled
  - CUDA benchmark mode: disabled

Experiment directory: subtask_1/experiments/bert-base-multilingual-cased/tat/restaurant/20241215_143022_1e-5_32

================================================================================
加载训练数据: data/raw_track_a/subtask_1/tat/tat_restaurant_train_task1.jsonl
过滤数据('Aspect'='NULL')条目:5
================================================================================
数据集大小统计:
  训练集: 1116 条
  验证集: 124 条
================================================================================

Loading tokenizer: bert-base-multilingual-cased (local_only=True)
Using device: cuda

Training bert-base-multilingual-cased on tat-restaurant dataset
Early Stopping: patience=3 (停止条件: 连续3个epoch验证集RMSE无改进)
================================================================================

Epoch 1/5:
  Train Loss: 2.345678 | Val Loss: 1.987654
  PCC_V: 0.654321 | PCC_A: 0.543210 | RMSE_VA: 0.123456
  >>> Best model saved (RMSE_VA: 0.123456) at epoch 1
...
```

#### 最优模型: `best_model.pth`
```python
{
    'epoch': int,                      # 最优模型的epoch
    'model_state_dict': dict,          # 模型权重
    'optimizer_state_dict': dict,      # 优化器状态
    'best_score': float,               # 最优RMSE_VA分数
    'eval_metrics': dict               # 评估指标
}
```

#### 预测结果: `pred_{lang}_{domain}.jsonl`
标准JSONL格式,每行一个样本:
```json
{"ID": "tat_restaurant_1", "Aspect_VA": [{"Aspect": "服务", "VA": "3.45#6.78"}, ...]}
```

## 模型说明

### TransformerVARegressor
- **Backbone**: 预训练的Transformer模型(BERT/mBERT/RoBERTa/XML-RoBERTa/DeBERTa等)
- **输入**: 将aspect和text作为两个独立的段输入
  - Aspect: 方面词
  - Text: 原始文本
  - 适用于BERT、mBERT、RoBERTa、XML-RoBERTa、DeBERTa等模型
- **输出**: 2维向量 [Valence, Arousal]
- **损失函数**: MSE Loss
- **优化器**: AdamW
- **随机种子**: 固定为42，确保实验可复现

### 评估指标
- **PCC_V**: Valence的Pearson相关系数
- **PCC_A**: Arousal的Pearson相关系数
- **RMSE_VA**: Valence和Arousal的归一化均方根误差

## 离线使用

### 方法1: 首次联网下载
1. 设置`use_local_only: False`
2. 运行一次训练,模型会缓存到`~/.cache/huggingface/`
3. 后续设置`use_local_only: True`即可离线使用

### 方法2: 手动配置本地路径
```yaml
# 使用镜像站(中国大陆)
# 在代码开头已设置: os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 或指定本地snapshot路径
model_path: /path/to/huggingface/hub/models--bert-base-multilingual-cased/snapshots/[hash]
use_local_only: True
```

## 代码结构说明

### 核心模块

- **main.py**: 主程序入口，负责配置加载、训练/预测流程控制
- **train.py**: 训练模块，包含完整的训练循环、验证、模型保存等
- **predict.py**: 预测模块，在测试集上生成预测结果
- **model.py**: 模型定义，基于Transformer的回归器
- **Dataset.py**: PyTorch数据集类，处理数据加载和tokenization

### 工具模块

- **utils.py**: 工具函数集合
  - `set_global_seed()`: 设置全局随机种子
  - `create_experiment_dir()`: 创建实验目录
  - `load_jsonl()`: 加载JSONL文件
  - `jsonl_to_df()`: JSONL转DataFrame
  - `df_to_jsonl()`: DataFrame转JSONL
  - `worker_init_fn()`: DataLoader的worker初始化函数

- **evaluation.py**: 评估函数
  - `get_prd()`: 获取模型预测结果
  - `evaluate_predictions_task1()`: 计算评估指标

## 常见问题

### Q1: 数据路径错误
确保`subtask`参数与实际数据目录名称一致(如`subtask_1`而非`subtask1`)

### Q2: 模型加载失败
如果使用PyTorch 2.6+,已自动设置`weights_only=False`参数

### Q3: 网络问题
代码已配置HuggingFace镜像站(`https://hf-mirror.com`),适合中国大陆使用

### Q4: GPU/CPU切换
程序自动检测设备,优先使用GPU(CUDA),无GPU时使用CPU

### Q5: 实验可复现性
代码已设置全局随机种子(seed=42)和CUDA确定性模式,确保每次运行结果一致

## 扩展使用

### 使用其他预训练模型
修改`config.yaml`中的`model_name`:
```yaml
model_name: xlm-roberta-base          # 多语言效果更好
model_name: bert-base-chinese         # 中文专用
model_name: bert-base-cased           # 英文专用
model_name: microsoft/deberta-base    # DeBERTa模型
```

### 调整超参数
```yaml
lr: 2e-5                              # 增大学习率(更快收敛,可能不稳定)
epochs: 10                            # 增加训练轮数
batch_size: 64                        # 调整批次大小
```

### 对比不同实验
所有实验按`模型/语言/领域/参数`组织，便于：
- 对比不同超参数的效果
- 查看历史实验结果
- 复现特定配置的实验

## 注意事项

1. **实验目录**: 每次训练都会创建新的实验目录，不会覆盖之前的实验结果
2. **随机种子**: 固定为42，如需不同随机性可修改`train.py`中的seed值
3. **模型保存**: 只有验证集上表现更好的模型才会被保存为`best_model.pth`
4. **训练日志**: 每个epoch的训练记录都会保存，便于分析训练过程
