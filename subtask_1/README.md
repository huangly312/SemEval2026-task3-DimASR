# Subtask 1: Valence-Arousal Regression

## 文件结构

```
subtask_1/
├── config.yaml              # 配置文件(数据集、模型、超参数)
├── main.py                  # 主训练脚本
├── model.py                 # TransformerVARegressor模型定义
├── data_processing.py       # 数据加载和预处理
├── evaluation.py            # 评估指标计算
├── checkpoints/             # 最优模型保存目录
├── logs/                    # 训练日志保存目录
└── output/                  # 预测结果输出目录
```

## 配置说明

### `config.yaml`参数

```yaml
task: task1                  # 任务类型(task1/task2/task3)
subtask: subtask_1           # 子任务目录(subtask_1/subtask_2/subtask_3)
lang: zho                    # 语言代码(eng/zho/jpn/rus/tat/ukr)
domain: finance              # 领域(restaurant/laptop/hotel/finance)
model_name: bert-base-multilingual-cased  # HuggingFace模型名称
use_local_only: False        # True=仅使用本地缓存, False=允许联网下载
lr: 1e-5                     # 学习率
epochs: 5                    # 训练轮数
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
```

### 3. 运行训练

```bash
cd /path/to/SemEval2026-task3-DimABSA
python subtask_1/main.py
```

### 4. 训练过程

程序会自动:
- 加载训练数据并划分验证集(9:1)
- 每个epoch结束后在验证集上评估(PCC_V, PCC_A, RMSE_VA)
- 保存RMSE_VA最优的模型到`checkpoints/best_model.pth`
- 记录训练历史到`logs/training_log_{lang}_{domain}.json`
- 使用最优模型对测试集进行预测
- 保存预测结果到`output/{model_name}_pred_{lang}_{domain}.jsonl`

### 5. 输出文件

#### 最优模型: `checkpoints/best_model.pth`
```python
{
    'epoch': int,                      # 最优模型的epoch
    'model_state_dict': dict,          # 模型权重
    'optimizer_state_dict': dict,      # 优化器状态
    'best_score': float,               # 最优RMSE_VA分数
    'eval_metrics': dict               # 评估指标
}
```

#### 训练日志: `logs/training_log_{lang}_{domain}.json`
```json
{
  "config": {...},                     // 配置信息
  "best_score": 0.087654,              // 最优分数
  "training_history": [                // 每个epoch的详细指标
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

#### 预测结果: `output/{model_name}_pred_{lang}_{domain}.jsonl`
标准JSONL格式,每行一个样本:
```json
{"ID": "zho_finance_1", "Aspect_VA": [{"Aspect": "银行", "VA": "3.45#6.78"}, ...]}
```

## 模型说明

### TransformerVARegressor
- **Backbone**: 预训练的Transformer模型(BERT/XLM-RoBERTa等)
- **输入**: `[Aspect]: [Text]` (例如: "keyboard: The keyboard is good")
- **输出**: 2维向量 [Valence, Arousal]
- **损失函数**: MSE Loss
- **优化器**: AdamW

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

## 常见问题

### Q1: 数据路径错误
确保`subtask`参数与实际数据目录名称一致(如`subtask_1`而非`subtask1`)

### Q2: 模型加载失败
如果使用PyTorch 2.6+,已自动设置`weights_only=False`参数

### Q3: 网络问题
代码已配置HuggingFace镜像站(`https://hf-mirror.com`),适合中国大陆使用

### Q4: GPU/CPU切换
程序自动检测设备,优先使用GPU(CUDA),无GPU时使用CPU

## 扩展使用

### 使用其他预训练模型
修改`config.yaml`中的`model_name`:
```yaml
model_name: xlm-roberta-base          # 多语言效果更好
model_name: bert-base-chinese         # 中文专用
model_name: bert-base-cased           # 英文专用
```

### 调整超参数
```yaml
lr: 2e-5                              # 增大学习率(更快收敛,可能不稳定)
epochs: 10                            # 增加训练轮数
```

### 修改batch size
在`main.py`中修改:
```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 默认64
```

