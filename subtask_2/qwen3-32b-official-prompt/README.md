# Qwen3-32B Official Prompt - Subtask 2 预测工具

基于 Qwen3-32B 模型和官方 prompt 格式的 DimASTE (Dimensional Aspect-Based Sentiment Triple Extraction) 任务预测工具。

## 📋 目录

- [功能特点](#功能特点)
- [文件结构](#文件结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [输出说明](#输出说明)
- [注意事项](#注意事项)
- [故障排除](#故障排除)

## ✨ 功能特点

- ✅ **官方 Prompt 格式**：使用官方提供的 `task2_prompt.txt` 模板
- ✅ **批量并行处理**：支持多线程并发调用 API，提高预测效率
- ✅ **自动重试机制**：API 调用失败时自动重试，提高稳定性
- ✅ **文本格式解析**：自动解析 LLM 返回的文本格式响应（非 JSON）
- ✅ **Thinking 模式支持**：可选的 thinking 模式配置
- ✅ **批量预测**：支持一次性预测全部 8 个语言-领域组合
- ✅ **详细日志**：完整的日志记录，便于调试和监控

## 📁 文件结构

```
qwen3-32b-official-prompt/
├── README.md              # 本文档
├── config.yaml            # 配置文件（API 配置、预测参数等）
├── api_client.py          # Qwen API 客户端（支持批量并行调用）
├── utils.py               # 工具函数（数据加载、保存、prompt 构建、解析）
├── predict.py             # 主预测脚本（单个语言-领域组合）
├── batch_predict_all.py   # 批量预测脚本（全部 8 个组合）
├── task2_prompt.txt       # 官方 prompt 模板文件
├── quick_test.py          # 快速测试脚本（可选）
└── predict_output/        # 预测结果输出目录（自动创建）
    ├── pred_{lang}_{domain}.jsonl          # 单个预测输出
    └── {timestamp}/                         # 批量预测输出目录
        └── pred_{lang}_{domain}.jsonl
```

## 🚀 快速开始

### 1. 环境要求

```bash
# 必需的 Python 包
pip install openai pyyaml tqdm
```

### 2. 配置设置

编辑 `config.yaml` 文件，设置 API 地址和其他参数：

```yaml
api:
  base_url: http://10.208.62.156:8002/v1   # 修改为你的 API 地址
  key: EMPTY                               # API 密钥（如无需鉴权保持 EMPTY）
  model: Qwen3-32B
```

### 3. 数据准备

确保测试数据文件位于正确路径：
```
data/raw_track_a/subtask_2/{lang}/{lang}_{domain}_dev_task2.jsonl
```

例如：
- `data/raw_track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl`
- `data/raw_track_a/subtask_2/zho/zho_restaurant_dev_task2.jsonl`

## ⚙️ 配置说明

### config.yaml 完整配置

```yaml
# 数据配置
subtask: subtask_2
lang: eng          # 语言代码: eng, zho, jpn, rus, tat, ukr
domain: laptop     # 领域: restaurant, laptop, hotel, finance
task: task2        # 任务标识

# API配置
api:
  base_url: http://10.208.62.156:8002/v1   # vLLM/OpenAI兼容接口
  key: EMPTY                               # 如无需鉴权可保持为EMPTY
  model: Qwen3-32B
  timeout: 60            # 请求超时时间（秒）
  max_retries: 3         # 最大重试次数
  retry_delay: 1.0       # 首次重试等待秒数，之后指数退避

# 预测参数
max_workers: 20          # 最大并发线程数（根据API限流调节）
max_tokens: 1000          # 最大生成token数
temperature: 0.0         # 温度参数（0-1，越小越确定，建议0.0-0.2）
top_p: 1.0               # 核采样参数（0-1，越大越随机）
presence_penalty: 0.0    # 惩罚参数（0-1，越大越惩罚）
top_k: 0                 # top-k采样参数（0-1，越大越随机）
enable_thinking: false   # 是否启用thinking模式（true/false）
```

### 配置参数说明

| 参数 | 说明 | 默认值 | 建议值 |
|------|------|--------|--------|
| `lang` | 语言代码 | `eng` | `eng`, `zho`, `jpn`, `rus`, `tat`, `ukr` |
| `domain` | 领域 | `laptop` | `restaurant`, `laptop`, `hotel`, `finance` |
| `max_workers` | 并发线程数 | `20` | 根据 API 限流调整（10-50） |
| `max_tokens` | 最大生成 token 数 | `1000` | 根据 prompt 长度调整 |
| `temperature` | 温度参数 | `0.0` | `0.0-0.2`（越小越确定） |
| `enable_thinking` | Thinking 模式 | `false` | `true`/`false` |

## 📖 使用方法

### 方法 0：快速测试（推荐首次使用）

在正式预测之前，建议先运行快速测试脚本，确保系统配置正确：

```bash
python subtask_2/llm_api_method/qwen3-32b-official-prompt/quick_test.py
```

测试脚本会检查：
- ✅ Prompt 模板加载
- ✅ 数据文件加载
- ✅ 三元组解析功能
- ✅ API 连接
- ✅ 端到端预测流程

如果所有测试通过，说明系统配置正确，可以进行正式预测。

### 方法 1：单个语言-领域组合预测

1. **修改配置文件**：编辑 `config.yaml`，设置 `lang` 和 `domain`

```yaml
lang: eng
domain: laptop
```

2. **运行预测**（在项目根目录）：

```bash
python subtask_2/llm_api_method/qwen3-32b-official-prompt/predict.py
```

3. **查看结果**：
   - 输出文件：`subtask_2/llm_api_method/qwen3-32b-official-prompt/predict_output/pred_eng_laptop.jsonl`

### 方法 2：批量预测全部 8 个组合

直接运行批量预测脚本（在项目根目录）：

```bash
python subtask_2/llm_api_method/qwen3-32b-official-prompt/batch_predict_all.py
```

**批量预测会按顺序处理以下 8 个组合：**
1. `eng-restaurant`
2. `eng-laptop`
3. `zho-restaurant`
4. `zho-laptop`
5. `jpn-hotel`
6. `rus-restaurant`
7. `tat-restaurant`
8. `ukr-restaurant`

**输出位置**：
```
subtask_2/llm_api_method/qwen3-32b-official-prompt/predict_output/{timestamp}/
├── pred_eng_restaurant.jsonl
├── pred_eng_laptop.jsonl
├── pred_zho_restaurant.jsonl
├── pred_zho_laptop.jsonl
├── pred_jpn_hotel.jsonl
├── pred_rus_restaurant.jsonl
├── pred_tat_restaurant.jsonl
└── pred_ukr_restaurant.jsonl
```

其中 `{timestamp}` 格式为 `YYYYMMDD_HHMMSS`，例如 `20241215_143022`。

## 📤 输出说明

### 输出格式

每个预测结果文件为 JSONL 格式，每行一个 JSON 对象：

```json
{"ID": "lap26_aste_dev_1", "Triplet": [{"Aspect": "performance", "Opinion": "great", "VA": "8.00#6.50"}]}
{"ID": "lap26_aste_dev_2", "Triplet": [{"Aspect": "display", "Opinion": "bright", "VA": "7.25#5.75"}]}
```

### 输出字段说明

- **ID**: 数据样本的唯一标识符
- **Triplet**: 三元组列表，每个三元组包含：
  - **Aspect**: 方面词（保持原始大小写）
  - **Opinion**: 观点词（保持原始大小写）
  - **VA**: Valence-Arousal 评分，格式为 `"V#A"`（例如 `"6.75#6.38"`）
    - V (Valence): 1.00-9.00（1.00=极度负面，9.00=极度正面）
    - A (Arousal): 1.00-9.00（1.00=非常平静，9.00=非常激动）

### 输出位置总结

| 预测方式 | 输出路径 |
|---------|---------|
| 单个预测 | `predict_output/pred_{lang}_{domain}.jsonl` |
| 批量预测 | `predict_output/{timestamp}/pred_{lang}_{domain}.jsonl` |

## ⚠️ 注意事项

### 1. 数据路径

- 确保从**项目根目录**运行脚本
- 测试数据路径：`data/raw_track_a/subtask_2/{lang}/{lang}_{domain}_dev_task2.jsonl`
- 如果数据文件不存在，程序会报错并退出

### 2. API 配置

- **base_url**: 确保 API 地址正确且可访问
- **timeout**: 根据网络情况调整超时时间（建议 60-120 秒）
- **max_workers**: 根据 API 服务器的限流策略调整并发数
  - 如果遇到限流错误，降低 `max_workers` 值
  - 如果 API 性能良好，可以适当提高

### 3. Thinking 模式

- `enable_thinking: true`: 启用 thinking 模式，模型会输出思考过程（可能增加 token 消耗）
- `enable_thinking: false`: 禁用 thinking 模式（默认，推荐用于生产环境）

### 4. Prompt 格式

- 使用官方提供的 `task2_prompt.txt` 模板
- LLM 返回格式：`[Triplet] (aspect, opinion, V#A), (aspect, opinion, V#A), ...`
- 解析函数会自动处理 `<think>...</think>` 标签（如果启用 thinking 模式）

### 5. 错误处理

- API 调用失败会自动重试（最多 3 次，指数退避）
- 解析失败的数据会记录警告日志，但不会中断整个流程
- 最终统计会显示成功/失败数量

## 🔧 故障排除

### 问题 1: 找不到数据文件

**错误信息**：
```
FileNotFoundError: 测试数据文件不存在: data/raw_track_a/subtask_2/...
```

**解决方法**：
- 确认从项目根目录运行脚本
- 检查数据文件路径是否正确
- 确认数据文件是否存在

### 问题 2: API 连接失败

**错误信息**：
```
❌ 请求失败 (尝试 1/3): Connection error
```

**解决方法**：
- 检查 `base_url` 是否正确
- 确认 API 服务是否运行
- 检查网络连接
- 尝试增加 `timeout` 值

### 问题 3: 限流错误

**错误信息**：
```
RateLimitError: Rate limit exceeded
```

**解决方法**：
- 降低 `max_workers` 值（例如从 20 改为 10）
- 增加 `retry_delay` 值
- 分批处理数据

### 问题 4: 解析失败

**错误信息**：
```
⚠️  ID xxx: 未解析到任何triplet，LLM原始输出(截断): ...
```

**解决方法**：
- 检查 LLM 输出格式是否符合预期
- 查看日志中的原始输出，确认格式是否正确
- 可能需要调整 prompt 模板

### 问题 5: 输出目录权限错误

**错误信息**：
```
PermissionError: [Errno 13] Permission denied
```

**解决方法**：
- 检查输出目录的写入权限
- 确认有足够的磁盘空间

## 📊 性能优化建议

1. **并发数调整**：
   - 根据 API 服务器性能调整 `max_workers`
   - 建议从 10 开始，逐步增加到 20-30
   - 如果遇到限流，降低并发数

2. **批量预测**：
   - 使用 `batch_predict_all.py` 一次性处理全部组合
   - 每个组合会顺序处理，避免同时占用过多资源

3. **日志级别**：
   - 生产环境可以调整日志级别为 WARNING，减少输出

## 📝 示例

### 示例 1: 预测英文餐厅数据

```yaml
# config.yaml
lang: eng
domain: restaurant
```

```bash
python subtask_2/llm_api_method/qwen3-32b-official-prompt/predict.py
```

### 示例 2: 启用 Thinking 模式

```yaml
# config.yaml
enable_thinking: true
```

### 示例 3: 调整并发数

```yaml
# config.yaml
max_workers: 10  # 降低并发数，避免限流
```

## 🧪 快速测试

使用 `quick_test.py` 可以快速测试系统的各个组件是否正常工作。

### 运行测试

在项目根目录运行：

```bash
python subtask_2/llm_api_method/qwen3-32b-official-prompt/quick_test.py
```

### 测试内容

测试脚本会依次运行以下测试：

1. **Prompt 构建测试**：测试 prompt 模板加载和构建功能
2. **数据加载测试**：测试 JSONL 数据文件加载功能
3. **三元组解析测试**：测试从模型响应中解析三元组的功能
4. **API 连接测试**：测试与 Qwen API 的连接和基本响应
5. **端到端预测测试**：测试完整的预测流程（单个样本）

### 测试说明

- 测试会尝试从 `config.yaml` 加载配置，如果配置文件不存在，会使用默认值
- 数据加载测试会尝试查找测试数据文件，如果找不到会跳过该测试
- API 连接测试需要确保 API 服务可访问
- 端到端测试会实际调用 API，可能需要一些时间

### 测试输出示例

```
============================================================
Qwen3-32B Official Prompt - 快速测试套件
============================================================

============================================================
测试 Prompt 构建
============================================================
✅ Prompt 模板加载成功 (长度: 500 字符)
...
✅ Prompt 构建成功！

============================================================
测试总结
============================================================
✓ Prompt 构建: 通过
✓ 数据加载: 通过
✓ 三元组解析: 通过
✓ API 连接: 通过
✓ 端到端预测: 通过

总计: 5/5 个测试通过

✅ 所有测试通过！系统可以使用。
```

## 🔗 相关文件

- **Prompt 模板**: `task2_prompt.txt` - 官方提供的 prompt 格式
- **快速测试**: `quick_test.py` - 用于测试 API 连接、数据加载、三元组解析等功能


---

**最后更新**: 2024-12-15
