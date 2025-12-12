"""
示例：展示单个样本经过 jsonl_to_df 和 VADataset 的处理过程
使用本地模型（mBERT 或 RoBERTa-base）查看 tokenizer 处理后的结果
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import pandas as pd
from Dataset import VADataset
from utils import jsonl_to_df
from transformers import AutoTokenizer

# 样本数据
sample_json = {
    "ID": "laptop_quad_dev_12",
    "Text": "some apps don ' t work but the play store is still in beta right now",
    "Quadruplet": [
        {"Aspect": "apps", "Category": "SOFTWARE#GENERAL", "Opinion": "NULL", "VA": "5.00#5.00"},
        {"Aspect": "play store", "Category": "SOFTWARE#GENERAL", "Opinion": "NULL", "VA": "5.00#5.00"}
    ]
}
# sample_json = {"ID": "rest16_quad_dev_11", "Text": "it was horrible .", "Quadruplet": [{"Aspect": "NULL", "Opinion": "horrible", "Category": "RESTAURANT#GENERAL", "VA": "2.38#7.88"}]}


print("=" * 80)
print("步骤 1: 原始 JSON 数据")
print("=" * 80)
print(json.dumps(sample_json, indent=2, ensure_ascii=False))
print()

# 步骤 2: 经过 jsonl_to_df 处理
print("=" * 80)
print("步骤 2: 经过 jsonl_to_df() 转换为 DataFrame")
print("=" * 80)
data = [sample_json]
df = jsonl_to_df(data)
print("\nDataFrame 内容:")
print(df.to_string())
print(f"\nDataFrame 形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"\n数据类型:")
print(df.dtypes)
print()

# 步骤 3: 展示 DataFrame 的每一行
print("=" * 80)
print("步骤 3: DataFrame 中的每一行（每个 Aspect 成为一行）")
print("=" * 80)
for idx, row in df.iterrows():
    print(f"\n行 {idx}:")
    print(f"  ID: {row['ID']}")
    print(f"  Text: {row['Text']}")
    print(f"  Aspect: {row['Aspect']}")
    print(f"  Valence: {row['Valence']} (类型: {type(row['Valence'])})")
    print(f"  Arousal: {row['Arousal']} (类型: {type(row['Arousal'])})")
print()

# 步骤 4: 经过 VADataset 处理
print("=" * 80)
print("步骤 4: 经过 VADataset 处理")
print("=" * 80)

# ========== 配置：选择要使用的本地模型 ==========
# 选项1: 使用 mBERT（多语言BERT）
# model_name = "bert-base-multilingual-cased"
# model_path = "/home/gomall/models/bert-base-multilingual-cased"  # 本地路径
# use_local_only = True

# 选项2: 使用 RoBERTa-base（取消注释以使用）
# model_name = "roberta-base"
# model_path = "/home/gomall/models/roberta-base"  # 本地路径
# use_local_only = True

# 选项3: 使用 xlm-RoBERTa-base（取消注释以使用）
model_name = "xlm-roberta-base"
model_path = "/home/gomall/models/xlm-roberta-base"  # 本地路径
use_local_only = True

# 选项4: 使用在线模型（如果本地模型不可用）
# model_name = "bert-base-uncased"
# model_path = model_name
# use_local_only = False
# ================================================

print(f"\n加载 tokenizer: {model_path} (local_only={use_local_only})")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
    print(f"✓ Tokenizer 加载成功")
    print(f"  - 模型类型: {model_name}")
    print(f"  - Vocab 大小: {len(tokenizer)}")
    print(f"  - 特殊 token: {tokenizer.special_tokens_map}")
except Exception as e:
    print(f"✗ 加载失败: {e}")
    print("尝试使用在线模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ 使用在线模型: {model_name}")

dataset = VADataset(df, tokenizer, max_len=128)

print(f"\n数据集长度: {len(dataset)}")
print(f"每个样本包含的字段: input_ids, attention_mask, labels")
print()

# 展示每个样本的详细信息
for idx in range(len(dataset)):
    sample = dataset[idx]
    aspect_text = df.iloc[idx]['Aspect']
    sentence_text = df.iloc[idx]['Text']
    
    print(f"\n样本 {idx}:")
    print(f"  原始 Aspect: '{aspect_text}'")
    print(f"  原始 Text: '{sentence_text}'")
    print(f"  组合输入: '{aspect_text}' + '{sentence_text}'")
    print()
    
    # Tokenizer 处理详情
    print(f"  【Tokenizer 处理结果】")
    print(f"  input_ids 形状: {sample['input_ids'].shape}")
    print(f"  attention_mask 形状: {sample['attention_mask'].shape}")
    
    # 获取 token_type_ids（如果有）
    if 'token_type_ids' in sample:
        print(f"  token_type_ids 形状: {sample['token_type_ids'].shape}")
        token_type_ids = sample['token_type_ids'].tolist()
        print(f"  token_type_ids (前40个): {token_type_ids[:40]}")
    else:
        print(f"  token_type_ids: 无（模型不支持或未返回）")
    
    # 显示前40个 token 的详细信息
    input_ids = sample['input_ids'].tolist()
    attention_mask = sample['attention_mask'].tolist()
    
    print(f"\n  【前40个 Token 详情】")
    print(f"  {'Index':<6} {'Token ID':<10} {'Token Text':<20} {'Attention':<10} {'Type ID':<10}")
    print(f"  {'-'*6} {'-'*10} {'-'*20} {'-'*10} {'-'*10}")
    
    for i in range(min(40, len(input_ids))):
        token_id = input_ids[i]
        token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
        attn = attention_mask[i]
        if 'token_type_ids' in sample:
            type_id = token_type_ids[i]
        else:
            type_id = 'N/A'
        print(f"  {i:<6} {token_id:<10} {token_text:<20} {attn:<10} {type_id}")
    
    # 解码完整文本
    print(f"\n  【解码结果】")
    decoded_full = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    decoded_no_special = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"  完整解码（包含特殊token）: {decoded_full}")
    print(f"  简化解码（跳过特殊token）: {decoded_no_special}")
    
    # Labels
    print(f"\n  【标签】")
    print(f"  labels 形状: {sample['labels'].shape}")
    print(f"  labels 值: {sample['labels'].tolist()}")
    print(f"    - Valence: {sample['labels'][0].item():.2f}")
    print(f"    - Arousal: {sample['labels'][1].item():.2f}")
    
    # 统计信息
    non_padding_tokens = sum(attention_mask)
    print(f"\n  【统计信息】")
    print(f"  总长度: {len(input_ids)}")
    print(f"  实际token数（非padding）: {non_padding_tokens.item()}")
    if 'token_type_ids' in sample:
        segment_0_count = sum(1 for x in token_type_ids if x == 0)
        segment_1_count = sum(1 for x in token_type_ids if x == 1)
        print(f"  段0 (Aspect) token数: {segment_0_count}")
        print(f"  段1 (Text) token数: {segment_1_count}")
    else:
        print(f"  段信息: 模型不支持 token_type_ids（如 RoBERTa）")
print()

# 步骤 5: 展示 DataLoader 中的批次格式
print("=" * 80)
print("步骤 5: DataLoader 批次格式（batch_size=2 示例）")
print("=" * 80)
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch_idx, batch in enumerate(loader):
    print(f"\n批次 {batch_idx}:")
    print(f"  input_ids 形状: {batch['input_ids'].shape}  # [batch_size, max_len]")
    print(f"  attention_mask 形状: {batch['attention_mask'].shape}  # [batch_size, max_len]")
    print(f"  labels 形状: {batch['labels'].shape}  # [batch_size, 2]")
    print(f"  labels 值:")
    for i in range(batch['labels'].shape[0]):
        print(f"    样本 {i}: Valence={batch['labels'][i][0].item():.2f}, Arousal={batch['labels'][i][1].item():.2f}")
    break  # 只显示第一个批次

