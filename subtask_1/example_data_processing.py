"""
示例：展示单个样本经过 jsonl_to_df 和 VADataset 的处理过程
"""
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

# 加载 tokenizer（使用一个简单的模型作为示例）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = VADataset(df, tokenizer, max_len=128)

print(f"\n数据集长度: {len(dataset)}")
print(f"每个样本包含的字段: input_ids, attention_mask, labels")
print()

# 展示每个样本的详细信息
for idx in range(len(dataset)):
    sample = dataset[idx]
    print(f"\n样本 {idx}:")
    print(f"  原始输入文本: '{df.iloc[idx]['Aspect']}: {df.iloc[idx]['Text']}'")
    print(f"  input_ids 形状: {sample['input_ids'].shape}")
    print(f"  input_ids (前40个token): {sample['input_ids'][:40].tolist()}")
    print(f"  attention_mask 形状: {sample['attention_mask'].shape}")
    print(f"  attention_mask (前40个): {sample['attention_mask'][:40].tolist()}")
    print(f"  labels 形状: {sample['labels'].shape}")
    print(f"  labels 值: {sample['labels'].tolist()} (Valence={sample['labels'][0].item():.2f}, Arousal={sample['labels'][1].item():.2f})")
    
    # 解码 token IDs 查看实际文本
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
    print(f"  解码后的文本: {decoded_text[:100]}...")
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

