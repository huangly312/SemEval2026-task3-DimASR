"""
模型集成脚本
对多个模型的预测结果进行集成（取平均值）
"""
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np

# 添加父目录到路径，以便导入utils模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from utils import load_jsonl, df_to_jsonl

# ========== 配置区域：针对不同 lang+domain 组合使用不同的模型列表 ==========
# 模型映射字典：将 (lang, domain) 组合映射到对应的模型文件夹名称列表
MODEL_MAPPING_1 = {
    # 英语组合
    ('eng', 'laptop'): [
        "large-32-1e-5-7-sigmoid",
        "large-32-2e-5-5-sigmoid",
        "large-32-1e-5-5-sigmoid"
    ],
    ('eng', 'restaurant'): [
        "large-32-8e-6-3-sigmoid",
        "large-32-1e-5-7-sigmoid"
    ],
    
    # 中文组合
    ('zho', 'restaurant'): [
        "large-32-1e-5-5-sigmoid",
        "large-32-8e-6-7",
        "large-32-1e-5-7-sigmoid"
    ],
    ('zho', 'laptop'): [
        "large-16-1e-5-7-sigmoid",
        "large-32-8e-6-7",
        "large-32-2e-5-5-sigmoid",
        "large-32-1e-5-7-sigmoid"
    ],
    ('zho', 'finance'): [
        "large-16-1e-5-7-sigmoid",
        "large-32-1e-5-7-sigmoid",
        "large-32-8e-6-7"
    ],
    
    # 日语组合
    ('jpn', 'hotel'): [
        "large-32-8e-6-7",
        "large-32-1e-5-5-sigmoid",
        "large-16-1e-5-7-sigmoid"
    ],
    ('jpn', 'finance'): [
        "large-32-1e-5-5-sigmoid",
        "large-32-1e-5-7-sigmoid",
        "large-32-8e-6-7"
    ],
    
    # 其他语言组合
    ('rus', 'restaurant'): [
        "large-32-2e-5-5-sigmoid",
        "large-16-1e-5-7-sigmoid",
        "large-32-1e-5-7-sigmoid"
    ],
    ('tat', 'restaurant'): [
        "large-32-1e-5-7-sigmoid",
        "large-32-8e-6-7"
    ],
    ('ukr', 'restaurant'): [
        "large-32-2e-5-5-sigmoid",
        "large-16-1e-5-7-sigmoid",
        "large-32-1e-5-7-sigmoid"
    ],
}

# 最优模型组合
MODEL_MAPPING = {
    ('eng', 'laptop'): ['large-32-1e-5-7-sigmoid', 'large-32-2e-5-5-sigmoid'],
    ('eng', 'restaurant'): ['large-32-1e-5-7-sigmoid', 'large-32-8e-6-3-sigmoid'],
    ('jpn', 'finance'): ['large-16-1e-5-7-sigmoid', 'large-32-1e-5-5-sigmoid', 'large-32-1e-5-7-sigmoid', 'large-32-8e-6-7'],
    ('jpn', 'hotel'): ['large-32-1e-5-5-sigmoid', 'large-32-8e-6-3-sigmoid', 'large-32-8e-6-7'],
    ('rus', 'restaurant'): ['large-16-1e-5-7-sigmoid', 'large-32-2e-5-5-sigmoid'],
    ('tat', 'restaurant'): ['large-32-1e-5-3', 'large-32-1e-5-7-sigmoid', 'large-32-8e-6-7'],
    ('ukr', 'restaurant'): ['large-16-1e-5-7-sigmoid', 'large-32-2e-5-5-sigmoid'],
    ('zho', 'finance'): ['large-16-1e-5-7-sigmoid', 'large-32-1e-5-7-sigmoid', 'large-32-8e-6-7'],
    ('zho', 'laptop'): ['large-16-1e-5-7-sigmoid', 'large-32-2e-5-5-sigmoid', 'large-32-8e-6-3-sigmoid', 'large-32-8e-6-7'],
    ('zho', 'restaurant'): ['large-32-1e-5-3', 'large-32-1e-5-5-sigmoid', 'large-32-8e-6-7'],
}

# 默认模型列表（如果某个组合没有在 MODEL_MAPPING 中定义，使用这个）
DEFAULT_MODEL_FOLDERS = [
    "large-16-1e-5-7-sigmoid",
    "large-32-1e-5-7-sigmoid"
]
# ==============================================================

# 定义所有语言-领域组合
LANG_DOMAIN_COMBINATIONS = [
    ('eng', 'laptop'),
    ('eng', 'restaurant'),
    ('zho', 'restaurant'),
    ('zho', 'laptop'),
    ('zho', 'finance'),
    ('jpn', 'hotel'),
    ('jpn', 'finance'),
    ('rus', 'restaurant'),
    ('tat', 'restaurant'),
    ('ukr', 'restaurant'),
]


def validate_model_folders_for_combination(input_dir, model_folders, lang, domain):
    """
    验证特定组合的模型文件夹是否存在，并检查是否包含对应的预测文件
    
    Args:
        input_dir: 输入目录路径
        model_folders: 模型文件夹名称列表
        lang: 语言代码
        domain: 领域名称
        
    Returns:
        valid_folders: 有效的模型文件夹路径列表
    """
    valid_folders = []
    pred_filename = f"pred_{lang}_{domain}.jsonl"
    
    for folder_name in model_folders:
        folder_path = os.path.join(input_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"  警告: 模型文件夹不存在，跳过: {folder_name}")
            continue
        
        if not os.path.isdir(folder_path):
            print(f"  警告: 不是目录，跳过: {folder_name}")
            continue
        
        # 检查是否包含对应的预测文件
        pred_file = os.path.join(folder_path, pred_filename)
        if not os.path.exists(pred_file):
            print(f"  警告: 模型文件夹 {folder_name} 缺少文件 {pred_filename}，跳过")
            continue
        
        valid_folders.append(folder_path)
    
    if not valid_folders:
        raise ValueError(f"没有找到 {lang}-{domain} 组合的有效模型文件夹！")
    
    return valid_folders


def load_predictions_from_models(model_folders, lang, domain):
    """
    从所有模型文件夹中加载指定语言-领域组合的预测结果
    
    支持两种格式：
    1. {"ID": "...", "Aspect_VA": [{"Aspect": "...", "VA": "V#A"}]}
    2. {"ID": "...", "Triplet": [{"Aspect": "...", "Opinion": "...", "VA": "V#A"}]}
    
    Args:
        model_folders: 模型文件夹路径列表
        lang: 语言代码
        domain: 领域名称
        
    Returns:
        predictions_list: 预测结果列表，每个元素是一个DataFrame
        data_format: 数据格式类型 ('Aspect_VA' 或 'Triplet')
    """
    predictions_list = []
    pred_filename = f"pred_{lang}_{domain}.jsonl"
    data_format = None
    
    for model_folder in model_folders:
        pred_path = os.path.join(model_folder, pred_filename)
        
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"预测文件不存在: {pred_path}")
        
        # 加载JSONL文件
        data = load_jsonl(pred_path)
        
        # 检测数据格式（使用第一个记录）
        if not data:
            raise ValueError(f"预测文件为空: {pred_path}")
        
        first_record = data[0]
        if 'Triplet' in first_record:
            current_format = 'Triplet'
        elif 'Aspect_VA' in first_record:
            current_format = 'Aspect_VA'
        else:
            raise ValueError(f"无法识别的数据格式: {pred_path}")
        
        # 确保所有记录使用相同的格式
        if data_format is None:
            data_format = current_format
        elif data_format != current_format:
            raise ValueError(f"数据格式不一致: 期望 {data_format}，但发现 {current_format}")
        
        # 根据格式解析数据
        rows = []
        for record in data:
            record_id = record['ID']
            
            if current_format == 'Triplet':
                triplet_list = record.get('Triplet', [])
                for item in triplet_list:
                    aspect = item['Aspect']
                    opinion = item.get('Opinion', '')  # Opinion字段可能不存在
                    va_str = item['VA']
                    # 解析VA值
                    v_str, a_str = va_str.split('#')
                    valence = float(v_str)
                    arousal = float(a_str)
                    
                    rows.append({
                        'ID': record_id,
                        'Aspect': aspect,
                        'Opinion': opinion,
                        'Valence': valence,
                        'Arousal': arousal
                    })
            else:  # Aspect_VA格式
                aspect_va_list = record.get('Aspect_VA', [])
                for item in aspect_va_list:
                    aspect = item['Aspect']
                    va_str = item['VA']
                    # 解析VA值
                    v_str, a_str = va_str.split('#')
                    valence = float(v_str)
                    arousal = float(a_str)
                    
                    rows.append({
                        'ID': record_id,
                        'Aspect': aspect,
                        'Valence': valence,
                        'Arousal': arousal
                    })
        
        # 转换为DataFrame
        df = pd.DataFrame(rows)
        
        # 确保DataFrame包含必需的列
        if current_format == 'Triplet':
            required_columns = ['ID', 'Aspect', 'Opinion', 'Valence', 'Arousal']
        else:
            required_columns = ['ID', 'Aspect', 'Valence', 'Arousal']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"预测文件格式不正确: {pred_path}")
        
        predictions_list.append(df)
    
    return predictions_list, data_format


def ensemble_predictions(predictions_list, data_format='Aspect_VA'):
    """
    对多个模型的预测结果进行集成（取平均值）
    保持与第一个输入文件完全一致的行顺序（不按 ID 重新排序）
    
    Args:
        predictions_list: 预测结果列表，每个元素是一个DataFrame
        data_format: 数据格式类型 ('Aspect_VA' 或 'Triplet')
        
    Returns:
        ensemble_df: 集成后的DataFrame（行顺序与第一个模型一致）
    """
    if not predictions_list:
        raise ValueError("预测结果列表为空")
    
    # 以第一个模型的 DataFrame 为基准，保持其行顺序（即第一个输入文件的顺序）
    base_df = predictions_list[0].copy()
    
    # 根据格式确定键
    if data_format == 'Triplet':
        key_cols = ['ID', 'Aspect', 'Opinion']
    else:
        key_cols = ['ID', 'Aspect']
    
    # 为每个模型构建 (key -> (valence, arousal)) 的映射，用于按 key 对齐取值
    model_va_maps = []
    for df in predictions_list:
        if data_format == 'Triplet':
            key_to_va = {
                (row['ID'], row['Aspect'], row.get('Opinion', '')): (row['Valence'], row['Arousal'])
                for _, row in df.iterrows()
            }
        else:
            key_to_va = {
                (row['ID'], row['Aspect']): (row['Valence'], row['Arousal'])
                for _, row in df.iterrows()
            }
        model_va_maps.append(key_to_va)
    
    # 验证各模型与第一个模型的 key 集合一致
    base_keys = set(model_va_maps[0].keys())
    for i, key_to_va in enumerate(model_va_maps[1:], 1):
        df_keys = set(key_to_va.keys())
        if base_keys != df_keys:
            missing_in_df = base_keys - df_keys
            extra_in_df = df_keys - base_keys
            error_msg = f"模型 {i+1} 的预测结果与模型 1 不一致:\n"
            if missing_in_df:
                error_msg += f"  缺少: {list(missing_in_df)[:5]}...\n"
            if extra_in_df:
                error_msg += f"  多余: {list(extra_in_df)[:5]}..."
            raise ValueError(error_msg)
    
    # 按 base_df 的行顺序遍历，对每条 key 从各模型取 VA 并求平均
    num_models = len(predictions_list)
    ensemble_valence = []
    ensemble_arousal = []
    
    for _, row in base_df.iterrows():
        if data_format == 'Triplet':
            key = (row['ID'], row['Aspect'], row.get('Opinion', ''))
        else:
            key = (row['ID'], row['Aspect'])
        v_list = [model_va_maps[m][key][0] for m in range(num_models)]
        a_list = [model_va_maps[m][key][1] for m in range(num_models)]
        v_mean = np.clip(np.round(np.mean(v_list), 2), 1.00, 9.00)
        a_mean = np.clip(np.round(np.mean(a_list), 2), 1.00, 9.00)
        ensemble_valence.append(v_mean)
        ensemble_arousal.append(a_mean)
    
    ensemble_df = base_df.copy()
    ensemble_df['Valence'] = ensemble_valence
    ensemble_df['Arousal'] = ensemble_arousal
    
    return ensemble_df


def convert_ensemble_to_jsonl(ensemble_df, data_format='Aspect_VA', output_file=None):
    """
    将集成后的DataFrame转换回JSONL格式
    
    Args:
        ensemble_df: 集成后的DataFrame
        data_format: 数据格式类型 ('Aspect_VA' 或 'Triplet')
        output_file: 输出文件路径（可选）
        
    Returns:
        records: 转换后的记录列表
    """
    records = []
    
    # 按ID分组，sort=False 保持 ID 在 DataFrame 中首次出现的顺序（即与输入文件一致）
    for record_id, group in ensemble_df.groupby('ID', sort=False):
        if data_format == 'Triplet':
            # Triplet格式
            triplets = []
            for _, row in group.iterrows():
                va_str = f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                triplet = {
                    'Aspect': row['Aspect'],
                    'Opinion': row.get('Opinion', ''),
                    'VA': va_str
                }
                triplets.append(triplet)
            record = {
                'ID': record_id,
                'Triplet': triplets
            }
        else:
            # Aspect_VA格式
            aspect_va_list = []
            for _, row in group.iterrows():
                va_str = f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                aspect_va = {
                    'Aspect': row['Aspect'],
                    'VA': va_str
                }
                aspect_va_list.append(aspect_va)
            record = {
                'ID': record_id,
                'Aspect_VA': aspect_va_list
            }
        
        records.append(record)
    
    # 如果需要保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return records


def main():
    """
    主函数：执行模型集成
    针对不同的 lang+domain 组合使用不同的模型列表
    """
    # 设置路径
    ensemble_dir = os.path.dirname(os.path.abspath(__file__))
    # input_dir = os.path.join(ensemble_dir, "input_test")
    # input_dir = os.path.join(ensemble_dir, "input_dev_1225")
    input_dir = os.path.join(ensemble_dir, "final","final_test_b")

    # output_base_dir = os.path.join(ensemble_dir, "output_test")
    # output_base_dir = os.path.join(ensemble_dir, "output_dev_1225")
    output_base_dir = os.path.join(ensemble_dir, "final_output")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_dir = os.path.join(output_base_dir, timestamp)
    output_dir = os.path.join(output_base_dir, "final_test_b_ensemble")
    # prediction_dir = os.path.join(output_dir, f"subtask_2")
    prediction_dir = os.path.join(output_base_dir, "final_test_b_ensemble")
    os.makedirs(prediction_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print("开始模型集成（针对不同组合使用不同模型列表）")
    print(f"{'='*80}")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"预测文件目录: {os.path.abspath(prediction_dir)}")
    print(f"{'='*80}\n")
    
    # 对每个语言-领域组合进行集成
    ensemble_files = []
    model_usage_log = {}  # 记录每个组合使用的模型列表
    
    for lang, domain in LANG_DOMAIN_COMBINATIONS:
        print(f"{'='*80}")
        print(f"处理: {lang}-{domain}")
        print(f"{'='*80}")
        
        # 获取该组合对应的模型列表
        combination_key = (lang, domain)
        if combination_key in MODEL_MAPPING:
            model_folder_names = MODEL_MAPPING[combination_key]
            print(f"使用配置的模型列表: {model_folder_names}")
        else:
            model_folder_names = DEFAULT_MODEL_FOLDERS
            print(f"使用默认模型列表: {model_folder_names}")
        
        try:
            # 验证该组合的模型文件夹
            valid_model_folders = validate_model_folders_for_combination(
                input_dir, model_folder_names, lang, domain
            )
            
            if not valid_model_folders:
                print(f"  ✗ 错误: 没有找到有效的模型文件夹\n")
                continue
            
            print(f"  找到 {len(valid_model_folders)} 个有效模型文件夹:")
            for folder_path in valid_model_folders:
                print(f"    - {os.path.basename(folder_path)}")
            print()
            
            # 从所有模型加载预测结果
            predictions_list, data_format = load_predictions_from_models(valid_model_folders, lang, domain)
            
            print(f"  检测到数据格式: {data_format}")
            
            # 进行集成
            ensemble_df = ensemble_predictions(predictions_list, data_format=data_format)
            
            # 保存集成结果（根据原始格式保存）
            output_file = os.path.join(prediction_dir, f"pred_{lang}_{domain}.jsonl")
            convert_ensemble_to_jsonl(ensemble_df, data_format=data_format, output_file=output_file)
            
            ensemble_files.append(f"pred_{lang}_{domain}.jsonl")
            
            # 记录该组合使用的模型列表
            model_usage_log[f"{lang}-{domain}"] = [os.path.basename(f) for f in valid_model_folders]
            
            print(f"  ✓ 完成: {output_file}")
            print(f"  使用的模型: {[os.path.basename(f) for f in valid_model_folders]}\n")
            
        except Exception as e:
            print(f"  ✗ 错误: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成日志文件
    log_data = {
        "timestamp": timestamp,
        "ensemble_info": {
            "model_mapping": model_usage_log
        },
        "prediction_files": {
            "count": len(ensemble_files),
            "files": ensemble_files
        }
    }
    
    # log_path = os.path.join(output_dir, "log.json")
    # with open(log_path, "w", encoding="utf-8") as f:
    #     json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    # 输出总结
    print(f"{'='*80}")
    print("模型集成完成")
    print(f"{'='*80}")
    print(f"成功集成: {len(ensemble_files)}/{len(LANG_DOMAIN_COMBINATIONS)} 个文件")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"预测文件目录: {os.path.abspath(prediction_dir)}")
    # print(f"日志文件: {os.path.abspath(log_path)}")
    print(f"\n预测文件列表:")
    for f in ensemble_files:
        print(f"  - {f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

