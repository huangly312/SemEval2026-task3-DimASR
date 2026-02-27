import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import TransformerVARegressor
from Dataset import VADataset
from utils import load_jsonl, jsonl_to_df, df_to_jsonl
from evaluation import get_prd
from train import LANG_DOMAIN_COMBINATIONS


def predict_all_test_sets(config, best_model_path, exp_dir=None, output_dir=None, lang_domain_combinations=None):
    """
    对指定的测试集进行预测并生成pred_jsonl文件
    
    Args:
        config: 配置字典，包含所有参数
        best_model_path: 最优模型路径
        exp_dir: 实验目录路径（可选）
        output_dir: 输出目录路径（可选），如果提供则优先保存到该目录
        lang_domain_combinations: 语言-领域组合列表，格式为 [('lang', 'domain'), ...]
                                 如果为None，则使用所有组合（LANG_DOMAIN_COMBINATIONS）
        
    Returns:
        pred_paths: 所有预测文件路径列表
    """
    # 从配置中提取参数
    subtask = config["subtask"]
    task = config["task"]
    model_name = config["model_name"]
    model_path = config.get("model_path", model_name)
    use_local_only = config.get("use_local_only", False)
    batch_size = int(config.get("batch_size", 64))
    
    # 如果没有指定组合，使用所有组合
    if lang_domain_combinations is None:
        lang_domain_combinations = LANG_DOMAIN_COMBINATIONS
    
    # 确定输出目录（优先级：output_dir > exp_dir/predictions > 默认路径）
    if output_dir:
        pred_dir = os.path.join(output_dir, "predictions")
    elif exp_dir:
        pred_dir = os.path.join(exp_dir, "predictions")
    else:
        pred_dir = os.path.join(subtask, "output", "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"模型checkpoint文件不存在: {best_model_path}")
    
    print(f"{'='*80}")
    print(f"预测输出目录:")
    print(f"  输出目录: {os.path.abspath(pred_dir)}")
    print(f"{'='*80}\n")
    
    # 加载tokenizer
    print(f"\n{'='*80}")
    print(f"Loading tokenizer: {model_path} (local_only={use_local_only})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 加载模型
    print(f"Loading model: {model_path} (local_only={use_local_only})")
    model = TransformerVARegressor(model_path=model_path, use_local_only=use_local_only).to(device)
    
    # 加载最优模型权重（仅包含 model_state_dict）
    print(f"Loading checkpoint from: {best_model_path}")
    state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights successfully\n")
    
    pred_paths = []
    
    # 判断是否使用所有组合
    is_all_combinations = (sorted(lang_domain_combinations) == sorted(LANG_DOMAIN_COMBINATIONS))
    
    print(f"{'='*80}")
    if is_all_combinations:
        print("开始对所有测试集进行预测")
    else:
        print(f"开始对指定的测试集进行预测（共 {len(lang_domain_combinations)} 个组合）")
    print(f"{'='*80}\n")
    
    # 对每个语言-领域组合进行预测
    for lang, domain in lang_domain_combinations:
        predict_path = f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_dev_{task}.jsonl"
        save_path = os.path.join(pred_dir, f"pred_{lang}_{domain}.jsonl")
        
        if not os.path.exists(predict_path):
            print(f"警告: 测试文件不存在，跳过 {lang}-{domain}: {predict_path}")
            continue
        
        print(f"预测: {lang}-{domain}")
        print(f"  输入: {predict_path}")
        
        # 加载测试数据
        predict_raw = load_jsonl(predict_path)
        predict_df = jsonl_to_df(predict_raw)
        print(f"  测试集大小: {len(predict_df)} 条")
        
        # 进行预测
        pred_dataset = VADataset(predict_df, tokenizer)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
        pred_v, pred_a = get_prd(model, pred_loader, device, type="pred")
        
        # 保存预测结果
        predict_df["Valence"] = pred_v
        predict_df["Arousal"] = pred_a
        df_to_jsonl(predict_df, save_path)
        
        print(f"  输出: {save_path}\n")
        pred_paths.append(save_path)
    
    print(f"{'='*80}")
    if is_all_combinations:
        print(f"所有预测完成！共生成 {len(pred_paths)} 个预测文件")
    else:
        print(f"指定组合的预测完成！共生成 {len(pred_paths)} 个预测文件")
    print(f"预测文件保存在: {os.path.abspath(pred_dir)}")
    print(f"\n预测文件列表:")
    for path in pred_paths:
        print(f"  - {os.path.abspath(path)}")
    print(f"{'='*80}\n")
    
    return pred_paths

