import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import TransformerVARegressor
from Dataset import VADataset
from utils import load_jsonl, jsonl_to_df, df_to_jsonl
from evaluation import get_prd


def predict_on_test_set(config, best_model_path=None, exp_dir=None):
    """
    在测试集上进行预测
    
    Args:
        config: 配置字典，包含所有参数
        best_model_path: 最优模型的保存路径（可选，如果提供则优先使用）
                        如果不提供，则从config中读取checkpoint_path
        exp_dir: 实验目录路径（可选），如果提供则保存到该目录
        
    Returns:
        save_path: 预测结果保存路径
    """
    # 从配置中提取参数
    subtask = config["subtask"]
    task = config["task"]
    lang = config["lang"]
    domain = config["domain"]
    model_name = config["model_name"]
    model_path = config.get("model_path", model_name)
    use_local_only = config.get("use_local_only", False)
    lr = float(config["lr"])
    epochs = int(config["epochs"])
    batch_size = int(config.get("batch_size", 64))
    
    # 确定使用的checkpoint路径
    if best_model_path is not None:
        checkpoint_path = best_model_path
    else:
        checkpoint_path = config.get("checkpoint_path", "")
        if not checkpoint_path:
            raise ValueError(
                "预测模式需要提供模型checkpoint路径。\n"
                "请在config.yaml中设置checkpoint_path字段，或通过best_model_path参数传入。"
            )
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型checkpoint文件不存在: {checkpoint_path}")
    
    # 构建路径
    predict_path = f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_dev_{task}.jsonl"
    
    # 如果提供了实验目录，保存到实验目录；否则保存到默认输出目录
    if exp_dir:
        save_path = os.path.join(exp_dir, f"pred_{lang}_{domain}.jsonl")
    else:
        save_path = f"{subtask}/output/pred_{lang}_{domain}({model_name}-{lr:.1e}-{epochs}).jsonl"
        os.makedirs(f"{subtask}/output", exist_ok=True)
    
    # 加载测试数据
    print(f"\n{'='*80}")
    print(f"加载测试数据: {predict_path}")
    predict_raw = load_jsonl(predict_path)
    predict_df = jsonl_to_df(predict_raw)
    print(f"测试集大小: {len(predict_df)} 条")
    print(f"{'='*80}\n")
    
    # 加载tokenizer
    print(f"Loading tokenizer: {model_path} (local_only={use_local_only})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 加载模型
    print(f"Loading model: {model_path} (local_only={use_local_only})")
    model = TransformerVARegressor(model_path=model_path, use_local_only=use_local_only).to(device)
    
    # 加载最优模型权重
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_info = checkpoint.get('epoch', 'unknown')
    score_info = checkpoint.get('best_score', 'unknown')
    print(f"Loaded model from epoch {epoch_info} (RMSE_VA: {score_info:.6f})\n")
    
    # 在测试集上预测
    print("Generating predictions on test set...")
    pred_dataset = VADataset(predict_df, tokenizer)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
    pred_v, pred_a = get_prd(model, pred_loader, device, type="pred")
    
    # 保存预测结果
    predict_df["Valence"] = pred_v
    predict_df["Arousal"] = pred_a
    
    df_to_jsonl(predict_df, save_path)
    print(f"Predictions saved to: {save_path}\n")
    
    return save_path

