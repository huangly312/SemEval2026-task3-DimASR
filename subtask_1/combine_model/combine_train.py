import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import json
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datetime import datetime
import sys

# 获取项目根目录（向上两级：combine_model -> subtask_1 -> 项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# 添加父目录到路径，以便导入模块
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from model import TransformerVARegressor
from Dataset import VADataset
from utils import load_jsonl, jsonl_to_df, df_to_jsonl, set_global_seed, worker_init_fn
from evaluation import get_prd, evaluate_predictions_task1


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


def load_all_training_data(subtask="subtask_1", task="task1", project_root=None):
    """
    加载所有语言和领域的训练数据并合并
    
    Args:
        subtask: 子任务目录名
        task: 任务类型
        project_root: 项目根目录路径（如果为None则使用全局PROJECT_ROOT）
        
    Returns:
        合并后的DataFrame
    """
    # 使用全局项目根目录或传入的路径
    if project_root is None:
        project_root = PROJECT_ROOT
    project_root = os.path.abspath(project_root)
    
    all_dataframes = []
    
    print(f"\n{'='*80}")
    print("加载所有语言和领域的训练数据")
    print(f"项目根目录: {project_root}")
    print(f"{'='*80}\n")
    
    for lang, domain in LANG_DOMAIN_COMBINATIONS:
        train_path = os.path.join(project_root, "data", "raw_track_a", subtask, lang, f"{lang}_{domain}_train_{task}.jsonl")
        
        if not os.path.exists(train_path):
            print(f"警告: 文件不存在，跳过 {lang}-{domain}: {train_path}")
            continue
        
        print(f"加载: {lang}-{domain} -> {train_path}")
        train_raw = load_jsonl(train_path)
        train_df = jsonl_to_df(train_raw)
        
        # 添加语言和领域信息（可选，用于分析）
        train_df['Lang'] = lang
        train_df['Domain'] = domain
        
        all_dataframes.append(train_df)
        print(f"  数据量: {len(train_df)} 条\n")
    
    if not all_dataframes:
        raise ValueError("没有找到任何训练数据文件！")
    
    # 合并所有数据
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"{'='*80}")
    print(f"数据合并完成:")
    print(f"  总数据量: {len(combined_df)} 条")
    print(f"  语言-领域组合数: {len(LANG_DOMAIN_COMBINATIONS)}")
    print(f"{'='*80}\n")
    
    return combined_df


def train_combined_model(
    model_name="bert-base-multilingual-cased",
    model_path=None,
    use_local_only=True,
    lr=1e-5,
    epochs=10,
    batch_size=32,
    subtask="subtask_1",
    task="task1",
    output_dir=None
):
    """
    使用所有语言和领域的数据训练mBERT模型
    
    Args:
        model_name: 模型名称
        model_path: 模型路径（如果为None则使用model_name）
        use_local_only: 是否仅使用本地缓存
        lr: 学习率
        epochs: 训练轮数
        batch_size: 批次大小
        subtask: 子任务目录名
        task: 任务类型
        output_dir: 输出目录（如果为None则使用默认路径）
        
    Returns:
        best_model_path: 最优模型保存路径
        training_log_path: 训练日志保存路径
    """
    # 设置全局随机种子
    seed = 42
    set_global_seed(seed)
    
    # 使用全局项目根目录
    project_root = os.path.abspath(PROJECT_ROOT)
    
    # 设置输出目录（基于项目根目录）
    if output_dir is None:
        output_dir = os.path.join(project_root, subtask, "combine_model", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    best_model_path = os.path.join(output_dir, "best_model.pth")
    training_log_path = os.path.join(output_dir, "trainlog.json")
    
    print(f"{'='*80}")
    print(f"输出目录设置:")
    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"  最优模型: {os.path.abspath(best_model_path)}")
    print(f"  训练日志: {os.path.abspath(training_log_path)}")
    print(f"{'='*80}\n")
    
    # 加载所有训练数据
    combined_df = load_all_training_data(subtask, task, project_root=project_root)
    
    # 划分训练集和验证集
    train_df, dev_df = train_test_split(combined_df, test_size=0.1, random_state=seed)
    
    print(f"{'='*80}")
    print(f"数据集划分:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(dev_df)} 条")
    print(f"{'='*80}\n")
    
    # 加载tokenizer
    if model_path is None:
        model_path = model_name
    
    print(f"Loading tokenizer: {model_path} (local_only={use_local_only})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 创建数据集和数据加载器
    train_dataset = VADataset(train_df, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn
    )
    
    dev_dataset = VADataset(dev_df, tokenizer)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载模型和优化器
    print(f"Loading model: {model_path} (local_only={use_local_only})")
    model = TransformerVARegressor(model_path=model_path, use_local_only=use_local_only).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    print(f"Model loaded successfully\n")
    
    # 训练循环
    best_score = float('inf')
    best_epoch = 0
    training_history = []
    
    # Early Stopping 参数
    patience = 3
    patience_counter = 0
    early_stopped = False
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} on combined dataset (all languages and domains)")
    print(f"Early Stopping: patience={patience}")
    print(f"{'='*80}\n")
    
    for epoch in range(epochs):
        train_loss = model.train_epoch(train_loader, optimizer, loss_fn, device)
        val_loss = model.eval_epoch(dev_loader, loss_fn, device)
        
        pred_v, pred_a, gold_v, gold_a = get_prd(model, dev_loader, device, type="dev")
        eval_metrics = evaluate_predictions_task1(pred_a, pred_v, gold_a, gold_v)
        
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": float(f"{train_loss:.6f}"),
            "val_loss": float(f"{val_loss:.6f}"),
            "PCC_V": float(f"{eval_metrics['PCC_V']:.6f}"),
            "PCC_A": float(f"{eval_metrics['PCC_A']:.6f}"),
            "RMSE_VA": float(f"{eval_metrics['RMSE_VA']:.6f}")
        }
        training_history.append(epoch_log)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"  PCC_V: {eval_metrics['PCC_V']:.6f} | PCC_A: {eval_metrics['PCC_A']:.6f} | RMSE_VA: {eval_metrics['RMSE_VA']:.6f}")
        
        current_score = eval_metrics['RMSE_VA']
        
        # 检查是否有改进
        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存模型（使用字典格式，与predict.py兼容）
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_score': best_score,
            }, best_model_path)
            print(f"  >>> Best model saved (RMSE_VA: {best_score:.6f}) at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        
        print()
        
        # Early Stopping 检查
        if patience_counter >= patience:
            early_stopped = True
            print(f"{'='*80}")
            print(f"Early Stopping triggered!")
            print(f"连续 {patience} 个epoch验证集RMSE没有改进，停止训练")
            print(f"最优模型在第 {best_epoch} 个epoch (RMSE_VA: {best_score:.6f})")
            print(f"{'='*80}\n")
            break
    
    # 保存训练日志
    with open(training_log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'model_name': model_name,
                'model_path': model_path,
                'lr': lr,
                'epochs': epochs,
                'batch_size': batch_size,
                'combined_languages_domains': LANG_DOMAIN_COMBINATIONS
            },
            'best_score_config': {
                'best_score': float(f"{best_score:.6f}") if best_score != float('inf') else None,
                'best_epoch': best_epoch if best_epoch > 0 else None
            },
            'early_stopping': {
                'enabled': True,
                'patience': patience,
                'triggered': early_stopped,
                'actual_epochs': len(training_history)
            },
            'training_history': training_history
        }, f, indent=2, ensure_ascii=False)
    
    print(f"{'='*80}")
    if early_stopped:
        print(f"Training stopped early (Early Stopping). Best RMSE_VA: {best_score:.6f}")
        print(f"Trained for {len(training_history)}/{epochs} epochs")
    else:
        print(f"Training completed. Best RMSE_VA: {best_score:.6f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training log saved to: {training_log_path}")
    print(f"{'='*80}\n")
    
    return best_model_path, training_log_path


def predict_all_test_sets(
    best_model_path,
    model_name="bert-base-multilingual-cased",
    model_path=None,
    use_local_only=True,
    batch_size=32,
    subtask="subtask_1",
    task="task1",
    output_dir=None
):
    """
    对所有测试集进行预测并生成pred_jsonl文件
    
    Args:
        best_model_path: 最优模型路径
        model_name: 模型名称
        model_path: 模型路径（如果为None则使用model_name）
        use_local_only: 是否仅使用本地缓存
        batch_size: 批次大小
        subtask: 子任务目录名
        task: 任务类型
        output_dir: 输出目录（如果为None则使用默认路径）
        
    Returns:
        pred_paths: 所有预测文件路径列表
    """
    # 使用全局项目根目录
    project_root = os.path.abspath(PROJECT_ROOT)
    
    if output_dir is None:
        # 如果未提供输出目录，使用默认路径（这种情况不应该发生，因为main函数会创建）
        output_dir = os.path.join(project_root, subtask, "combine_model", "output")
    os.makedirs(output_dir, exist_ok=True)
    # 预测文件单独目录，与 best_model.pth 同级
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    if model_path is None:
        model_path = model_name
    
    # 检查模型文件是否存在
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"模型checkpoint文件不存在: {best_model_path}")
    
    print(f"{'='*80}")
    print(f"预测输出目录:")
    print(f"  项目根目录: {project_root}")
    print(f"  输出目录: {os.path.abspath(output_dir)}")
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
    
    # 加载最优模型权重
    print(f"Loading checkpoint from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_info = checkpoint.get('epoch', 'unknown')
    score_info = checkpoint.get('best_score', 'unknown')
    print(f"Loaded model from epoch {epoch_info} (RMSE_VA: {score_info:.6f})\n")
    
    pred_paths = []
    
    print(f"{'='*80}")
    print("开始对所有测试集进行预测")
    print(f"{'='*80}\n")
    
    # 对每个语言-领域组合进行预测
    for lang, domain in LANG_DOMAIN_COMBINATIONS:
        predict_path = os.path.join(project_root, "data", "raw_track_a", subtask, lang, f"{lang}_{domain}_dev_{task}.jsonl")
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
    print(f"所有预测完成！共生成 {len(pred_paths)} 个预测文件")
    print(f"预测文件保存在: {os.path.abspath(pred_dir)}")
    print(f"\n预测文件列表:")
    for path in pred_paths:
        print(f"  - {os.path.abspath(path)}")
    print(f"{'='*80}\n")
    
    return pred_paths


def main():
    """
    主函数：训练组合模型并对所有测试集进行预测
    """
    # 显示项目根目录信息
    print(f"{'='*80}")
    print("组合模型训练和预测")
    print(f"{'='*80}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"脚本位置: {SCRIPT_DIR}")
    
    # 检查项目根目录是否存在必要的目录
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw_track_a")
    if not os.path.exists(data_dir):
        print(f"\n错误: 数据目录不存在: {data_dir}")
        print("请确保在项目根目录下运行此脚本")
        sys.exit(1)
    
    # 配置参数
    # model_name = "bert-base-multilingual-cased"
    # model_path = "/home/gomall/models/bert-base-multilingual-cased"
    model_name = "xlm-roberta-base"
    model_path = "/home/gomall/models/xlm-roberta-base"
    # model_name = "xlm-roberta-large"
    # model_path = "/home/gomall/models/xlm-roberta-large"
    use_local_only = True
    lr = 1e-5
    epochs = 15
    batch_size = 16
    subtask = "subtask_1"
    task = "task1"
    
    # 创建唯一的输出目录（包含时间戳和配置信息）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_str = str(lr)  # 直接使用字符串表示，避免过度格式化导致的近似
    model_short_name = model_name.split('/')[-1] if model_name else "mbert"
    exp_dir_name = f"{timestamp}_{model_short_name}_lr{lr_str}_bs{batch_size}"
    output_dir = os.path.join(PROJECT_ROOT, subtask, "combine_model", "output", exp_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n配置参数:")
    print(f"  模型: {model_name}")
    print(f"  学习率: {lr}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  语言-领域组合数: {len(LANG_DOMAIN_COMBINATIONS)}")
    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"{'='*80}\n")
    
    # 训练模型
    print("=" * 80)
    print("开始训练模型")
    print("=" * 80)
    best_model_path, training_log_path = train_combined_model(
        model_name=model_name,
        model_path=model_path,
        use_local_only=use_local_only,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        subtask=subtask,
        task=task,
        output_dir=output_dir
    )
    
    # 对所有测试集进行预测
    print("=" * 80)
    print("开始对所有测试集进行预测")
    print("=" * 80)
    pred_paths = predict_all_test_sets(
        best_model_path=best_model_path,
        model_name=model_name,
        model_path=model_path,
        use_local_only=use_local_only,
        batch_size=batch_size,
        subtask=subtask,
        task=task,
        output_dir=output_dir
    )
    
    # 输出总结
    print("=" * 80)
    print("任务完成总结")
    print("=" * 80)
    print(f"最优模型路径: {best_model_path}")
    print(f"训练日志路径: {training_log_path}")
    print(f"预测文件数量: {len(pred_paths)}")
    print("\n预测文件列表:")
    for path in pred_paths:
        print(f"  - {path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

