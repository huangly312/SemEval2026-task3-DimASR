import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import json
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from model import TransformerVARegressor
from Dataset import VADataset
from utils import load_jsonl, jsonl_to_df
from evaluation import get_prd, evaluate_predictions_task1
from utils import set_global_seed, worker_init_fn

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


def load_all_training_data(subtask="subtask_1", task="task1", lang_domain_combinations=None):
    """
    加载指定语言和领域的训练数据并合并
    
    Args:
        subtask: 子任务目录名
        task: 任务类型
        lang_domain_combinations: 语言-领域组合列表，格式为 [('lang', 'domain'), ...]
                                 如果为None，则使用所有组合（LANG_DOMAIN_COMBINATIONS）
        
    Returns:
        合并后的DataFrame
    """
    # 如果没有指定组合，使用所有组合
    if lang_domain_combinations is None:
        lang_domain_combinations = LANG_DOMAIN_COMBINATIONS
    
    all_dataframes = []
    
    print(f"\n{'='*80}")
    if lang_domain_combinations == LANG_DOMAIN_COMBINATIONS:
        print("加载所有语言和领域的训练数据")
    else:
        print(f"加载指定的语言和领域组合的训练数据（共 {len(lang_domain_combinations)} 个组合）")
    print(f"{'='*80}\n")
    
    for lang, domain in lang_domain_combinations:
        train_path = f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_train_{task}.jsonl"
        if lang == 'zho' and domain == 'restaurant':
            train_path = f"data/raw_track_a/{subtask}/{lang}/zho_restaurant_combine_sighan2024.jsonl"
        
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
    print(f"  语言-领域组合数: {len(lang_domain_combinations)}")
    print(f"{'='*80}\n")
    
    return combined_df


def train_model(config, exp_dir):
    """
    训练模型的主函数（使用所有语言和领域的数据）
    
    Args:
        config: 配置字典，包含所有训练参数
        exp_dir: 实验目录路径（由 main.py 创建）
        
    Returns:
        best_model_path: 最优模型保存路径
        training_log_path: 训练日志保存路径
        exp_dir: 实验目录路径
    """
    # 从配置中提取参数
    subtask = config["subtask"]
    task = config["task"]
    model_name = config["model_name"]
    model_path = config.get("model_path", model_name)
    use_local_only = config.get("use_local_only", False)
    lr = float(config["lr"])
    epochs = int(config["epochs"])
    batch_size = int(config.get("batch_size", 64))
    
    # 设置全局随机种子，确保实验的可复现性
    seed = 42  # 固定随机种子，确保每次运行结果一致
    set_global_seed(seed)
    
    # 使用实验目录保存模型和日志
    best_model_path = os.path.join(exp_dir, "best_model.pth")
    training_log_path = os.path.join(exp_dir, "trainlog.json")
    
    # 注意：日志记录已在 main.py 中设置，这里不需要再次设置
    # 获取语言-领域组合（如果配置中指定了，则使用指定的；否则使用所有组合）
    lang_domain_combinations = config.get("lang_domain_combinations", None)
    if lang_domain_combinations is not None:
        # 将配置中的列表转换为元组列表
        # 配置格式可能是：[['eng', 'laptop'], ['eng', 'restaurant']] 或 [['eng', 'laptop']]
        lang_domain_combinations = [tuple(combo) if isinstance(combo, list) else combo 
                                    for combo in lang_domain_combinations]
    
    # 加载训练数据
    combined_df = load_all_training_data(subtask, task, lang_domain_combinations=lang_domain_combinations)
    
    # 划分训练集和验证集
    train_df, dev_df = train_test_split(combined_df, test_size=0.1, random_state=seed)
    
    # 输出数据集大小
    print(f"{'='*80}")
    print(f"数据集划分:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(dev_df)} 条")
    print(f"{'='*80}\n")
    
    # 加载tokenizer
    print(f"Loading tokenizer: {model_path} (local_only={use_local_only})")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 创建数据集和数据加载器
    train_dataset = VADataset(train_df, tokenizer)
    # 使用 worker_init_fn 确保多进程数据加载的可复现性
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
    best_epoch = 0  # 记录最优模型对应的epoch
    training_history = []
    
    # Early Stopping 参数
    patience = 3  # 连续3个epoch没有改进则停止
    patience_counter = 0  # 当前连续没有改进的epoch数
    early_stopped = False  # 是否因为early stopping而停止
    
    # 确定使用的组合列表（用于日志记录）
    used_combinations = lang_domain_combinations if lang_domain_combinations else LANG_DOMAIN_COMBINATIONS
    is_all_combinations = (lang_domain_combinations is None or 
                          sorted(used_combinations) == sorted(LANG_DOMAIN_COMBINATIONS))
    
    print(f"\n{'='*80}")
    if is_all_combinations:
        print(f"Training {model_name} on combined dataset (all languages and domains)")
    else:
        print(f"Training {model_name} on combined dataset ({len(used_combinations)} specified language-domain combinations)")
    print(f"Early Stopping: patience={patience} (停止条件: 连续{patience}个epoch验证集RMSE无改进)")
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
            best_epoch = epoch + 1  # 更新最优模型对应的epoch
            patience_counter = 0  # 重置patience计数器
            # 保存模型（仅保存 model_state_dict）
            torch.save(model.state_dict(), best_model_path)
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
                'combined_languages_domains': used_combinations
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
    print(f"Experiment directory: {exp_dir}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training log saved to: {training_log_path}")
    print(f"{'='*80}\n")
    
    return best_model_path, training_log_path, exp_dir, used_combinations
