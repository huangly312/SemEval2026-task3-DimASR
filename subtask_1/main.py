import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import yaml
import sys
from datetime import datetime
from train import train_model, LANG_DOMAIN_COMBINATIONS
from predict import predict_all_test_sets
from utils import Tee


def load_config(config_path="subtask_1/config.yaml"):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"Configuration loaded successfully: {config}\n")
    return config


def main():
    """
    主函数：训练组合模型并对测试集进行预测
    默认使用所有语言和领域的数据，也可以通过配置指定特定的组合
    """
    # 加载配置
    config = load_config()
    
    # 从配置中提取参数
    subtask = config["subtask"]
    model_name = config["model_name"]
    lr = float(config["lr"])
    batch_size = int(config.get("batch_size", 64))
    
    # 获取语言-领域组合（如果配置中指定了，则使用指定的；否则使用所有组合）
    lang_domain_combinations = config.get("lang_domain_combinations", None)
    if lang_domain_combinations is not None:
        # 将配置中的列表转换为元组列表
        # 配置格式可能是：[['eng', 'laptop'], ['eng', 'restaurant']]
        lang_domain_combinations = [tuple(combo) if isinstance(combo, list) else combo 
                                    for combo in lang_domain_combinations]
        # 如果和所有组合相同，则设为None使用默认
        if sorted(lang_domain_combinations) == sorted(LANG_DOMAIN_COMBINATIONS):
            lang_domain_combinations = None
    
    # 创建唯一的输出目录（包含时间戳和配置信息）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_str = str(lr)  # 直接使用字符串表示，避免过度格式化导致的近似
    model_short_name = model_name.split('/')[-1] if model_name else "mbert"
    exp_dir_name = f"{timestamp}_{model_short_name}_lr{lr_str}_bs{batch_size}"
    exp_dir = os.path.join(subtask, "experiments", exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 日志文件路径
    log_file_path = os.path.join(exp_dir, "training.log")
    
    print(f"\n配置参数:")
    print(f"  模型: {model_name}")
    print(f"  学习率: {lr}")
    print(f"  批次大小: {batch_size}")
    if lang_domain_combinations:
        print(f"  指定组合数: {len(lang_domain_combinations)}")
    else:
        print(f"  使用所有组合: 10 个语言-领域组合")
    print(f"  输出目录: {os.path.abspath(exp_dir)}")
    print(f"{'='*80}\n")
    
    # 设置全局日志记录：所有输出同时写入文件和终端
    with Tee(log_file_path, mode='w', encoding='utf-8'):
        print(f"日志文件: {log_file_path}")
        print("=" * 80)
        
        # 执行训练
        print("=" * 80)
        if lang_domain_combinations:
            print(f"开始训练模型（使用指定的 {len(lang_domain_combinations)} 个语言-领域组合）")
        else:
            print("开始训练模型（使用所有语言和领域的数据）")
        print("=" * 80)
        best_model_path, training_log_path, exp_dir, used_combinations = train_model(config, exp_dir)
        print(f"训练完成！最优模型保存在: {best_model_path}\n")
        
        # 执行预测（对指定的测试集进行预测）
        print("=" * 80)
        is_all_combinations = (sorted(used_combinations) == sorted(LANG_DOMAIN_COMBINATIONS))
        if is_all_combinations:
            print("开始对所有测试集进行预测")
        else:
            print(f"开始对指定的测试集进行预测（{len(used_combinations)} 个组合）")
        print("=" * 80)
        pred_paths = predict_all_test_sets(config, best_model_path, exp_dir=exp_dir, 
                                          lang_domain_combinations=used_combinations)
        print(f"预测完成！共生成 {len(pred_paths)} 个预测文件\n")
        
        # 输出总结
        print("=" * 80)
        print("任务完成总结")
        print("=" * 80)
        print(f"使用的模型路径: {best_model_path}")
        print(f"训练日志路径: {training_log_path}")
        print(f"预测文件数量: {len(pred_paths)}")
        print("\n预测文件列表:")
        for path in pred_paths:
            print(f"  - {path}")
        print(f"完整日志文件: {log_file_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
