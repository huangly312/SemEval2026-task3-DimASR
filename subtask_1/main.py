import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import yaml
import sys
from datetime import datetime
from train import train_model
from predict import predict_on_test_set
from utils import Tee, create_experiment_dir


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
    主函数：根据配置中的mode决定执行训练、预测或两者都执行
    """
    # 加载配置
    config = load_config()
    
    # 获取运行模式，默认为"both"
    mode = config.get("mode", "both").lower()
    
    if mode not in ["train", "predict", "both"]:
        print(f"错误: 无效的运行模式 '{mode}'")
        print("请设置 mode 为 'train'、'predict' 或 'both'")
        sys.exit(1)
    
    # 确定日志文件位置
    subtask = config["subtask"]
    log_file_path = None
    exp_dir = None
    
    # 如果模式包含训练，创建实验目录（统一在main.py中管理）
    if mode in ["train", "both"]:
        exp_dir = create_experiment_dir(config, subtask)
        log_file_path = os.path.join(exp_dir, "training.log")
    else:
        # 仅预测模式，创建日志目录
        lang = config["lang"]
        domain = config["domain"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(subtask, "logs", f"{lang}_{domain}")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"predict_{timestamp}.log")
    
    # 设置全局日志记录：所有输出同时写入文件和终端
    with Tee(log_file_path, mode='w', encoding='utf-8'):
        print(f"日志文件: {log_file_path}")
        print("=" * 80)
        
        best_model_path = None
        training_log_path = None
        save_path = None
        
        # 执行训练
        if mode in ["train", "both"]:
            print("=" * 80)
            print("开始训练模型")
            print("=" * 80)
            best_model_path, training_log_path, exp_dir = train_model(config, exp_dir)
            print(f"训练完成！最优模型保存在: {best_model_path}\n")
        
        # 执行预测
        if mode in ["predict", "both"]:
            print("=" * 80)
            print("开始在测试集上预测")
            print("=" * 80)
            
            # 如果刚训练完，使用训练得到的最优模型；否则从配置中读取
            if best_model_path is None:
                # 仅预测模式，从配置中读取checkpoint_path
                checkpoint_path = config.get("checkpoint_path", "")
                if not checkpoint_path:
                    print("错误: 预测模式需要提供模型checkpoint路径")
                    print("请在config.yaml中设置checkpoint_path字段")
                    sys.exit(1)
                best_model_path = checkpoint_path
            
            save_path = predict_on_test_set(config, best_model_path, exp_dir=exp_dir)
            print(f"预测完成！结果保存在: {save_path}\n")
        
        # 输出总结
        print("=" * 80)
        print("任务完成总结")
        print("=" * 80)
        if best_model_path:
            print(f"使用的模型路径: {best_model_path}")
        if training_log_path:
            print(f"训练日志路径: {training_log_path}")
        if save_path:
            print(f"预测结果路径: {save_path}")
        print(f"完整日志文件: {log_file_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
