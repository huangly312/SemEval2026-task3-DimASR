import os
import sys
import random
import numpy as np
import torch
from datetime import datetime
import json
from typing import List, Dict
import pandas as pd
import re


def set_global_seed(seed):
    """
    设置全局随机种子，确保实验的可复现性
    
    设置以下组件的随机种子：
    - Python 内置 random 模块
    - NumPy
    - PyTorch (CPU)
    - PyTorch (CUDA, 如果可用)
    - 环境变量 PYTHONHASHSEED (用于 Python hash 的随机性)
    
    Args:
        seed (int): 随机种子值
    """
    # Python 内置 random 模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA (如果可用)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
        # 确保 CUDA 操作的确定性（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设置 Python hash 随机种子（用于字典等数据结构的哈希）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Global seed set to: {seed}")
    if torch.cuda.is_available():
        print(f"  - CUDA deterministic mode: enabled")
        print(f"  - CUDA benchmark mode: disabled")


def worker_init_fn(worker_id):
    """
    DataLoader 的 worker_init_fn，用于确保多进程数据加载的可复现性
    
    注意：这个函数需要在创建 DataLoader 时使用，并且需要配合 set_global_seed 使用
    由于 worker 进程是独立的，需要在每个 worker 中重新设置随机种子
    
    Args:
        worker_id (int): worker 进程的 ID
    """
    # 获取主进程设置的初始种子
    # torch.initial_seed() 返回当前 worker 的初始种子
    # 由于我们在主进程中已经设置了全局种子，这里使用它来确保一致性
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_experiment_dir(config, subtask="subtask_1"):
    """
    创建实验目录，按照 模型 -> 语言 -> 领域 -> 时间戳+参数 的层级结构
    
    目录结构:
    subtask_1/experiments/{model_name}/{lang}/{domain}/YYYYMMDD_HHMMSS_{lr}_{batch_size}/
    
    Args:
        config: 配置字典
        subtask: 子任务目录名
        
    Returns:
        exp_dir: 实验目录路径
    """
    # 从配置中提取参数
    lang = config["lang"]
    domain = config["domain"]
    model_name = config["model_name"]
    lr = float(config["lr"])
    batch_size = int(config.get("batch_size", 64))
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 格式化学习率（例如 1e-5 -> 1e-5）
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    
    # 创建实验目录名
    exp_dir_name = f"{timestamp}_{lr_str}_{batch_size}"
    
    # 构建完整路径: experiments/{model_name}/{lang}/{domain}/{exp_dir_name}
    exp_dir = os.path.join(subtask, "experiments", model_name, lang, domain, exp_dir_name)
    
    # 创建目录
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Experiment directory: {exp_dir}")
    
    return exp_dir


def load_jsonl(filepath: str) -> List[Dict]:
    """
    加载 JSONL 文件
    
    Args:
        filepath: JSONL 文件路径
        
    Returns:
        数据列表，每个元素是一个字典
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def jsonl_to_df(data):
    """
    将 JSONL 数据转换为 DataFrame
    
    支持以下格式：
    - Quadruplet: 包含 Quadruplet 字段
    - Triplet: 包含 Triplet 字段
    - Aspect_VA: 包含 Aspect_VA 字段
    - Aspect: 包含 Aspect 字段
    
    Args:
        data: JSONL 数据列表
        
    Returns:
        处理后的 DataFrame，包含 ID, Text, Aspect, Valence, Arousal 列
    """
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Opinion'])  # drop unnecessary columns
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect_VA' in data[0]:
        df = pd.json_normalize(data, 'Aspect_VA', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')  # remove duplicate ID+Aspect

    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})  # rename to Aspect
        df['Valence'] = 0  # default value
        df['Arousal'] = 0  # default value

    else:
        raise ValueError("Invalid format: must include 'Quadruplet' or 'Triplet' or 'Aspect'")

    # Filter out rows where Aspect is "NULL"
    original_len = len(df)
    df = df[df['Aspect'] != 'NULL'].reset_index(drop=True)
    filtered_count = original_len - len(df)
    print(f"过滤数据('Aspect'='NULL')条目:{filtered_count}")

    return df


def df_to_jsonl(df, out_path):
    """
    将 DataFrame 转换为 JSONL 格式并保存
    
    Args:
        df: 包含 ID, Aspect, Valence, Arousal 的 DataFrame
        out_path: 输出文件路径
    """
    def extract_num(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else -1
    
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {
                "ID": gid,
                "Aspect_VA": []
            }
            for _, row in gdf.iterrows():
                record["Aspect_VA"].append({
                    "Aspect": row["Aspect"],
                    "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                })
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Tee:
    """
    同时将输出写入文件和终端（类似 Unix 的 tee 命令）
    
    支持上下文管理器（with 语句），自动处理资源清理
    
    用法1 (推荐，使用上下文管理器):
        with Tee('output.log'):
            print("这行会同时显示在终端和文件中")
        # 自动关闭文件并恢复输出
    
    用法2 (手动管理):
        tee = Tee('output.log')
        print("这行会同时显示在终端和文件中")
        tee.close()
    """
    def __init__(self, file_path, mode='w', encoding='utf-8'):
        """
        Args:
            file_path: 日志文件路径
            mode: 文件打开模式（'w' 覆盖, 'a' 追加）
            encoding: 文件编码
        """
        self.file = open(file_path, mode, encoding=encoding)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def write(self, text):
        """写入文本到文件和终端"""
        self.file.write(text)
        self.file.flush()  # 立即刷新到文件
        self.stdout.write(text)
        self.stdout.flush()
        
    def flush(self):
        """刷新缓冲区"""
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        """关闭文件并恢复原始输出"""
        if self.file and not self.file.closed:
            self.file.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
    
    def __enter__(self):
        """上下文管理器入口：重定向输出"""
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口：恢复输出并关闭文件"""
        self.close()
        return False  # 不抑制异常

