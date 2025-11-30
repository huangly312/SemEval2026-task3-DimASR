#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VA分布分析脚本
分析subtask_1中所有语言的train_alltasks.jsonl文件中的VA字段分布情况
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict


def read_jsonl_files(base_path: str) -> List[Dict]:
    """
    读取指定路径下所有语言的train数据文件
    
    Args:
        base_path: subtask_1的基础路径
    
    Returns:
        所有数据行的列表
    """
    all_data = []
    
    # 查找所有包含train的jsonl文件（包括train_alltasks和train_task1等）
    pattern = os.path.join(base_path, "**", "*train*.jsonl")
    all_files = glob.glob(pattern, recursive=True)
    
    # 过滤出训练数据文件（排除dev等其他文件）
    files = [f for f in all_files if '_train_' in os.path.basename(f)]
    
    print(f"找到 {len(files)} 个训练数据文件:")
    for file in sorted(files):
        print(f"  - {file}")
    
    # 读取每个文件
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    all_data.append(data)
    
    print(f"\n总共读取了 {len(all_data)} 条数据记录")
    return all_data


def extract_va_values(data_list: List[Dict]) -> Tuple[List[float], List[float]]:
    """
    从数据中提取所有的V和A值
    支持两种数据格式：
    1. Quadruplet 格式（train_alltasks.jsonl）
    2. Aspect_VA 格式（train_task1.jsonl）
    
    Args:
        data_list: 数据记录列表
    
    Returns:
        (V值列表, A值列表)
    """
    v_values = []
    a_values = []
    
    for record in data_list:
        # 处理 Quadruplet 格式
        if 'Quadruplet' in record:
            for quad in record['Quadruplet']:
                if 'VA' in quad:
                    va_str = quad['VA']
                    try:
                        # 拆分VA字段，格式为"V#A"
                        parts = va_str.split('#')
                        if len(parts) == 2:
                            v = float(parts[0])
                            a = float(parts[1])
                            v_values.append(v)
                            a_values.append(a)
                    except (ValueError, IndexError) as e:
                        print(f"警告: 无法解析VA值 '{va_str}': {e}")
        
        # 处理 Aspect_VA 格式
        elif 'Aspect_VA' in record:
            for aspect_va in record['Aspect_VA']:
                if 'VA' in aspect_va:
                    va_str = aspect_va['VA']
                    try:
                        # 拆分VA字段，格式为"V#A"
                        parts = va_str.split('#')
                        if len(parts) == 2:
                            v = float(parts[0])
                            a = float(parts[1])
                            v_values.append(v)
                            a_values.append(a)
                    except (ValueError, IndexError) as e:
                        print(f"警告: 无法解析VA值 '{va_str}': {e}")
    
    return v_values, a_values


def analyze_validity(values: List[float], value_name: str) -> Tuple[List[float], List[float], Dict]:
    """
    分析值的合法性（在[1,9]区间内为合法）
    
    Args:
        values: 待分析的值列表
        value_name: 值的名称（'V'或'A'）
    
    Returns:
        (合法值列表, 非法值列表, 统计信息字典)
    """
    valid_values = []
    invalid_values = []
    
    for val in values:
        if 1 <= val <= 9:
            valid_values.append(val)
        else:
            invalid_values.append(val)
    
    stats = {
        'total': len(values),
        'valid': len(valid_values),
        'invalid': len(invalid_values),
        'valid_ratio': len(valid_values) / len(values) * 100 if values else 0
    }
    
    print(f"\n{value_name}值统计:")
    print(f"  总数量: {stats['total']}")
    print(f"  合法数量: {stats['valid']} ({stats['valid_ratio']:.2f}%)")
    print(f"  非法数量: {stats['invalid']}")
    
    if invalid_values:
        print(f"  非法值示例: {invalid_values[:10]}")
    
    return valid_values, invalid_values, stats


def compute_statistics(values: List[float], value_name: str) -> Dict:
    """
    计算值的统计信息
    
    Args:
        values: 值列表
        value_name: 值的名称
    
    Returns:
        统计信息字典
    """
    if not values:
        print(f"\n{value_name}值统计信息: 无数据")
        return {}
    
    stats = {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values)
    }
    
    print(f"\n{value_name}值统计信息（仅合法值）:")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    print(f"  平均值: {stats['mean']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    
    return stats


def plot_distribution(values: List[float], value_name: str, output_dir: str):
    """
    绘制值的分布直方图
    
    Args:
        values: 值列表
        value_name: 值的名称
        output_dir: 输出目录
    """
    if not values:
        print(f"警告: 无法绘制{value_name}的分布图，因为没有合法数据")
        return
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    n, bins, patches = plt.hist(values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # 添加统计线
    mean_val = np.mean(values)
    median_val = np.median(values)
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_val:.2f}')
    
    # 设置标题和标签
    plt.title(f'{value_name}值分布直方图（合法值）', fontsize=14, fontweight='bold', fontproperties='SimHei')
    plt.xlabel(f'{value_name}值', fontsize=12, fontproperties='SimHei')
    plt.ylabel('频数', fontsize=12, fontproperties='SimHei')
    plt.legend(prop={'family': 'SimHei', 'size': 10})
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{value_name}_distribution.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{value_name}值分布图已保存到: {output_path}")
    
    plt.close()


def plot_va_combined(v_values: List[float], a_values: List[float], output_dir: str):
    """
    在同一张图中绘制V和A的分布，使用上下两个子图
    
    Args:
        v_values: V值列表
        a_values: A值列表
        output_dir: 输出目录
    """
    if not v_values and not a_values:
        print("警告: 无法绘制VA组合分布图，因为没有合法数据")
        return
    
    # 创建上下两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 定义横坐标范围（1-9）
    bins = np.arange(1, 10.5, 0.25)  # 从1到9.5，步长为0.25
    
    # 绘制V值分布（上子图）
    if v_values:
        ax1.hist(v_values, bins=bins, alpha=0.7, color='steelblue', edgecolor='steelblue', linewidth=0.8, label='Valence')
        # ax1.hist(v_values, bins=bins, alpha=0.7, color='steelblue', edgecolor='none', label='Valence')
        # ax1.set_title('V值分布直方图（合法值）', fontsize=14, fontweight='bold', fontproperties='SimHei')
        ax1.set_title('Valence and Arousal Data Distribution', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Number of samples', fontsize=14)
        ax1.set_xlim(1, 9)
        ax1.set_ylim(0, 5000)
        ax1.legend(fontsize=14)
        ax1.grid(True, alpha=0.3)
    
    # 绘制A值分布（下子图）
    if a_values:
        ax2.hist(a_values, bins=bins, alpha=0.7, color='coral', edgecolor='coral', label='Arousal')
        # ax2.set_title('A值分布直方图（合法值）', fontsize=14, fontweight='bold', fontproperties='SimHei')
        ax2.set_xlabel('Scores', fontsize=14)
        ax2.set_ylabel('Number of samples', fontsize=14)
        ax2.set_xlim(1, 9)
        ax2.set_ylim(0, 5000)
        ax2.legend(fontsize=14)
        ax2.grid(True, alpha=0.3)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'VA_combined_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVA组合分布图已保存到: {output_path}")
    
    plt.close()


def main():
    """
    主函数
    """
    # 设置学术论文常用字体
    # 选项1: Times New Roman (传统学术论文常用，衬线字体)
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    # plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # 选项2: Computer Modern (LaTeX默认字体，学术风格)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif']
    
    # 选项3: Arial/Helvetica (无衬线字体，现代风格)
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体（可选）
    
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_track_a", "subtask_1")
    # 输出到script/script_output文件夹
    output_dir = os.path.join(base_dir, "script", "va_distribution_output")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("VA分布分析脚本")
    print("="*60)
    
    # 1. 读取所有数据
    print("\n[步骤1] 读取数据文件...")
    all_data = read_jsonl_files(data_path)
    
    # 2. 提取V和A值
    print("\n[步骤2] 提取V和A值...")
    v_values, a_values = extract_va_values(all_data)
    print(f"提取到 {len(v_values)} 个V值和 {len(a_values)} 个A值")
    
    # 3. 分析合法性
    print("\n[步骤3] 分析数据合法性...")
    valid_v, invalid_v, v_stats = analyze_validity(v_values, 'V')
    valid_a, invalid_a, a_stats = analyze_validity(a_values, 'A')
    
    # 4. 计算统计信息
    print("\n[步骤4] 计算统计信息...")
    v_statistics = compute_statistics(valid_v, 'V')
    a_statistics = compute_statistics(valid_a, 'A')
    
    # 5. 绘制分布图
    # print("\n[步骤5] 绘制分布图...")
    # plot_distribution(valid_v, 'V', output_dir)
    # plot_distribution(valid_a, 'A', output_dir)
    
    # 5.1 绘制VA组合分布图（上下两个子图）
    print("\n[步骤5.1] 绘制VA组合分布图...")
    plot_va_combined(valid_v, valid_a, output_dir)
    # return 
    
    # 6. 保存统计结果
    print("\n[步骤6] 保存统计结果...")
    summary = {
        'V': {
            'validity': v_stats,
            'statistics': v_statistics
        },
        'A': {
            'validity': a_stats,
            'statistics': a_statistics
        }
    }
    
    # 保存JSON格式
    summary_path = os.path.join(output_dir, 'va_analysis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"统计摘要(JSON)已保存到: {summary_path}")

    print("分析完成！所有输出文件已保存到: " + output_dir)
    print("="*60)


if __name__ == "__main__":
    main()

