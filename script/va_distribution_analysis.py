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
    读取指定路径下所有语言的train_alltasks.jsonl文件
    
    Args:
        base_path: subtask_1的基础路径
    
    Returns:
        所有数据行的列表
    """
    all_data = []
    
    # 查找所有train_alltasks.jsonl文件
    pattern = os.path.join(base_path, "**", "*train_alltasks.jsonl")
    files = glob.glob(pattern, recursive=True)
    
    print(f"找到 {len(files)} 个train_alltasks.jsonl文件:")
    for file in files:
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
    
    Args:
        data_list: 数据记录列表
    
    Returns:
        (V值列表, A值列表)
    """
    v_values = []
    a_values = []
    
    for record in data_list:
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


def main():
    """
    主函数
    """
    # 设置中文字体（根据系统调整）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正常显示")
    
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_track_a", "subtask_1")
    # 输出到script/script_output文件夹
    output_dir = os.path.join(base_dir, "script", "script_output")
    
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
    print("\n[步骤5] 绘制分布图...")
    plot_distribution(valid_v, 'V', output_dir)
    plot_distribution(valid_a, 'A', output_dir)
    
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

