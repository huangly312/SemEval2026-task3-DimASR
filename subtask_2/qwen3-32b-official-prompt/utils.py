"""
工具函数模块（Qwen3-32B Official Prompt 版本）
提供数据加载、保存以及提示词构建和解析功能
"""

import json
import os
import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSONL 文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️  第 {line_num} 行JSON解析失败: {e}")
                continue
    
    logger.info(f"✅ 成功加载 {len(data)} 条数据 from {filepath}")
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """保存数据到 JSONL 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"✅ 成功保存 {len(data)} 条数据到 {filepath}")


def load_task2_prompt_template(prompt_file: str = None) -> str:
    """
    加载 Task 2 的 prompt 模板
    
    Args:
        prompt_file: prompt文件路径，如果为None则使用默认路径
        
    Returns:
        prompt模板字符串
    """
    if prompt_file is None:
        # 默认路径：相对于当前文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(current_dir, "task2_prompt.txt")
    
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # 如果文件不存在，返回默认模板
        logger.warning(f"⚠️  Prompt文件不存在: {prompt_file}，使用默认模板")
        return """Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

### Instruction:
Given a textual instance [Text], extract all (A, O, VA) triplets, where:
- A is an Aspect term (a phrase describing an entity mentioned in [Text])
- O is an Opinion term
- VA is a Valence–Arousal score in the format (valence#arousal)

Valence ranges from 1 (negative) to 9 (positive),
Arousal ranges from 1 (calm) to 9 (excited).

### Example:
Input:
[Text] average to good thai food, but terrible delivery.
[Aspect] thai food, delivery

Output:
[Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)

### Question:
Now complete the following example:
Input:
"""


def build_subtask2_prompt(text: str, prompt_template: str = None) -> str:
    """
    构建 Subtask 2 的提示词
    
    Args:
        text: 输入文本
        prompt_template: prompt模板，如果为None则从文件加载
        
    Returns:
        完整的prompt字符串
    """
    if prompt_template is None:
        prompt_template = load_task2_prompt_template()
    
    # 与 data_processor.py 中的 format_for_inference() 保持一致
    prompt = prompt_template + '[Text] ' + text + '\n\nOutput:'
    
    return prompt


def parse_triplet_from_text_response(response_text: str, data_id: str) -> List[Dict]:
    """
    从LLM的文本响应中解析Triplet信息
    期望格式：[Triplet] (aspect, opinion, V#A), (aspect, opinion, V#A), ...
    
    Args:
        response_text: LLM返回的文本响应
        data_id: 数据ID，用于日志
        
    Returns:
        Triplet列表，格式为 [{"Aspect": "...", "Opinion": "...", "VA": "V#A"}, ...]
    """
    try:
        response_text = response_text.strip()
        
        # 去除可能的思考块 <think> ... </think>
        if "<think>" in response_text:
            if "</think>" in response_text:
                response_text = response_text.split("</think>", 1)[1].strip()
            else:
                # 如果只有开始标记，尝试找到第一个非think行
                lines = response_text.splitlines()
                for i, line in enumerate(lines):
                    if "</think>" in line or (i > 0 and not line.strip().startswith("<think")):
                        response_text = "\n".join(lines[i:]).strip()
                        break
        
        # Pattern for (Aspect, Opinion, VA)
        pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
        matches = re.findall(pattern, response_text)
        
        result = []
        for aspect, opinion, va in matches:
            result.append({
                "Aspect": aspect.strip(),
                "Opinion": opinion.strip(),
                "VA": va.strip()
            })
        
        return result
        
    except Exception as e:
        logger.error(f"❌ ID {data_id}: 解析响应时发生异常: {e}")
        logger.debug(f"响应内容: {response_text[:800]}")
        return []
