"""
工具函数模块（Qwen3-32B 版本）
提供数据加载、保存以及提示词构建和解析功能
"""

import json
import os
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


def parse_triplet_from_llm_response(response_text: str, data_id: str) -> List[Dict]:
    """
    从LLM响应中解析Triplet信息
    期望返回JSON格式的Triplet列表
    """
    try:
        response_text = response_text.strip()
        
        # 去除可能的思考块 <think> ... </think>
        if response_text.startswith("<think>"):
            if "</think>" in response_text:
                response_text = response_text.split("</think>", 1)[1].strip()
            else:
                response_text = "\n".join(response_text.splitlines()[1:]).strip()
                
        
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            start_idx = 1
            end_idx = len(lines) - 1
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end_idx = i
                    break
            response_text = "\n".join(lines[start_idx:end_idx])
        
        triplets = json.loads(response_text)
        
        if isinstance(triplets, dict):
            if "Triplet" in triplets:
                triplets = triplets["Triplet"]
            elif "triplets" in triplets:
                triplets = triplets["triplets"]
            else:
                triplets = [triplets]
        
        if not isinstance(triplets, list):
            logger.warning(f"⚠️  ID {data_id}: 响应格式不是列表，尝试转换")
            triplets = [triplets]
        
        valid_triplets = []
        for triplet in triplets:
            if not isinstance(triplet, dict):
                continue
            
            aspect = triplet.get("Aspect") or triplet.get("aspect") or triplet.get("a")
            opinion = triplet.get("Opinion") or triplet.get("opinion") or triplet.get("o")
            va = triplet.get("VA") or triplet.get("va") or triplet.get("v") or triplet.get("Valence-Arousal")
            
            if not aspect or not opinion or not va:
                logger.warning(f"⚠️  ID {data_id}: Triplet字段不完整，跳过: {triplet}")
                continue
            
            if not isinstance(va, str) or "#" not in va:
                logger.warning(f"⚠️  ID {data_id}: VA格式错误，应为'V#A'格式: {va}")
                continue
            
            try:
                v_str, a_str = va.split("#", 1)
                v = float(v_str.strip())
                a = float(a_str.strip())
                v = max(1.00, min(9.00, v))
                a = max(1.00, min(9.00, a))
                va_formatted = f"{v:.2f}#{a:.2f}"
            except (ValueError, AttributeError) as e:
                logger.warning(f"⚠️  ID {data_id}: VA值解析失败: {va}, 错误: {e}")
                continue
            
            valid_triplets.append({
                "Aspect": str(aspect).strip(),
                "Opinion": str(opinion).strip(),
                "VA": va_formatted
            })
        
        return valid_triplets
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ ID {data_id}: JSON解析失败: {e}")
        logger.debug(f"响应内容: {response_text[:800]}")
        return []
    except Exception as e:
        logger.error(f"❌ ID {data_id}: 解析响应时发生异常: {e}")
        logger.debug(f"响应内容: {response_text[:800]}")
        return []


# def build_subtask2_prompt(text: str, language: str = "English") -> str:
#     """构建Subtask 2的提示词"""
#     prompt = f"""请从下面的“输入文本”中提取所有的(Aspect, Opinion, VA)三元组。

# ### 任务要求：
# 1. Aspect: 文本中提到的观点目标词或短语（保持原始大小写）
# 2. Opinion: 与aspect相关的情感表达词或短语（保持原始大小写）
# 3. VA: 情感的Valence-Arousal评分，格式为"V#A"，其中：
#    - Valence (V): 情感的正负性，范围1.00-9.00（1.00=极度负面，5.00=中性，9.00=极度正面）
#    - Arousal (A): 情感的强度，范围1.00-9.00（1.00=非常平静，5.00=中等，9.00=非常激动）
#    - 每个值保留两位小数
#    - 格式示例："8.00#6.50" 或 "7.25#5.75"

# ### 返回格式（严格返回JSON格式）：
# [
#     {{
#         "Aspect": "aspect term",
#         "Opinion": "opinion term",
#         "VA": "V#A"
#     }}
# ]

# ### 示例：
# 输入："肉粿每一塊都好脆好恰好喜歡。"
# 输出："
# {
# {"Aspect": "肉粿", "Opinion": "好脆", "VA": "6.25#6.00"},
# {"Aspect": "肉粿", "Opinion": "好恰", "VA": "6.12#6.00"},
# {"Aspect": "肉粿", "Opinion": "好喜歡", "VA": "6.62#6.62"}
# }"


# 输入文本：
# {text}


# 如果没有找到任何三元组，请返回空列表 []。
# 请只返回JSON，不要包含任何其他解释性文字。"""
    
#     return prompt


# def build_subtask2_system_prompt() -> str:
#     """构建Subtask 2的系统提示词"""
#     return """你是一个专业的自然语言处理专家，擅长从文本中提取方面-观点-情感三元组。
# 你的任务是准确识别文本中的aspect terms、opinion terms，并预测对应的Valence-Arousal评分。
# 请确保：
# 1. Aspect和Opinion的提取保持原始文本的大小写
# 2. VA评分准确反映情感的维度和强度
# 3. 返回的JSON格式严格符合要求"""

def build_subtask2_prompt(text: str) -> str:
    """
    构建 Subtask 2 的提示词（英文），不依赖语言/领域参数，不包含 few-shot。

    Args:
        text: 输入文本
    """
    prompt = f"""Extract all (Aspect, Opinion, VA) triplets from the following text.
Input text:{text}
"""

    return prompt


def build_subtask2_system_prompt() -> str:
    """构建 Subtask 2 的系统提示词（英文）。"""
    system_prompt = """
    You are an expert in Dimensional Aspect-Based Sentiment Analysis (DimABSA). Your task is to perform **DimASTE**: given a text, extract all sentiment triplets of the form (Aspect, Opinion, VA), where:
- "Aspect" is the exact aspect term as it appears in the input text (case-sensitive). It refers to a word or phrase indicating an opinion target, such as appetizer, waiter, laptop, battery, or screen.
- "Opinion" is the exact opinion term associated with that aspect (case-sensitive). It refers to a sentiment-bearing word or phrase associated with a specific aspect term, such as great, terrible, or satisfactory.
- "VA" is a valence-arousal score in the format "V#A", where both V (valence) and A (arousal) are real numbers in the range [1.00, 9.00], rounded to two decimal places. Format example: "8.00#6.50" or "7.25#5.75".
    - V (Valence): Measures the degree of positivity or negativity(1.00 = extremely negative, 5.00 = neutral, 9.00 = extremely positive).
    - A (Arousal): Measures the intensity of emotion(1.00 = very calm, 5.00 = medium, 9.00 = very excited).

Output ONLY a valid JSON object with the following structure:
[
    {
        "Aspect": "aspect term",
        "Opinion": "opinion term",
        "VA": "V#A"
    }
    ...
]

Example
Input:"average to good thai food, but terrible delivery."
Output:"
[
{"Aspect": "thai food", "Opinion": "average to good", "VA": "6.75#6.38"},
{"Aspect": "delivery", "Opinion": "terrible", "VA": "2.88#6.62"}
]"

Important rules:
- Do NOT add any extra fields, explanations, comments, or markdown.
- Preserve original word forms, spacing, and case from the input text.
- Ensure VA values are within [1.00, 9.00] and formatted as "X.XX#Y.YY".
- Output must be parseable JSON — no trailing commas, syntax errors, or free text.
"""
    return system_prompt

# - If aspect or opinion cannot be identified, set "Opinion": "NULL" or "Aspect": "NULL".


