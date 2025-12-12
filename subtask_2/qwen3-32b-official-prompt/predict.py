"""
Subtask 2 预测主脚本
使用Qwen3-32B API进行DimASTE任务预测（并行）
使用官方prompt格式
"""

import os
import sys
import yaml
import logging
from typing import Dict
from tqdm import tqdm

from api_client import QwenClient
from utils import (
    load_jsonl,
    save_jsonl,
    build_subtask2_prompt,
    load_task2_prompt_template,
    parse_triplet_from_text_response,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict:
    """加载配置文件"""
    if config_path is None:
        # 默认路径：相对于当前文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ 成功加载配置文件: {config_path}")
    return config


def predict_subtask2(
    subtask: str,
    lang: str,
    domain: str,
    task: str = "task2",
    max_workers: int = 10,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    top_k: int = 0,
    enable_thinking: bool = False,
    api_config: Dict = None,
    output_dir: str = None,
):
    """对测试集进行预测"""
    # 数据路径：从根目录读取
    predict_path = f"data/raw_track_a/{subtask}/{lang}/{lang}_{domain}_dev_{task}.jsonl"
    
    if not os.path.exists(predict_path):
        raise FileNotFoundError(f"测试数据文件不存在: {predict_path}")
    
    if output_dir is None:
        # 默认输出目录：相对于当前文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "predict_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pred_{lang}_{domain}.jsonl")
    
    logger.info("=" * 80)
    logger.info("Subtask 2 预测任务 (Qwen3-32B Official Prompt)")
    logger.info("=" * 80)
    logger.info(f"📂 输入文件: {predict_path}")
    logger.info(f"📂 输出文件: {output_path}")
    logger.info(f"🌐 语言: {lang}, 领域: {domain}")
    logger.info(f"⚙️  并发数: {max_workers}, Temperature: {temperature}")
    logger.info(f"🧠 Thinking模式: {'启用' if enable_thinking else '禁用'}")
    logger.info("=" * 80)
    
    logger.info("📖 加载测试数据...")
    test_data = load_jsonl(predict_path)
    logger.info(f"✅ 成功加载 {len(test_data)} 条测试数据")
    
    logger.info("🔧 初始化Qwen API客户端...")
    api_cfg = api_config or {}
    client = QwenClient(
        base_url=api_cfg.get("base_url"),
        api_key=api_cfg.get("key"),
        model=api_cfg.get("model"),
        timeout=api_cfg.get("timeout", 60),
        max_retries=api_cfg.get("max_retries", 3),
        retry_delay=api_cfg.get("retry_delay", 1.0),
        top_p=api_cfg.get("top_p", top_p),
        presence_penalty=api_cfg.get("presence_penalty", presence_penalty),
        top_k=api_cfg.get("top_k", top_k),
        enable_thinking=enable_thinking,
    )
    
    # 加载prompt模板
    logger.info("📝 加载prompt模板...")
    prompt_template = load_task2_prompt_template()
    
    logger.info("📝 构建提示词...")
    prompts = []
    for item in test_data:
        text = item.get("Text", "")
        prompt = build_subtask2_prompt(text, prompt_template)
        prompts.append(prompt)
    logger.info(f"✅ 成功构建 {len(prompts)} 个提示词")
    
    logger.info("🚀 开始批量调用API...")
    progress_bar = tqdm(total=len(prompts), desc="预测进度", unit="条")
    
    def progress_callback(current, total):
        progress_bar.update(1)
    
    results = client.generate_batch(
        prompts=prompts,
        system_prompt=None,  # 不使用system prompt，所有内容都在user prompt中
        max_tokens=max_tokens,
        temperature=temperature,
        max_workers=max_workers,
        progress_callback=progress_callback,
    )
    progress_bar.close()
    
    logger.info("📊 处理预测结果...")
    output_data = []
    success_count = 0
    fail_count = 0
    
    for i, (item, result) in enumerate(zip(test_data, results)):
        data_id = item.get("ID", f"unknown_{i}")
        
        if not result or not result.get("success", False):
            logger.warning(f"⚠️  ID {data_id}: API调用失败: {result.get('error', '未知错误') if result else '未知错误'}")
            output_data.append({"ID": data_id, "Triplet": []})
            fail_count += 1
            continue
        
        reply = result.get("reply", "")
        triplets = parse_triplet_from_text_response(reply, data_id)
        output_data.append({"ID": data_id, "Triplet": triplets})
        
        if triplets:
            success_count += 1
        else:
            logger.warning(f"⚠️  ID {data_id}: 未解析到任何triplet，LLM原始输出(截断): {reply[:800]}")
    
    logger.info("💾 保存预测结果...")
    save_jsonl(output_data, output_path)
    
    logger.info("=" * 80)
    logger.info("📈 预测统计")
    logger.info("=" * 80)
    logger.info(f"总数据量: {len(test_data)}")
    logger.info(f"API调用成功: {len(test_data) - fail_count}")
    logger.info(f"API调用失败: {fail_count}")
    logger.info(f"解析到Triplet: {success_count}")
    logger.info(f"空结果: {len(test_data) - success_count}")
    logger.info(f"✅ 结果已保存到: {output_path}")
    logger.info("=" * 80)


def main():
    """主函数"""
    try:
        config = load_config()
    except FileNotFoundError:
        logger.error("❌ 配置文件不存在，请先创建 config.yaml")
        sys.exit(1)
    
    subtask = config.get("subtask", "subtask_2")
    lang = config.get("lang", "eng")
    domain = config.get("domain", "restaurant")
    task = config.get("task", "task2")
    max_workers = config.get("max_workers", 10)
    max_tokens = config.get("max_tokens", 1000)
    temperature = config.get("temperature", 0.0)
    top_p = config.get("top_p", 1.0)
    presence_penalty = config.get("presence_penalty", 0.0)
    top_k = config.get("top_k", 0)
    enable_thinking = config.get("enable_thinking", False)
    
    api_config = config.get("api", {})
    
    try:
        predict_subtask2(
            subtask=subtask,
            lang=lang,
            domain=domain,
            task=task,
            max_workers=max_workers,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            top_k=top_k,
            enable_thinking=enable_thinking,
            api_config=api_config,
        )
    except Exception as e:
        logger.error(f"❌ 预测过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
