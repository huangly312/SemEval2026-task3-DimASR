"""
批量运行 Qwen3-32B Official Prompt，生成全部 8 个语言-领域组合的预测文件。

使用方式（在项目根目录）：
    python subtask_2/llm_api_method/qwen3-32b-official-prompt/batch_predict_all.py
"""

import logging
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime

from predict import load_config, predict_subtask2

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全部 8 个语言-领域组合
LANG_DOMAIN_COMBINATIONS: List[Tuple[str, str]] = [
    ("eng", "restaurant"),
    ("eng", "laptop"),
    ("zho", "restaurant"),
    ("zho", "laptop"),
    ("jpn", "hotel"),
    ("rus", "restaurant"),
    ("tat", "restaurant"),
    ("ukr", "restaurant"),
]


def run_all_predictions():
    """按顺序跑完 8 个语言-领域组合的预测。"""
    try:
        config: Dict = load_config()
    except FileNotFoundError:
        logger.error("❌ 配置文件不存在，请先创建 config.yaml")
        sys.exit(1)

    # 公共配置
    subtask = config.get("subtask", "subtask_2")
    task = config.get("task", "task2")
    max_workers = config.get("max_workers", 10)
    max_tokens = config.get("max_tokens", 1000)
    temperature = config.get("temperature", 0.0)
    top_p = config.get("top_p", 1.0)
    presence_penalty = config.get("presence_penalty", 0.0)
    top_k = config.get("top_k", 0)
    enable_thinking = config.get("enable_thinking", False)

    # API 配置
    api_cfg = config.get("api", {})

    # 为本次批量运行创建时间戳输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(
        current_dir,
        "predict_output",
        timestamp,
    )
    os.makedirs(base_output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("开始批量预测（Qwen3-32B Official Prompt）")
    logger.info(f"共 {len(LANG_DOMAIN_COMBINATIONS)} 个语言-领域组合")
    logger.info(f"本次输出目录: {os.path.abspath(base_output_dir)}")
    logger.info("=" * 80)

    success, failed = [], []

    for lang, domain in LANG_DOMAIN_COMBINATIONS:
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
                api_config=api_cfg,
                output_dir=base_output_dir,
            )
            success.append(f"{lang}-{domain}")
        except Exception as e:
            logger.error(f"❌ {lang}-{domain} 预测失败: {e}", exc_info=True)
            failed.append(f"{lang}-{domain}")

    logger.info("=" * 80)
    logger.info("批量预测完成")
    logger.info(f"成功: {len(success)} / {len(LANG_DOMAIN_COMBINATIONS)}")
    if success:
        logger.info("成功列表: " + ", ".join(success))
    if failed:
        logger.info("失败列表: " + ", ".join(failed))
    logger.info(f"输出目录: {os.path.abspath(base_output_dir)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_all_predictions()
