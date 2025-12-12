"""
Qwen3-32B API客户端模块
使用OpenAI兼容接口并支持并行调用
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable
import logging

from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIError

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenClient:
    """Qwen3-32B API客户端"""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        top_p: float = 0.8,
        presence_penalty: float = 1.2,
        top_k: int = 20
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.top_k = top_k
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        logger.info("✅ Qwen客户端初始化完成")
        logger.info(f"📋 使用的模型: {self.model}")
        logger.info(f"⏱️  超时设置: {self.timeout}秒, 重试: {self.max_retries}")
    
    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> Dict:
        """单条生成，带重试"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    extra_body={
                        "top_k": self.top_k,
                        "chat_template_kwargs": {"enable_thinking": False}, #禁用思考
                    },
                    timeout=self.timeout,
                )
                elapsed_time = time.time() - start_time
                
                if response and response.choices:
                    reply = response.choices[0].message.content
                    return {
                        "success": True,
                        "reply": reply,
                        "full_response": response,
                        "elapsed_time": elapsed_time,
                        "tokens": getattr(response.usage, "total_tokens", 0),
                    }
                
                error_msg = "响应为空或无choices字段"
                last_error = error_msg
                logger.warning(f"❌ 请求返回异常: {error_msg}")
            
            except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
                last_error = str(e)
                logger.warning(f"❌ 请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"❌ 未知错误 (尝试 {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
        
        return {"success": False, "error": last_error or "所有重试均失败"}
    
    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        max_workers: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict]:
        """并行批量生成"""
        total = len(prompts)
        results = [None] * total
        
        def process_single(idx_and_prompt):
            idx, prompt = idx_and_prompt
            result = self.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result["index"] = idx
            return idx, result
        
        logger.info(f"🚀 开始并行处理 {total} 个请求（最大并发数: {max_workers}）")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single, (idx, prompt)): idx
                for idx, prompt in enumerate(prompts)
            }
            
            completed = 0
            for future in as_completed(future_to_idx):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = future_to_idx[future]
                    results[idx] = {
                        "success": False,
                        "error": f"处理异常: {str(e)}",
                        "index": idx,
                    }
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total)
                elif completed % 10 == 0 or completed == total:
                    logger.info(f"📊 进度: {completed}/{total} ({completed/total*100:.1f}%)")
        
        elapsed_time = time.time() - start_time
        success_count = sum(1 for r in results if r and r.get("success", False))
        logger.info(f"✅ 批量处理完成！成功: {success_count}/{total}, 耗时: {elapsed_time:.2f}秒")
        if elapsed_time > 0:
            logger.info(f"📈 平均速度: {total/elapsed_time:.2f} 请求/秒")
        
        return results

