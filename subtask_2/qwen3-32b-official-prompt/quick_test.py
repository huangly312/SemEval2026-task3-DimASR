"""
快速测试脚本 - Qwen3-32B Official Prompt 版本
用于测试 API 连接、数据加载、三元组解析等功能
"""

import os
import sys
from api_client import QwenClient
from utils import (
    load_jsonl,
    parse_triplet_from_text_response,
    build_subtask2_prompt,
    load_task2_prompt_template,
)
from predict import load_config


def test_api_connection():
    """
    测试 API 连接和基本模型响应
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("测试 API 连接")
    print("="*60)
    
    try:
        # 尝试从配置文件加载
        try:
            config = load_config()
            api_cfg = config.get("api", {})
            base_url = api_cfg.get("base_url", "http://10.208.62.156:8002/v1")
            api_key = api_cfg.get("key", "EMPTY")
            model = api_cfg.get("model", "Qwen3-32B")
        except:
            # 如果配置文件不存在，使用默认值
            base_url = "http://10.208.62.156:8002/v1"
            api_key = "EMPTY"
            model = "Qwen3-32B"
        
        print(f"API 地址: {base_url}")
        print(f"模型: {model}")
        
        client = QwenClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=30,
            max_retries=1,
        )
        
        result = client.generate_response(
            prompt="Hello, please respond with 'OK' if you can read this.",
            max_tokens=50,
            temperature=0.0,
        )
        
        if result.get("success", False):
            content = result.get("reply", "")
            print(f"响应: {content}")
            print("✅ API 连接成功！\n")
            return True
        else:
            print(f"❌ API 调用失败: {result.get('error', '未知错误')}\n")
            return False
        
    except Exception as e:
        print(f"❌ API 连接失败: {e}\n")
        return False


def test_data_loading():
    """
    测试数据加载功能
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("测试数据加载")
    print("="*60)
    
    try:
        # 尝试加载一个示例文件
        test_file = "data/raw_track_a/subtask_2/eng/eng_restaurant_dev_task2.jsonl"
        
        if not os.path.exists(test_file):
            # 尝试其他可能的文件
            alt_files = [
                "data/raw_track_a/subtask_2/zho/zho_restaurant_dev_task2.jsonl",
                "data/raw_track_a/subtask_2/eng/eng_laptop_dev_task2.jsonl",
            ]
            found = False
            for alt_file in alt_files:
                if os.path.exists(alt_file):
                    test_file = alt_file
                    found = True
                    break
            
            if not found:
                print(f"⚠️  测试文件未找到，尝试的路径:")
                print(f"  - {test_file}")
                for alt_file in alt_files:
                    print(f"  - {alt_file}")
                print("跳过数据加载测试\n")
                return False
        
        dataset = load_jsonl(test_file)
        print(f"✅ 成功加载 {len(dataset)} 条数据 from {test_file}")
        
        # 显示第一条数据示例
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n示例数据:")
            print(f"  ID: {sample.get('ID', 'N/A')}")
            print(f"  Text: {sample.get('Text', 'N/A')[:100]}...")
        
        print("\n✅ 数据加载成功！\n")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}\n")
        return False


def test_triplet_parsing():
    """
    测试三元组解析功能
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("测试三元组解析")
    print("="*60)
    
    try:
        # 测试标准格式
        test_cases = [
            "(food, delicious, 7.5#6.0), (service, slow, 3.0#5.5)",
            "[Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)",
            "(food, delicious, 7.5#6.0)",
            "Some text before (food, delicious, 7.5#6.0), (service, slow, 3.0#5.5) some text after",
            # 测试 thinking 模式输出
            "<think>\nThis is a thinking block with some analysis.\n</think>\n[Triplet] (food, great, 8.0#7.0), (service, excellent, 9.0#8.0)",
            "<think>Some thinking content</think>(food, good, 7.5#6.5)",
            # 测试无效格式（应该被过滤）
            "(, opinion, 5.0#6.0)",  # 空 aspect
            "(aspect, , 5.0#6.0)",   # 空 opinion
        ]
        
        print("测试用例:")
        for i, test_output in enumerate(test_cases, 1):
            print(f"\n用例 {i}:")
            print(f"  输入: {test_output}")
            triplets = parse_triplet_from_text_response(test_output, f"test_{i}")
            print(f"  输出: {triplets}")
            if triplets:
                print(f"  ✅ 解析成功，提取到 {len(triplets)} 个三元组")
            else:
                print(f"  ⚠️  未解析到三元组")
        
        print("\n✅ 三元组解析测试完成！\n")
        return True
        
    except Exception as e:
        print(f"❌ 三元组解析测试失败: {e}\n")
        return False


def test_prompt_building():
    """
    测试 Prompt 构建功能
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("测试 Prompt 构建")
    print("="*60)
    
    try:
        # 加载 prompt 模板
        prompt_template = load_task2_prompt_template()
        print(f"✅ Prompt 模板加载成功 (长度: {len(prompt_template)} 字符)")
        print(f"\n模板前 200 字符:")
        print(prompt_template[:200] + "...")
        
        # 测试构建 prompt
        test_text = "The food was delicious but the service was slow."
        prompt = build_subtask2_prompt(test_text, prompt_template)
        
        print(f"\n测试文本: {test_text}")
        print(f"\n构建的 Prompt (前 300 字符):")
        print(prompt[:300] + "...")
        
        print("\n✅ Prompt 构建成功！\n")
        return True
        
    except Exception as e:
        print(f"❌ Prompt 构建失败: {e}\n")
        return False


def test_end_to_end():
    """
    测试端到端预测（单个样本）
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("测试端到端预测")
    print("="*60)
    
    try:
        # 从配置文件加载 API 配置
        try:
            config = load_config()
            api_cfg = config.get("api", {})
            base_url = api_cfg.get("base_url", "http://10.208.62.156:8002/v1")
            api_key = api_cfg.get("key", "EMPTY")
            model = api_cfg.get("model", "Qwen3-32B")
            max_tokens = config.get("max_tokens", 1000)
            temperature = config.get("temperature", 0.0)
            enable_thinking = config.get("enable_thinking", False)
        except:
            # 如果配置文件不存在，使用默认值
            base_url = "http://10.208.62.156:8002/v1"
            api_key = "EMPTY"
            model = "Qwen3-32B"
            max_tokens = 1000
            temperature = 0.0
            enable_thinking = False
        
        # 初始化客户端
        client = QwenClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=60,
            max_retries=2,
            enable_thinking=enable_thinking,
        )
        
        # 构建测试样本
        test_sample = {
            "ID": "test_0",
            "Text": "The food was delicious but the service was slow."
        }
        
        # 构建 prompt
        prompt_template = load_task2_prompt_template()
        prompt = build_subtask2_prompt(test_sample["Text"], prompt_template)
        
        print(f"测试文本: {test_sample['Text']}")
        print(f"\n发送请求到模型...")
        
        # 生成预测
        result = client.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if not result.get("success", False):
            print(f"❌ API 调用失败: {result.get('error', '未知错误')}\n")
            return False
        
        content = result.get("reply", "")
        print(f"\n模型响应:")
        print(content[:500] + ("..." if len(content) > 500 else ""))
        
        # 解析三元组
        triplets = parse_triplet_from_text_response(content, test_sample["ID"])
        print(f"\n解析的三元组: {triplets}")
        
        if triplets:
            print(f"\n✅ 端到端预测成功！提取到 {len(triplets)} 个三元组")
        else:
            print(f"\n⚠️  未解析到三元组，但 API 调用成功")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ 端到端预测失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Qwen3-32B Official Prompt - 快速测试套件")
    print("="*60 + "\n")
    
    results = {}
    
    # 运行测试（按顺序）
    results['Prompt 构建'] = test_prompt_building()
    results['数据加载'] = test_data_loading()
    results['三元组解析'] = test_triplet_parsing()
    results['API 连接'] = test_api_connection()
    results['端到端预测'] = test_end_to_end()
    
    # 测试总结
    print("="*60)
    print("测试总结")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "通过" if passed else "失败"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n总计: {passed_tests}/{total_tests} 个测试通过")
    
    if passed_tests == total_tests:
        print("\n✅ 所有测试通过！系统可以使用。")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} 个测试失败，请检查上面的错误信息。")


if __name__ == "__main__":
    main()

