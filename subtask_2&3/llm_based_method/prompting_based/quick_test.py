"""
Quick test script for DimABSA online LLM prediction system
"""

import os
from openai import OpenAI
from data_processor import DataProcessor


def test_api_connection():
    """
    Test API connection and basic model response
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("Testing API Connection")
    print("="*60)
    
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://10.208.62.156:8002/v1",
        )
        
        response = client.chat.completions.create(
            model="Qwen3-32B",
            messages=[
                {"role": "user", "content": "Hello, please respond with 'OK' if you can read this."}
            ],
            max_tokens=100,
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        print(f"Response: {content}")
        print("API connection successful!\n")
        return True
        
    except Exception as e:
        print(f"API connection failed: {e}\n")
        return False


def test_data_loading():
    """
    Test data loading functionality
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        # Test Task 2
        processor_task2 = DataProcessor(
            task_type="task2",
            domain="restaurant",
            exclude_null=True
        )
        
        # Try to load a sample dev file
        test_file = "data/raw_track_a/subtask_2/zho/zho_restaurant_dev_task2.jsonl"
        
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            print("Skipping data loading test\n")
            return False
        
        dataset = processor_task2.load_dev_dataset(test_file)
        print(f"Loaded {len(dataset)} samples from {test_file}")
        
        # Test format_for_inference
        if len(dataset) > 0:
            messages = processor_task2.format_for_inference(dataset[0])
            print(f"\nSample prompt (first 300 chars):")
            print(messages[0]['content'][:300] + "...")
        
        print("\nData loading successful!\n")
        return True
        
    except Exception as e:
        print(f"Data loading failed: {e}\n")
        return False


def test_prediction_extraction():
    """
    Test prediction extraction functionality
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("Testing Prediction Extraction")
    print("="*60)
    
    try:
        # Test Task 2 extraction
        task2_output = "(food, delicious, 7.5#6.0), (service, slow, 3.0#5.5)"
        extracted_task2 = DataProcessor.extract_predictions(task2_output, "task2")
        print(f"Task 2 extraction:")
        print(f"  Input: {task2_output}")
        print(f"  Output: {extracted_task2}")
        
        # Test Task 3 extraction
        task3_output = "(food, FOOD#QUALITY, delicious, 7.5#6.0), (service, SERVICE#GENERAL, slow, 3.0#5.5)"
        extracted_task3 = DataProcessor.extract_predictions(task3_output, "task3")
        print(f"\nTask 3 extraction:")
        print(f"  Input: {task3_output}")
        print(f"  Output: {extracted_task3}")
        
        print("\nPrediction extraction successful!\n")
        return True
        
    except Exception as e:
        print(f"Prediction extraction failed: {e}\n")
        return False


def test_prompt_templates():
    """
    Test prompt template loading
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("Testing Prompt Templates")
    print("="*60)
    
    try:
        from data_processor import PromptTemplate
        
        # Test Task 2 prompt
        task2_prompt = PromptTemplate.load_task2_prompt("prompt/task2_prompt.txt")
        print(f"Task 2 prompt loaded (length: {len(task2_prompt)})")
        print(f"First 200 chars: {task2_prompt[:200]}...")
        
        # Test Task 3 prompt
        task3_prompt = PromptTemplate.load_task3_prompt("restaurant", "prompt/task3_prompt_template.txt")
        print(f"\nTask 3 prompt loaded (length: {len(task3_prompt)})")
        print(f"First 200 chars: {task3_prompt[:200]}...")
        
        print("\nPrompt template loading successful!\n")
        return True
        
    except Exception as e:
        print(f"Prompt template loading failed: {e}\n")
        return False


def test_end_to_end():
    """
    Test end-to-end prediction on a single sample
    
    Returns:
        Boolean indicating success
    """
    print("="*60)
    print("Testing End-to-End Prediction")
    print("="*60)
    
    try:
        # Initialize client
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://10.208.62.156:8002/v1",
        )
        
        # Initialize processor
        processor = DataProcessor(
            task_type="task2",
            domain="restaurant",
            exclude_null=True
        )
        
        # Create a test sample
        test_sample = {
            "ID": "test_0",
            "Text": "The food was delicious but the service was slow."
        }
        
        # Format for inference
        messages = processor.format_for_inference(test_sample)
        
        print(f"Test text: {test_sample['Text']}")
        print(f"\nSending request to model...")
        
        # Generate prediction
        response = client.chat.completions.create(
            model="Qwen3-32B",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={"top_k": 20},
        )
        
        content = response.choices[0].message.content
        print(f"Model response: {content}")
        
        # Extract predictions
        extracted = processor.extract_predictions(content, "task2")
        print(f"\nExtracted triplets: {extracted}")
        
        print("\nEnd-to-end prediction successful!\n")
        return True
        
    except Exception as e:
        print(f"End-to-end prediction failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("DimABSA Online LLM Prediction - Quick Test Suite")
    print("="*60 + "\n")
    
    results = {}
    
    # Run tests
    results['Prompt Templates'] = test_prompt_templates()
    results['Data Loading'] = test_data_loading()
    results['Prediction Extraction'] = test_prediction_extraction()
    results['API Connection'] = test_api_connection()
    results['End-to-End'] = test_end_to_end()
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\nAll tests passed! System is ready to use.")
    else:
        print(f"\n{total_tests - passed_tests} test(s) failed. Please check the errors above.")


if __name__ == "__main__":
    main()

