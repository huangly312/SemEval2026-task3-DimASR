"""
Test script to verify the setup and basic functionality
"""

import sys
import torch


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"  ERROR: transformers not found - {e}")
        return False
    
    try:
        import peft
        print(f"  peft: {peft.__version__}")
    except ImportError as e:
        print(f"  ERROR: peft not found - {e}")
        return False
    
    try:
        import bitsandbytes
        print(f"  bitsandbytes: {bitsandbytes.__version__}")
    except ImportError as e:
        print(f"  ERROR: bitsandbytes not found - {e}")
        return False
    
    try:
        import datasets
        print(f"  datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"  ERROR: datasets not found - {e}")
        return False
    
    try:
        import accelerate
        print(f"  accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"  ERROR: accelerate not found - {e}")
        return False
    
    try:
        import yaml
        print(f"  pyyaml: installed")
    except ImportError as e:
        print(f"  ERROR: pyyaml not found - {e}")
        return False
    
    print("  All imports successful!")
    return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("  WARNING: CUDA not available. Training will be very slow on CPU.")
    
    return True


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import load_config
        config = load_config("config.yaml")
        print(f"  Configuration loaded successfully")
        print(f"  Task: {config.task.task_type}")
        print(f"  Model: {config.model.model_id}")
        print(f"  Output: {config.get_output_dir()}")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to load configuration - {e}")
        return False


def test_data_processor():
    """Test data processor"""
    print("\nTesting data processor...")
    
    try:
        from utils import DataProcessor
        
        processor = DataProcessor(
            task_type="task2",
            domain="restaurant",
            exclude_null=True
        )
        print(f"  DataProcessor initialized successfully")
        print(f"  Task type: task2")
        
        # Test prompt template
        if processor.instruction:
            print(f"  Prompt template loaded (length: {len(processor.instruction)} chars)")
        
        return True
    except Exception as e:
        print(f"  ERROR: Failed to initialize DataProcessor - {e}")
        return False


def test_model_utils():
    """Test model utilities"""
    print("\nTesting model utilities...")
    
    try:
        from utils import load_model_and_tokenizer, prepare_lora_model
        print(f"  Model utilities imported successfully")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to import model utilities - {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("DimABSA Setup Test")
    print("="*60)
    
    tests = [
        test_imports,
        test_cuda,
        test_config,
        test_data_processor,
        test_model_utils,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! Setup is complete.")
        return 0
    else:
        print(f"Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


