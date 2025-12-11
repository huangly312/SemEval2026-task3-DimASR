"""
Quick test script to verify data loading and model initialization
This script tests the pipeline without actually training
"""

import os
import argparse


def test_data_loading(config):
    """Test data loading and processing"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    from utils import DataProcessor
    
    # Initialize processor
    processor = DataProcessor(
        task_type=config.task.task_type,
        domain=config.task.domain,
        exclude_null=config.task.exclude_null
    )
    
    print(f"Task type: {config.task.task_type}")
    print(f"Domain: {config.task.domain}")
    
    # Load training data
    train_file = config.get_train_file_path()
    print(f"\nLoading training data from: {train_file}")
    
    if not os.path.exists(train_file):
        print(f"ERROR: Training file not found!")
        return False
    
    train_dataset = processor.load_train_dataset(train_file)
    print(f"Number of training samples: {len(train_dataset)}")
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        return False
    
    # Show sample
    print(f"\nFirst training sample:")
    print("-" * 60)
    print(train_dataset[0]['text'][:500])
    print("...")
    print("-" * 60)
    
    # Load dev data
    dev_file = config.get_dev_file_path()
    print(f"\nLoading dev data from: {dev_file}")
    
    if not os.path.exists(dev_file):
        print(f"ERROR: Dev file not found!")
        return False
    
    dev_dataset = processor.load_dev_dataset(dev_file)
    print(f"Number of dev samples: {len(dev_dataset)}")
    
    # Test inference formatting
    if len(dev_dataset) > 0:
        messages = processor.format_for_inference(dev_dataset[0])
        print(f"\nInference format example:")
        print("-" * 60)
        print(messages[0]['content'][:300])
        print("...")
        print("-" * 60)
    
    print("\nData loading test: PASSED")
    return True


def test_model_initialization(config):
    """Test model and tokenizer initialization"""
    print("\n" + "="*60)
    print("Testing Model Initialization")
    print("="*60)
    
    import torch
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Skipping model test.")
        return True
    
    try:
        from utils import load_model_and_tokenizer, prepare_lora_model
        
        # Get actual model path (local or remote)
        model_path = config.get_model_path()
        
        if config.model.use_local_model:
            print(f"Loading LOCAL model from: {model_path}")
        else:
            print(f"Loading model: {model_path}")
            print("This may take a few minutes for the first time...")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_id=model_path,
            max_seq_length=config.model.max_seq_length,
            load_in_4bit=config.model.quantization.get('load_in_4bit', True),
            use_flash_attention_2=config.model.use_flash_attention_2,
            bnb_4bit_compute_dtype=config.model.quantization.get(
                'bnb_4bit_compute_dtype', 'bfloat16'
            ),
            bnb_4bit_use_double_quant=config.model.quantization.get(
                'bnb_4bit_use_double_quant', True
            ),
            bnb_4bit_quant_type=config.model.quantization.get(
                'bnb_4bit_quant_type', 'nf4'
            )
        )
        
        print(f"\nModel loaded successfully!")
        print(f"Model device: {model.device}")
        print(f"Model dtype: {model.dtype}")
        
        # Test tokenization
        test_text = "This is a test sentence."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"\nTokenization test:")
        print(f"Input: {test_text}")
        print(f"Token count: {len(tokens['input_ids'][0])}")
        
        # Prepare LoRA model
        print(f"\nPreparing LoRA model...")
        model = prepare_lora_model(
            model=model,
            r=config.model.lora.get('r', 16),
            lora_alpha=config.model.lora.get('lora_alpha', 16),
            lora_dropout=config.model.lora.get('lora_dropout', 0.05),
            bias=config.model.lora.get('bias', 'none'),
            task_type=config.model.lora.get('task_type', 'CAUSAL_LM'),
            target_modules=config.model.lora.get('target_modules'),
            use_gradient_checkpointing=config.training.gradient_checkpointing
        )
        
        print(f"\nModel initialization test: PASSED")
        return True
        
    except Exception as e:
        print(f"\nERROR during model initialization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Quick test of the DimABSA pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--skip_model", action="store_true", help="Skip model initialization test")
    args = parser.parse_args()
    
    print("="*60)
    print("DimABSA Quick Test")
    print("="*60)
    
    # Load configuration
    from config import load_config
    config = load_config(args.config)
    
    print(f"\nConfiguration:")
    print(f"  Task: {config.task.task_type} | {config.task.subtask}")
    print(f"  Language: {config.task.language}")
    print(f"  Domain: {config.task.domain}")
    print(f"  Model: {config.model.model_id}")
    
    # Test data loading
    data_ok = test_data_loading(config)
    
    if not data_ok:
        print("\nData loading test failed. Please check your data files.")
        return 1
    
    # Test model initialization
    if not args.skip_model:
        model_ok = test_model_initialization(config)
        
        if not model_ok:
            print("\nModel initialization test failed.")
            print("Note: You can skip this test with --skip_model flag")
            return 1
    else:
        print("\nSkipping model initialization test")
    
    print("\n" + "="*60)
    print("All tests passed! You can proceed with training.")
    print("="*60)
    print("\nTo start training, run:")
    print("  python train.py --config config.yaml")
    print("\nOr use the example script:")
    print("  bash run_example.sh")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


