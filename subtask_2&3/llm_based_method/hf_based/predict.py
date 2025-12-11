"""
Prediction/Inference script for DimABSA tasks using trained LoRA models
"""

import os
import argparse
import torch
from tqdm import tqdm

from config import load_config
from utils import DataProcessor, load_lora_model


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run inference with trained LoRA model for DimABSA tasks"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["task2", "task3"],
        help="Task type (overrides config)"
    )
    
    parser.add_argument(
        "--subtask",
        type=str,
        choices=["subtask_2", "subtask_3"],
        help="Subtask (overrides config)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        choices=["eng", "jpn", "rus", "tat", "ukr", "zho"],
        help="Language code (overrides config)"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        choices=["restaurant", "laptop", "hotel", "finance"],
        help="Domain (overrides config)"
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        help="Base model identifier (overrides config)"
    )
    
    parser.add_argument(
        "--use_local_model",
        action="store_true",
        default=None,
        help="Use local model instead of downloading from HuggingFace"
    )
    
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Path to local model directory"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file path (overrides config)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file path (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for inference (overrides config)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Maximum new tokens to generate (overrides config)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (overrides config)"
    )
    
    return parser.parse_args()


def generate_prediction(
    model,
    tokenizer,
    messages,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    do_sample=True
):
    """
    Generate prediction for a single example
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        messages: List of message dictionaries
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        
    Returns:
        Generated text string
    """
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    # Try to extract only the assistant's response
    if "<|assistant|>" in generated_text:
        parts = generated_text.split("<|assistant|>")
        if len(parts) > 1:
            return parts[-1].strip()
    
    # Fallback: return everything after the input
    input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    if generated_text.startswith(input_text):
        return generated_text[len(input_text):].strip()
    
    return generated_text


def main():
    """Main inference function"""
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"\n{'='*60}")
    print(f"Configuration loaded from: {args.config}")
    print(f"{'='*60}\n")
    print(config)
    print(f"\n{'='*60}\n")
    
    # Update config from command line arguments
    config.update_from_args(**vars(args))
    
    # Get paths
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = config.get_dev_file_path()
    
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = config.get_prediction_path()
    
    checkpoint_path = args.checkpoint
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Initialize data processor
    data_processor = DataProcessor(
        task_type=config.task.task_type,
        domain=config.task.domain,
        exclude_null=config.task.exclude_null
    )
    
    # Load development data
    print(f"\nLoading input data...")
    dev_dataset = data_processor.load_dev_dataset(input_file)
    print(f"Input samples: {len(dev_dataset)}")
    
    # Load model with LoRA adapters
    print(f"\n{'='*60}")
    print("Loading model with LoRA adapters")
    print(f"{'='*60}\n")
    
    # Get actual model path (local or remote)
    model_path = config.get_model_path()
    if config.model.use_local_model:
        print(f"Using local model from: {model_path}")
    else:
        print(f"Downloading model from HuggingFace: {model_path}")
    
    model, tokenizer = load_lora_model(
        base_model_id=model_path,
        lora_checkpoint_path=checkpoint_path,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.quantization.get('load_in_4bit', True)
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Get inference parameters
    max_new_tokens = args.max_new_tokens if args.max_new_tokens else config.inference.max_new_tokens
    temperature = args.temperature if args.temperature else config.inference.temperature
    top_p = config.inference.top_p
    top_k = config.inference.top_k
    do_sample = config.inference.do_sample
    
    print(f"\nInference parameters:")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  top_k: {top_k}")
    print(f"  do_sample: {do_sample}")
    
    # Run inference
    print(f"\n{'='*60}")
    print("Running inference")
    print(f"{'='*60}\n")
    
    results = []
    key = "Triplet" if config.task.task_type == "task2" else "Quadruplet"
    
    for i, sample in enumerate(tqdm(dev_dataset, desc="Predicting")):
        # Format for inference
        messages = data_processor.format_for_inference(sample)
        
        # Generate prediction
        generated_text = generate_prediction(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample
        )
        
        # Extract structured predictions
        extracted = data_processor.extract_predictions(
            generated_text,
            config.task.task_type
        )
        
        # Create output record
        result = {
            "ID": sample.get("ID", f"sample_{i}"),
            "Text": sample["Text"],
            key: extracted
        }
        
        results.append(result)
        
        # Print first few examples
        if i < 3:
            print(f"\nExample {i+1}:")
            print(f"Text: {sample['Text'][:100]}...")
            print(f"Generated: {generated_text[:200]}...")
            print(f"Extracted: {extracted}")
    
    # Save predictions
    print(f"\n{'='*60}")
    print("Saving predictions")
    print(f"{'='*60}\n")
    
    data_processor.save_predictions(results, output_file)
    
    print(f"\nInference completed successfully!")
    print(f"Total predictions: {len(results)}")


if __name__ == "__main__":
    main()


