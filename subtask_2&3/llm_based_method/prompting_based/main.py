"""
Online LLM-based prediction script for DimABSA tasks (Task 2 & Task 3)
"""

import os
import argparse
import json
from tqdm import tqdm
from openai import OpenAI

from data_processor import DataProcessor, PromptTemplate


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run DimABSA prediction using online LLM API"
    )
    
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["task2", "task3"],
        help="Task type: task2 or task3"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["eng", "jpn", "rus", "tat", "ukr", "zho"],
        help="Language code"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["restaurant", "laptop", "hotel", "finance"],
        help="Domain"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process first 5 samples and print results (no file saved)"
    )
    
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode for model"
    )
    
    return parser.parse_args()


def get_default_input_path(task_type, language, domain):
    """
    Generate default input file path based on task, language, and domain
    
    Args:
        task_type: "task2" or "task3"
        language: Language code
        domain: Domain name
        
    Returns:
        Default input file path
    """
    subtask = "subtask_2" if task_type == "task2" else "subtask_3"
    filename = f"{language}_{domain}_dev_{task_type}.jsonl"
    return f"data/raw_track_a/{subtask}/{language}/{filename}"


def get_output_path(task_type, language, domain, thinking=False):
    """
    Generate output file path
    
    Args:
        task_type: "task2" or "task3"
        language: Language code
        domain: Domain name
        thinking: Whether thinking mode is enabled
        
    Returns:
        Output file path
    """
    subtask = "subtask_2" if task_type == "task2" else "subtask_3"
    thinking_suffix = "_thinking" if thinking else ""
    filename = f"pred_Qwen3-32B_{language}_{domain}{thinking_suffix}.jsonl"
    return os.path.join("output", subtask, filename)


def generate_prediction(client, messages, thinking=False):
    """
    Generate prediction using OpenAI API
    
    Args:
        client: OpenAI client
        messages: List of message dictionaries
        thinking: Whether to enable thinking mode
        
    Returns:
        Generated text string
    """
    try:
        extra_body = {"top_k": 20}
        if thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        
        response = client.chat.completions.create(
            model="Qwen3-32B",
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body=extra_body,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating prediction: {e}")
        return ""


def main():
    """Main prediction function"""
    
    # Parse arguments
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("DimABSA Online LLM Prediction")
    print(f"{'='*60}\n")
    
    # Determine input and output files
    input_file = get_default_input_path(args.task_type, args.language, args.domain)
    output_file = get_output_path(args.task_type, args.language, args.domain, args.thinking)
    
    print(f"Task: {args.task_type} | Language: {args.language} | Domain: {args.domain}")
    print(f"Thinking: {'Enabled' if args.thinking else 'Disabled'}")
    if args.test:
        print(f"Mode: TEST (first 5 samples only, no file saved)")
    print(f"Input:  {input_file}")
    if not args.test:
        print(f"Output: {output_file}")
    print(f"\n{'='*60}\n")
    
    # Check input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://10.208.62.156:8002/v1",
    )
    
    # Initialize data processor
    data_processor = DataProcessor(
        task_type=args.task_type,
        domain=args.domain,
        exclude_null=True
    )
    
    # Load development dataset
    print("Loading input data...")
    dev_dataset = data_processor.load_dev_dataset(input_file)
    dev_dataset = dev_dataset.select(range(min(3, len(dev_dataset))))
    
    # Limit to 5 samples in test mode
    if args.test:
        dev_dataset = dev_dataset.select(range(min(5, len(dev_dataset))))
        print(f"Test mode: processing {len(dev_dataset)} samples\n")
    else:
        print(f"Total samples: {len(dev_dataset)}\n")
    
    # Run predictions
    print("Running predictions...")
    print(f"{'='*60}\n")
    
    results = []
    key = "Triplet" if args.task_type == "task2" else "Quadruplet"
    
    for i, sample in enumerate(tqdm(dev_dataset, desc="Predicting", disable=args.test)):
        # Format for inference
        messages = data_processor.format_for_inference(sample)
        
        # Generate prediction
        generated_text = generate_prediction(client, messages, args.thinking)
        
        # Extract structured predictions
        extracted = data_processor.extract_predictions(
            generated_text,
            args.task_type
        )
        
        # Create output record
        result = {
            "ID": sample.get("ID", f"sample_{i}"),
            "Text": sample["Text"],
            key: extracted
        }
        
        results.append(result)
        
        # Print results in test mode
        if args.test:
            print(f"\n{'='*60}")
            print(f"Sample {i+1}/{len(dev_dataset)}")
            print(f"{'='*60}")
            print(f"ID: {result['ID']}")
            print(f"Text: {result['Text']}")
            print(f"\nGenerated output:\n{generated_text}")
            print(f"\nExtracted {key}:")
            for item in extracted:
                print(f"  {json.dumps(item, ensure_ascii=False)}")
            if not extracted:
                print(f"  (empty)")
    
    # Save predictions (skip in test mode)
    if args.test:
        print(f"\n{'='*60}")
        print(f"Test mode completed! Processed {len(results)} samples")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        data_processor.save_predictions(results, output_file)
        print(f"Completed! Total: {len(results)} samples")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
