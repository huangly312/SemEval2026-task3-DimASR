"""
Training script for DimABSA tasks using HuggingFace Transformers + PEFT
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
from datetime import datetime
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed
)

from config import load_config
from utils import DataProcessor, load_model_and_tokenizer, prepare_lora_model


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # HuggingFace transformers
    set_seed(seed)
    
    logging.info(f"Random seed set to: {seed}")


def setup_logging(output_dir: str) -> logging.Logger:
    """
    Setup logging to both console and file
    
    Args:
        output_dir: Directory to save log file
        
    Returns:
        Logger instance
    """
    # Create logs directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    
    return logger


class MetricTracker(TrainerCallback):
    """
    Callback to track best metrics and log training progress
    """
    
    def __init__(self, metric_for_best_model: str = "eval_loss"):
        """
        Args:
            metric_for_best_model: Metric name to track for best model
        """
        self.metric_for_best_model = metric_for_best_model
        self.best_metric = float('inf')
        self.best_checkpoint = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging - log training metrics to file"""
        if logs is None:
            return
        
        # Log training metrics
        log_msg = f"Step {state.global_step}"
        if 'loss' in logs:
            log_msg += f" | Train Loss: {logs['loss']:.4f}"
        if 'eval_loss' in logs:
            log_msg += f" | Val Loss: {logs['eval_loss']:.4f}"
        if 'learning_rate' in logs:
            log_msg += f" | LR: {logs['learning_rate']:.2e}"
        if 'epoch' in logs:
            log_msg += f" | Epoch: {logs['epoch']:.2f}"
        if 'grad_norm' in logs:
            log_msg += f" | Grad Norm: {logs['grad_norm']:.4f}"
            
        logging.info(log_msg)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics is None:
            return
            
        current_metric = metrics.get(self.metric_for_best_model)
        if current_metric is not None and current_metric < self.best_metric:
            self.best_metric = current_metric
            self.best_checkpoint = f"checkpoint-{state.global_step}"
            logging.info(f"🎯 New best model! {self.metric_for_best_model}: {current_metric:.4f}")
            
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        if self.best_checkpoint:
            logging.info(f"✅ Best checkpoint: {self.best_checkpoint} "
                        f"with {self.metric_for_best_model}: {self.best_metric:.4f}")
        else:
            logging.info("No evaluation performed during training")


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train LoRA model for DimABSA tasks"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
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
        default=None,  # Use None as default instead of False
        help="Use local model instead of downloading from HuggingFace"
    )
    
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Path to local model directory"
    )
    
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--exclude_null",
        action="store_true",
        default=None,
        help="Exclude NULL aspects"
    )
    
    parser.add_argument(
        "--no_exclude_null",
        action="store_true",
        default=None,
        help="Include NULL aspects"
    )
    
    return parser.parse_args()


def preprocess_function(examples, tokenizer, max_seq_length):
    """
    Tokenize examples for training
    
    Args:
        examples: Batch of examples with 'text' field
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",  # Pad to max_length for consistent batch size
    )
    
    # Set labels to input_ids for causal language modeling
    result["labels"] = result["input_ids"].copy()
    
    return result


def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory first
    output_dir = config.get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("="*60)
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info("="*60)
    
    # Debug: print config BEFORE update
    logger.info(f"DEBUG - BEFORE update_from_args:")
    logger.info(f"  use_local_model: {config.model.use_local_model}")
    logger.info(f"  local_model_path: {config.model.local_model_path}")
    
    # Update config from command line arguments FIRST
    arg_dict = vars(args)
    if arg_dict.get('exclude_null'):
        arg_dict['exclude_null'] = True
    elif arg_dict.get('no_exclude_null'):
        arg_dict['exclude_null'] = False
    
    logger.info(f"DEBUG - Command line args: {arg_dict}")
    config.update_from_args(**arg_dict)
    
    # Debug: print config AFTER update
    logger.info(f"DEBUG - AFTER update_from_args:")
    logger.info(f"  use_local_model: {config.model.use_local_model}")
    logger.info(f"  local_model_path: {config.model.local_model_path}")
    
    # Print final config
    logger.info(f"\n{config}")
    logger.info("="*60)
    
    # Set random seed for reproducibility
    seed = config.training.seed
    set_random_seed(seed)
    
    # Get data paths
    train_file = config.get_train_file_path()
    
    logger.info(f"Training data: {train_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if training file exists
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    # Initialize data processor
    data_processor = DataProcessor(
        task_type=config.task.task_type,
        domain=config.task.domain,
        exclude_null=config.task.exclude_null
    )
    
    # Load and process training data
    logger.info("Loading training data...")
    full_dataset = data_processor.load_train_dataset(train_file)
    logger.info(f"Total samples: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        raise ValueError("No training samples found after filtering")
    
    # Split into train and validation (80/20)
    train_val_split = full_dataset.train_test_split(test_size=0.2, seed=config.training.seed)
    train_dataset = train_val_split['train']
    eval_dataset = train_val_split['test']
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")
    
    # Print sample
    logger.info("Sample training example:")
    logger.info(train_dataset[0]['text'][:500] + "...")
    logger.info("")
    
    # Load model and tokenizer
    logger.info("="*60)
    logger.info("Loading model and tokenizer")
    logger.info("="*60)
    
    # Get actual model path (local or remote)
    # Debug: print model config
    logger.info(f"DEBUG - use_local_model: {config.model.use_local_model}")
    logger.info(f"DEBUG - local_model_path: {config.model.local_model_path}")
    logger.info(f"DEBUG - model_id: {config.model.model_id}")
    
    model_path = config.get_model_path()
    if config.model.use_local_model:
        logger.info(f"Using local model from: {model_path}")
    else:
        logger.info(f"Downloading model from HuggingFace: {model_path}")
    
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
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.model.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.model.max_seq_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval"
    )
    
    # Prepare LoRA model
    logger.info("="*60)
    logger.info("Preparing LoRA model")
    logger.info("="*60)
    
    # Don't save embedding layers to reduce trainable parameters
    # The warning is harmless and can be ignored
    model = prepare_lora_model(
        model=model,
        r=config.model.lora.get('r', 16),
        lora_alpha=config.model.lora.get('lora_alpha', 16),
        lora_dropout=config.model.lora.get('lora_dropout', 0.05),
        bias=config.model.lora.get('bias', 'none'),
        task_type=config.model.lora.get('task_type', 'CAUSAL_LM'),
        target_modules=config.model.lora.get('target_modules'),
        use_gradient_checkpointing=config.training.gradient_checkpointing,
        modules_to_save=None  # Don't save embeddings to keep parameters efficient
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        lr_scheduler_type=config.training.lr_scheduler_type,
        report_to=config.training.report_to,
        seed=config.training.seed,
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="steps",
        eval_strategy="steps",  # Use eval_strategy for compatibility
        eval_steps=config.training.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        logging_first_step=True,
        save_safetensors=True,
    )
    
    # Data collator for language modeling
    # Note: We use padding="max_length" in preprocessing, so pad_to_multiple_of is not needed
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8  # Pad to multiple of 8 for better GPU efficiency
    )
    
    # Initialize metric tracker callback
    metric_tracker = MetricTracker(metric_for_best_model="loss")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[metric_tracker],
    )
    
    # Log training info
    logger.info("="*60)
    logger.info("Starting training")
    logger.info("="*60)
    logger.info(f"Number of training samples: {len(tokenized_train)}")
    logger.info(f"Number of validation samples: {len(tokenized_eval)}")
    logger.info(f"Number of epochs: {config.training.num_train_epochs}")
    logger.info(f"Batch size per device: {config.training.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    logger.info(f"Total batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Total optimization steps: {len(tokenized_train) // (config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps) * config.training.num_train_epochs}")
    logger.info(f"Evaluation every {config.training.save_steps} steps")
    logger.info("")
    
    # Train
    train_result = trainer.train()
    
    # Log training results
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info("="*60)
    logger.info(f"Total training time: {train_result.metrics.get('train_runtime', 0):.2f} seconds")
    logger.info(f"Training loss: {train_result.metrics.get('train_loss', 0):.4f}")
    logger.info(f"Samples per second: {train_result.metrics.get('train_samples_per_second', 0):.2f}")
    logger.info(f"Steps per second: {train_result.metrics.get('train_steps_per_second', 0):.2f}")
    
    # Save final model
    logger.info("")
    logger.info("="*60)
    logger.info("Saving final model")
    logger.info("="*60)
    
    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Save training metrics to file
    metrics_file = os.path.join(output_dir, "training_metrics.txt")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("Training Metrics\n")
        f.write("="*60 + "\n")
        for key, value in train_result.metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        if metric_tracker.best_checkpoint:
            f.write(f"Best checkpoint: {metric_tracker.best_checkpoint}\n")
            f.write(f"Best {metric_tracker.metric_for_best_model}: {metric_tracker.best_metric:.4f}\n")
    
    logger.info(f"Model saved to: {final_output_dir}")
    logger.info(f"Training metrics saved to: {metrics_file}")
    logger.info("")
    logger.info("Training completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()


