"""
Model utilities for loading and configuring models with LoRA
"""

import torch
from typing import Tuple, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)


def load_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 512,
    load_in_4bit: bool = True,
    use_flash_attention_2: bool = False,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_quant_type: str = "nf4",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base model and tokenizer with optional 4-bit quantization
    
    Args:
        model_id: HuggingFace model identifier
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load model in 4-bit precision
        use_flash_attention_2: Whether to use Flash Attention 2
        bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
        bnb_4bit_use_double_quant: Whether to use double quantization
        bnb_4bit_quant_type: Quantization type (nf4 or fp4)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_id}")
    
    # Configure quantization
    quantization_config = None
    if load_in_4bit:
        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        compute_dtype = compute_dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=bnb_4bit_quant_type
        )
        print(f"Using 4-bit quantization with {bnb_4bit_quant_type} and compute dtype {bnb_4bit_compute_dtype}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Model loading arguments
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if not load_in_4bit else None,
        "device_map": "auto",
    }
    
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    if use_flash_attention_2:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except Exception as e:
            print(f"Flash Attention 2 not available: {e}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure model
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    print(f"Model loaded successfully")
    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {model.device}")
    
    return model, tokenizer


def prepare_lora_model(
    model: AutoModelForCausalLM,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    target_modules: Optional[list] = None,
    use_gradient_checkpointing: bool = True,
    modules_to_save: Optional[list] = None
) -> PeftModel:
    """
    Prepare model for LoRA fine-tuning
    
    Args:
        model: Base model to apply LoRA
        r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout rate
        bias: Bias training strategy
        task_type: Task type for LoRA
        target_modules: List of module names to apply LoRA
        use_gradient_checkpointing: Whether to use gradient checkpointing
        modules_to_save: List of modules to save (e.g., embedding layers)
        
    Returns:
        PEFT model with LoRA adapters
    """
    print("Preparing model for LoRA fine-tuning")
    
    # Default target modules if not specified
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Prepare model for k-bit training if quantized
    if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        print("Model prepared for 4-bit training")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,
        modules_to_save=modules_to_save,  # Save embedding layers if resized
        inference_mode=False
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    return model


def print_trainable_parameters(model: PeftModel):
    """
    Print the number of trainable parameters in the model
    
    Args:
        model: PEFT model
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {100 * trainable_params / all_params:.4f}%"
    )


def load_lora_model(
    base_model_id: str,
    lora_checkpoint_path: str,
    max_seq_length: int = 512,
    load_in_4bit: bool = True
) -> Tuple[PeftModel, AutoTokenizer]:
    """
    Load base model with trained LoRA adapters
    
    Args:
        base_model_id: HuggingFace base model identifier
        lora_checkpoint_path: Path to LoRA checkpoint directory
        max_seq_length: Maximum sequence length
        load_in_4bit: Whether to load in 4-bit precision
        
    Returns:
        Tuple of (lora_model, tokenizer)
    """
    print(f"Loading base model: {base_model_id}")
    print(f"Loading LoRA adapters from: {lora_checkpoint_path}")
    
    # Load base model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_id=base_model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        model,
        lora_checkpoint_path,
        torch_dtype=torch.bfloat16
    )
    
    print("LoRA model loaded successfully")
    
    return model, tokenizer


def merge_and_save_model(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    output_path: str
):
    """
    Merge LoRA weights with base model and save
    
    Args:
        model: PEFT model with LoRA adapters
        tokenizer: Tokenizer
        output_path: Output directory path
    """
    print(f"Merging LoRA weights and saving to: {output_path}")
    
    # Merge weights
    model = model.merge_and_unload()
    
    # Save merged model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("Model and tokenizer saved successfully")


