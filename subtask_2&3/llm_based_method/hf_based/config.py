"""
Configuration management module for DimABSA tasks
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskConfig:
    """Task-specific configuration"""
    subtask: str = "subtask_2"
    task_type: str = "task2"
    language: str = "zho"
    domain: str = "restaurant"
    exclude_null: bool = True


@dataclass
class ModelConfig:
    """Model configuration including quantization and LoRA"""
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_length: int = 512
    use_flash_attention_2: bool = False
    use_local_model: bool = False
    local_model_path: Optional[str] = None
    quantization: Dict[str, Any] = field(default_factory=dict)
    lora: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    logging_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    report_to: str = "none"
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference generation parameters"""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    do_sample: bool = True
    batch_size: int = 1


@dataclass
class DataConfig:
    """Data paths configuration"""
    data_root: str = "data/raw_track_a"
    train_file_pattern: str = "{lang}_{domain}_train_alltasks.jsonl"
    dev_file_pattern: str = "{lang}_{domain}_dev_{task_type}.jsonl"


@dataclass
class OutputConfig:
    """Output paths configuration"""
    prediction_dir: str = "./predictions"
    model_save_dir: str = "./saved_models"
    log_dir: str = "./logs"


@dataclass
class EnvironmentConfig:
    """Environment variables and settings"""
    hf_endpoint: str = "https://hf-mirror.com"
    cuda_visible_devices: str = "0"
    num_workers: int = 4


class Config:
    """
    Main configuration class that loads and manages all configurations
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self._load_config()
        self._setup_environment()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        self.task = TaskConfig(**config_dict.get('task', {}))
        self.model = ModelConfig(**config_dict.get('model', {}))
        self.training = TrainingConfig(**config_dict.get('training', {}))
        self.inference = InferenceConfig(**config_dict.get('inference', {}))
        self.data = DataConfig(**config_dict.get('data', {}))
        self.output = OutputConfig(**config_dict.get('output', {}))
        self.environment = EnvironmentConfig(**config_dict.get('environment', {}))
    
    def _setup_environment(self):
        """Setup environment variables"""
        os.environ["HF_ENDPOINT"] = self.environment.hf_endpoint
        os.environ["CUDA_VISIBLE_DEVICES"] = self.environment.cuda_visible_devices
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def get_train_file_path(self) -> str:
        """
        Get training data file path
        
        Returns:
            Full path to training data file
        """
        filename = self.data.train_file_pattern.format(
            lang=self.task.language,
            domain=self.task.domain
        )
        return os.path.join(
            self.data.data_root,
            self.task.subtask,
            self.task.language,
            filename
        )
    
    def get_dev_file_path(self) -> str:
        """
        Get development data file path
        
        Returns:
            Full path to development data file
        """
        filename = self.data.dev_file_pattern.format(
            lang=self.task.language,
            domain=self.task.domain,
            task_type=self.task.task_type
        )
        return os.path.join(
            self.data.data_root,
            self.task.subtask,
            self.task.language,
            filename
        )
    
    def get_model_short_name(self) -> str:
        """
        Get short model name from model_id
        
        Returns:
            Short model name (e.g., "qwen3-4b" from "Qwen/Qwen3-4B-Instruct-2507")
        """
        model_path = self.model.local_model_path if self.model.use_local_model else self.model.model_id
        model_name = model_path.split("/")[-1].lower()
        
        # Simplify model name
        if "qwen3" in model_name or "qwen-3" in model_name:
            if "4b" in model_name:
                return "qwen3-4b"
            elif "8b" in model_name:
                return "qwen3-8b"
            elif "14b" in model_name:
                return "qwen3-14b"
        elif "qwen2.5" in model_name or "qwen-2.5" in model_name:
            if "4b" in model_name:
                return "qwen2.5-4b"
            elif "7b" in model_name:
                return "qwen2.5-7b"
            elif "14b" in model_name:
                return "qwen2.5-14b"
        
        # Fallback: return first part of model name
        return model_name.split("-")[0]
    
    def get_output_dir(self) -> str:
        """
        Get model output directory with task-specific naming
        Format: {model}_{subtask}_{lang}_{domain}_{null_flag}_{date}
        
        Returns:
            Output directory path
        """
        model_name = self.get_model_short_name()
        null_suffix = "no_NULL" if self.task.exclude_null else "has_NULL"
        date_str = datetime.now().strftime("%m-%d")  # 月-日格式
        
        dir_name = f"{model_name}_{self.task.subtask}_{self.task.language}_{self.task.domain}_{null_suffix}_{date_str}"
        return os.path.join(self.training.output_dir, dir_name)
    
    def get_prediction_path(self) -> str:
        """
        Get prediction output file path
        Format: pred_{model}_{lang}_{domain}_{null_flag}_{date}.jsonl
        
        Returns:
            Prediction file path
        """
        model_name = self.get_model_short_name()
        null_suffix = "no_NULL" if self.task.exclude_null else "has_NULL"
        date_str = datetime.now().strftime("%m-%d")  # 月-日格式
        
        filename = f"pred_{model_name}_{self.task.language}_{self.task.domain}_{null_suffix}_{date_str}.jsonl"
        pred_dir = os.path.join(self.output.prediction_dir, self.task.subtask)
        os.makedirs(pred_dir, exist_ok=True)
        return os.path.join(pred_dir, filename)
    
    def get_model_path(self) -> str:
        """
        Get actual model path to load (local or remote)
        
        Returns:
            Model path
        """
        if self.model.use_local_model and self.model.local_model_path:
            if not os.path.exists(self.model.local_model_path):
                raise FileNotFoundError(f"Local model not found: {self.model.local_model_path}")
            return self.model.local_model_path
        return self.model.model_id
    
    def update_from_args(self, **kwargs):
        """
        Update configuration from command line arguments
        
        Args:
            **kwargs: Key-value pairs to update configuration
        """
        # Skip internal arguments
        skip_keys = {'config', 'checkpoint', 'input_file', 'output_file', 
                     'exclude_null', 'no_exclude_null'}  # These are handled separately
        
        for key, value in kwargs.items():
            # Skip None values, False for boolean flags, and internal arguments
            if value is None or key in skip_keys:
                continue
            
            # For boolean flags from argparse action="store_true", 
            # False means not provided, so skip it
            if isinstance(value, bool) and value is False:
                continue
            
            if hasattr(self.task, key):
                setattr(self.task, key, value)
            elif hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
            elif hasattr(self.inference, key):
                setattr(self.inference, key, value)
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return (
            f"Config(\n"
            f"  Task: {self.task.task_type} | {self.task.subtask}\n"
            f"  Language: {self.task.language} | Domain: {self.task.domain}\n"
            f"  Model: {self.model.model_id}\n"
            f"  Output: {self.get_output_dir()}\n"
            f")"
        )


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


