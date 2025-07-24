"""Resonance Base Configuration Classes.

This module defines the foundational configuration classes that serve as building
blocks for all Resonance configurations. These classes provide type safety,
validation, and serialization capabilities for experiment settings.

The base configuration system follows these principles:
- Immutability: Configurations are frozen after creation
- Type safety: All fields have proper type hints and validation
- Serialization: Configs can be saved/loaded as JSON/YAML
- Inheritance: Specialized configs extend base classes
- Validation: Input validation with helpful error messages

Classes:
    BaseConfig: Abstract base for all configurations
    TrainingConfig: Training loop and optimization settings
    ModelConfig: Model architecture and loading settings
    DataConfig: Dataset and preprocessing settings
    EvalConfig: Evaluation benchmark and metric settings

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass(frozen=True)
class BaseConfig(ABC):
    """Abstract base class for all Resonance configurations.

    Provides common functionality for configuration serialization,
    validation, and merging. All configuration classes should inherit
    from this base class.

    Features:
        - Immutable after creation (frozen dataclass)
        - JSON/YAML serialization support
        - Configuration merging and updating
        - Validation framework
        - Pretty printing for debugging
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, BaseConfig):
                result[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    item.to_dict() if isinstance(item, BaseConfig) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string.

        Args:
            indent: Number of spaces for indentation

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_yaml(self) -> str:
        """Convert configuration to YAML string.

        Returns:
            YAML string representation
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file.

        Supports both JSON (.json) and YAML (.yaml, .yml) formats
        based on file extension.

        Args:
            file_path: Path where to save the configuration
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() == ".json":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration parameters.

        Should raise ValueError with descriptive message if
        configuration is invalid.

        Raises:
            ValueError: If configuration is invalid
        """
        pass


@dataclass(frozen=True)
class TrainingConfig(BaseConfig):
    """Training loop and optimization configuration.

    Contains settings for the training process including batch sizes,
    learning rates, optimization parameters, and training duration.

    Attributes:
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Steps to accumulate gradients
        num_train_epochs: Number of training epochs
        max_steps: Maximum training steps (overrides epochs if set)
        learning_rate: Peak learning rate
        weight_decay: L2 regularization coefficient
        warmup_ratio: Fraction of steps for learning rate warmup
        lr_scheduler_type: Learning rate scheduler type
        save_strategy: When to save checkpoints ('steps' or 'epoch')
        save_steps: Steps between checkpoint saves
        eval_strategy: When to evaluate ('steps', 'epoch', or 'no')
        eval_steps: Steps between evaluations
        logging_steps: Steps between log outputs
        gradient_checkpointing: Enable gradient checkpointing to save memory
        fp16: Use 16-bit floating point training
        bf16: Use bfloat16 training (better than fp16 on modern hardware)
        dataloader_num_workers: Number of data loading workers
        remove_unused_columns: Remove unused columns from dataset
        report_to: Experiment tracking services (wandb, tensorboard, etc.)
        run_name: Experiment run name for tracking
        seed: Random seed for reproducibility
    """

    # Batch and accumulation settings
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # Training duration
    num_train_epochs: float = 3.0
    max_steps: int = -1  # -1 means use num_train_epochs

    # Optimization settings
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    # Checkpointing and evaluation
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_steps: int = 10

    # Memory optimization
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False

    # Experiment tracking
    report_to: Optional[List[str]] = field(default_factory=lambda: ["none"])
    run_name: Optional[str] = None
    seed: int = 42

    def validate(self) -> None:
        """Validate training configuration."""
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")

        if self.save_strategy not in ["steps", "epoch"]:
            raise ValueError("save_strategy must be 'steps' or 'epoch'")

        if self.eval_strategy not in ["steps", "epoch", "no"]:
            raise ValueError("eval_strategy must be 'steps', 'epoch', or 'no'")

        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    """Model architecture and loading configuration.

    Contains settings for model loading, architecture modifications,
    and parameter-efficient training techniques like LoRA.

    Attributes:
        model_name_or_path: Hugging Face model name or local path
        model_revision: Specific model revision/branch to use
        torch_dtype: PyTorch data type for model parameters
        trust_remote_code: Allow loading remote code for custom models
        use_auth_token: Use authentication token for private models
        freeze_vision_tower: Freeze vision encoder parameters
        freeze_language_model: Freeze language model parameters
        use_flash_attention_2: Use FlashAttention-2 for efficiency
        use_lora: Enable LoRA parameter-efficient training
        lora_r: LoRA rank (dimension of adaptation)
        lora_alpha: LoRA scaling parameter
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA adaptation
        modules_to_save: Modules to fully update (not LoRA)
        quantization_config: Model quantization settings
        device_map: Device placement strategy
        max_memory: Maximum memory per device
    """

    # Model loading
    model_name_or_path: str
    model_revision: str = "main"
    torch_dtype: Optional[str] = None
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None

    # Architecture modifications
    freeze_vision_tower: bool = True
    freeze_language_model: bool = False
    use_flash_attention_2: bool = False

    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None

    # Quantization and device settings
    quantization_config: Optional[Dict[str, Any]] = None
    device_map: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None

    def validate(self) -> None:
        """Validate model configuration."""
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path is required")

        if self.use_lora:
            if self.lora_r <= 0:
                raise ValueError("lora_r must be positive when using LoRA")

            if self.lora_alpha <= 0:
                raise ValueError("lora_alpha must be positive when using LoRA")

            if not 0 <= self.lora_dropout <= 1:
                raise ValueError("lora_dropout must be between 0 and 1")


@dataclass(frozen=True)
class DataConfig(BaseConfig):
    """Dataset and preprocessing configuration.

    Contains settings for data loading, preprocessing, tokenization,
    and augmentation for vision-language training.

    Attributes:
        dataset_name: Name of the dataset type
        data_path: Path to dataset file(s)
        image_root: Root directory for images
        data_ratio: Fraction of data to use (for quick experiments)
        max_length: Maximum sequence length for tokenization
        max_prompt_length: Maximum prompt length (for preference training)
        max_target_length: Maximum target length (for preference training)
        preprocessing_num_workers: Number of workers for data preprocessing
        dataloader_pin_memory: Pin memory in data loaders
        ignore_data_skip: Ignore resuming from data checkpoint
        streaming: Use streaming datasets for large data
        cache_dir: Directory for caching processed datasets
    """

    # Dataset specification
    dataset_name: str
    data_path: str
    image_root: Optional[str] = None
    data_ratio: float = 1.0

    # Tokenization settings
    max_length: int = 512
    max_prompt_length: int = 256
    max_target_length: int = 256

    # Data loading settings
    preprocessing_num_workers: int = 4
    dataloader_pin_memory: bool = True
    ignore_data_skip: bool = False
    streaming: bool = False
    cache_dir: Optional[str] = None

    def validate(self) -> None:
        """Validate data configuration."""
        if not self.dataset_name:
            raise ValueError("dataset_name is required")

        if not self.data_path:
            raise ValueError("data_path is required")

        if not 0 < self.data_ratio <= 1:
            raise ValueError("data_ratio must be between 0 and 1")

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        if self.max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive")

        if self.max_target_length <= 0:
            raise ValueError("max_target_length must be positive")


@dataclass(frozen=True)
class EvalConfig(BaseConfig):
    """Evaluation benchmark and metric configuration.

    Contains settings for running evaluations on various vision-language
    benchmarks and computing metrics.

    Attributes:
        eval_dataset_name: Name of evaluation dataset
        eval_data_path: Path to evaluation data
        eval_image_root: Root directory for evaluation images
        eval_batch_size: Batch size for evaluation
        metric_for_best_model: Metric to use for model selection
        greater_is_better: Whether higher metric values are better
        evaluation_strategy: When to run evaluation
        save_eval_results: Save detailed evaluation results
        eval_accumulation_steps: Steps to accumulate eval outputs
    """

    eval_dataset_name: Optional[str] = None
    eval_data_path: Optional[str] = None
    eval_image_root: Optional[str] = None
    eval_batch_size: int = 8
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    evaluation_strategy: str = "steps"
    save_eval_results: bool = True
    eval_accumulation_steps: Optional[int] = None

    def validate(self) -> None:
        """Validate evaluation configuration."""
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")

        if self.evaluation_strategy not in ["steps", "epoch", "no"]:
            raise ValueError("evaluation_strategy must be 'steps', 'epoch', or 'no'")
