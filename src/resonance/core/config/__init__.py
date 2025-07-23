"""Resonance Configuration Management.

This module provides a centralized configuration system for the Resonance framework,
supporting different training algorithms, model configurations, and experiment setups.

The configuration system uses a hierarchical approach:
- Base configurations for common settings
- Algorithm-specific configurations (SFT, DPO, PPO, RM)
- Model-specific configurations (LLaVA, Qwen-VL, etc.)
- Experiment configurations combining multiple settings

Key Components:
    - BaseConfig: Core configuration interface
    - TrainingConfig: Training-specific settings
    - ModelConfig: Model architecture settings
    - DataConfig: Dataset and preprocessing settings
    - EvalConfig: Evaluation benchmark settings

Example:
    ```python
    from resonance.core.config import SFTConfig

    config = SFTConfig(
        model_name="Qwen/Qwen-VL-Chat",
        dataset_name="vlquery_json",
        output_dir="./output/sft"
    )
    ```

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from .base import BaseConfig, TrainingConfig, ModelConfig, DataConfig, EvalConfig
from .algorithms import SFTConfig, DPOConfig, PPOConfig, RewardModelConfig

# Note: Model-specific configs would be implemented in future versions
# from .models import LlavaConfig, QwenVLConfig, InstructBlipConfig, InternLMConfig
from .factory import create_config, load_config_from_file

__all__ = [
    # Base configurations
    "BaseConfig",
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    "EvalConfig",
    # Algorithm configurations
    "SFTConfig",
    "DPOConfig",
    "PPOConfig",
    "RewardModelConfig",
    # Model configurations (future implementation)
    # "LlavaConfig",
    # "QwenVLConfig",
    # "InstructBlipConfig",
    # "InternLMConfig",
    # Factory functions
    "create_config",
    "load_config_from_file",
]
