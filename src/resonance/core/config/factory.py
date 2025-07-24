"""Resonance Configuration Factory Functions.

This module provides factory functions for creating and loading configurations
from various sources including files, command-line arguments, and dictionaries.

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

import json
from pathlib import Path
from typing import Dict, Any, Union, Type

import yaml

from .base import BaseConfig
from .algorithms import SFTConfig, DPOConfig, PPOConfig, RewardModelConfig


def create_config(config_type: str, **kwargs) -> BaseConfig:
    """Create a configuration of the specified type.

    Args:
        config_type: Type of configuration ('sft', 'dpo', 'ppo', 'rm')
        **kwargs: Configuration parameters

    Returns:
        Configuration instance
    """
    config_map = {
        "sft": SFTConfig,
        "dpo": DPOConfig,
        "ppo": PPOConfig,
        "rm": RewardModelConfig,
        "reward_model": RewardModelConfig,
    }

    if config_type not in config_map:
        raise ValueError(f"Unknown config type: {config_type}")

    config_class = config_map[config_type]
    return config_class(**kwargs)


def load_config_from_file(file_path: Union[str, Path]) -> BaseConfig:
    """Load configuration from file.

    Args:
        file_path: Path to configuration file

    Returns:
        Loaded configuration instance
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    # Load file content based on extension
    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif file_path.suffix.lower() in [".yaml", ".yml"]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Extract config type
    if "config_type" not in data:
        raise ValueError("Configuration file must specify 'config_type'")

    config_type = data.pop("config_type")

    # Create configuration
    return create_config(config_type, **data)
