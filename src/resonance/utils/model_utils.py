"""Resonance Model Utilities.

This module provides utility functions for model operations including loading,
saving, parameter management, and DeepSpeed integration. It consolidates
model-related functionality from the previous common.py and auto_load.py files.

Key Features:
- Safe model loading with error handling and validation
- DeepSpeed Zero Stage 3 parameter management
- Model saving utilities for different training frameworks
- Vision tower extraction and manipulation
- Memory optimization and GPU management

Functions:
- load_model_safe: Safely load models with comprehensive error handling
- save_model_safe: Save models with proper validation and backup
- get_vision_tower: Extract vision components from multi-modal models
- maybe_zero_3: Handle DeepSpeed Zero Stage 3 parameter gathering
- safe_save_model_for_hf_trainer: Save models from HuggingFace trainers
- safe_save_model_for_ppo_trainer: Save models from PPO trainers

The utilities handle various model architectures and training frameworks,
providing a unified interface for model operations across Resonance.

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, Literal

import torch
from transformers import PreTrainedModel, Trainer
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import deepspeed
from trl import PPOTrainer

logger = logging.getLogger(__name__)


def maybe_zero_3(param: torch.Tensor) -> torch.Tensor:
    """Handle DeepSpeed Zero Stage 3 parameter gathering.

    This function safely extracts parameter data when using DeepSpeed Zero Stage 3,
    which partitions model parameters across multiple GPUs. If the parameter is
    partitioned, it gathers the full parameter; otherwise, it returns a copy.

    Args:
        param: Model parameter tensor (potentially partitioned)

    Returns:
        Full parameter tensor on the current device

    Example:
        ```python
        # Get full parameter data even with DeepSpeed ZeRO-3
        full_param = maybe_zero_3(model.layer.weight)
        ```
    """
    if hasattr(param, "ds_id"):
        # DeepSpeed parameter - check if it's partitioned
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            # Parameter is partitioned, need to gather
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            # Parameter is available locally
            param = param.detach().cpu().clone()
    else:
        # Regular parameter
        param = param.detach().cpu().clone()

    return param


def get_vision_tower(model: PreTrainedModel) -> Optional[torch.nn.Module]:
    """Get vision tower from a vision-language model.

    This function extracts the vision tower component from various
    vision-language model architectures. Used for freezing vision
    components during training or accessing vision features.

    Args:
        model: Vision-language model instance

    Returns:
        Vision tower component or None if not found

    Example:
        ```python
        vision_tower = get_vision_tower(model)
        if vision_tower is not None:
            # Freeze vision parameters
            for param in vision_tower.parameters():
                param.requires_grad = False
        ```
    """
    # Check common vision tower attribute names
    vision_tower_attrs = [
        "vision_tower",
        "visual_encoder",
        "vision_model",
        "visual",
        "clip_model",
        "vision_encoder",
    ]

    # Try direct attributes first
    for attr in vision_tower_attrs:
        if hasattr(model, attr):
            vision_component = getattr(model, attr)
            if vision_component is not None:
                return vision_component

    # Check nested attributes (e.g., model.model.vision_tower)
    if hasattr(model, "model"):
        for attr in vision_tower_attrs:
            if hasattr(model.model, attr):
                vision_component = getattr(model.model, attr)
                if vision_component is not None:
                    return vision_component

    # Check for other nested structures
    nested_paths = ["base_model", "transformer", "backbone"]
    for base_attr in nested_paths:
        if hasattr(model, base_attr):
            base_model = getattr(model, base_attr)
            for attr in vision_tower_attrs:
                if hasattr(base_model, attr):
                    vision_component = getattr(base_model, attr)
                    if vision_component is not None:
                        return vision_component

    logger.warning("Could not find vision tower in model")
    return None


def load_model_safe(
    model_path: Union[str, Path],
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
) -> PreTrainedModel:
    """Safely load a model with comprehensive error handling.

    This function provides a robust interface for loading models with
    proper validation, error handling, and automatic fallback strategies.

    Args:
        model_path: Path to the model directory or HuggingFace model name
        device_map: Device placement strategy
        torch_dtype: PyTorch data type for model parameters
        **kwargs: Additional arguments passed to model loading

    Returns:
        Loaded model instance

    Raises:
        ValueError: If model path is invalid or model cannot be loaded
        RuntimeError: If model loading fails due to system constraints

    Example:
        ```python
        model = load_model_safe(
            "Qwen/Qwen-VL-Chat",
            device_map="auto",
            torch_dtype=torch.float16
        )
        ```
    """
    from transformers import AutoModelForCausalLM

    model_path = str(model_path)

    try:
        logger.info(f"Loading model from: {model_path}")

        # Prepare loading arguments
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,  # Required for many VL models
            **kwargs,
        }

        # Remove None values
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        logger.info(f"Successfully loaded model: {model.__class__.__name__}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e


def save_model_safe(
    model: PreTrainedModel,
    output_path: Union[str, Path],
    tokenizer: Optional[Any] = None,
    create_backup: bool = True,
    **kwargs,
) -> None:
    """Safely save a model with validation and backup.

    Args:
        model: Model to save
        output_path: Directory to save the model
        tokenizer: Optional tokenizer to save alongside model
        create_backup: Whether to create backup if directory exists
        **kwargs: Additional arguments for saving

    Raises:
        ValueError: If output path is invalid
        RuntimeError: If saving fails

    Example:
        ```python
        save_model_safe(
            model,
            "output/my_model",
            tokenizer=tokenizer,
            create_backup=True
        )
        ```
    """
    output_path = Path(output_path)

    try:
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Create backup if requested and directory has content
        if create_backup and any(output_path.iterdir()):
            backup_path = output_path.with_suffix(".backup")
            if backup_path.exists():
                import shutil

                shutil.rmtree(backup_path)
            output_path.rename(backup_path)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created backup at: {backup_path}")

        # Save the model
        logger.info(f"Saving model to: {output_path}")
        model.save_pretrained(output_path, **kwargs)

        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path)
            logger.info("Saved tokenizer alongside model")

        logger.info("Model saved successfully")

    except Exception as e:
        logger.error(f"Failed to save model to {output_path}: {e}")
        raise RuntimeError(f"Model saving failed: {e}") from e


def safe_save_model_for_hf_trainer(
    trainer: Trainer, output_dir: Union[str, Path], **kwargs
) -> None:
    """Safely save model from HuggingFace trainer with DeepSpeed support.

    This function handles model saving from HuggingFace trainers with
    proper DeepSpeed Zero Stage 3 parameter gathering and state management.

    Args:
        trainer: HuggingFace trainer instance
        output_dir: Directory to save the model
        **kwargs: Additional save arguments

    Example:
        ```python
        safe_save_model_for_hf_trainer(
            trainer,
            "output/final_model"
        )
        ```
    """
    output_dir = Path(output_dir)

    try:
        logger.info(f"Saving HF trainer model to: {output_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Handle DeepSpeed models
        if trainer.is_deepspeed_enabled:
            # For DeepSpeed, we need to save the model state properly
            trainer.accelerator.wait_for_everyone()

            # Save model state
            if trainer.accelerator.is_main_process:
                # Get the unwrapped model for saving
                unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=trainer.accelerator.is_main_process,
                    save_function=trainer.accelerator.save,
                    state_dict=trainer.accelerator.get_state_dict(trainer.model),
                    **kwargs,
                )
        else:
            # Standard saving for non-DeepSpeed models
            trainer.save_model(output_dir, **kwargs)

        # Save tokenizer
        if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(output_dir)

        logger.info("HF trainer model saved successfully")

    except Exception as e:
        logger.error(f"Failed to save HF trainer model: {e}")
        raise RuntimeError(f"HF trainer model saving failed: {e}") from e


def safe_save_model_for_ppo_trainer(
    trainer: PPOTrainer, output_dir: Union[str, Path], **kwargs
) -> None:
    """Safely save model from PPO trainer.

    Args:
        trainer: PPO trainer instance
        output_dir: Directory to save the model
        **kwargs: Additional save arguments

    Example:
        ```python
        safe_save_model_for_ppo_trainer(
            ppo_trainer,
            "output/ppo_model"
        )
        ```
    """
    output_dir = Path(output_dir)

    try:
        logger.info(f"Saving PPO trainer model to: {output_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the policy model
        if hasattr(trainer.model, "save_pretrained"):
            trainer.model.save_pretrained(output_dir, **kwargs)
        else:
            # Fallback to torch save
            torch.save(trainer.model.state_dict(), output_dir / "pytorch_model.bin")

        # Save tokenizer if available
        if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(output_dir)

        logger.info("PPO trainer model saved successfully")

    except Exception as e:
        logger.error(f"Failed to save PPO trainer model: {e}")
        raise RuntimeError(f"PPO trainer model saving failed: {e}") from e


def flatten_list(input_list: list) -> list:
    """Flatten a nested list structure.

    Args:
        input_list: List that may contain nested lists

    Returns:
        Flattened list

    Example:
        ```python
        nested = [[1, 2], [3, 4], [5]]
        flat = flatten_list(nested)  # [1, 2, 3, 4, 5]
        ```
    """
    if not input_list:
        return []

    output = []
    if isinstance(input_list[0], list):
        for item in input_list:
            output.extend(item)
    else:
        output = input_list

    return output


def pad_to_length(
    tensor: torch.Tensor, target_length: int, pad_value: int = 0, dim: int = -1
) -> torch.Tensor:
    """Pad tensor to target length along specified dimension.

    Args:
        tensor: Input tensor to pad
        target_length: Desired length after padding
        pad_value: Value to use for padding
        dim: Dimension to pad along

    Returns:
        Padded tensor

    Example:
        ```python
        # Pad sequence to length 512
        padded = pad_to_length(input_ids, 512, pad_value=tokenizer.pad_token_id)
        ```
    """
    current_length = tensor.size(dim)

    if current_length >= target_length:
        return tensor

    pad_size = target_length - current_length

    # Create padding configuration
    pad_config = [0] * (2 * tensor.ndim)
    pad_config[-(2 * (dim + 1) - 1)] = pad_size

    return torch.nn.functional.pad(tensor, pad_config, value=pad_value)
