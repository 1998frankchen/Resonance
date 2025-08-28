"""Resonance Common Utilities.

This module contains common utility functions used throughout Resonance,
including DeepSpeed integration helpers, parameter management utilities,
and other shared functionality.

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from typing import Literal, Union
import torch
from transformers import Trainer
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import deepspeed
from trl import PPOTrainer
import os


def maybe_zero_3(param):
    """Handle DeepSpeed Zero Stage 3 parameter gathering.

    This function safely extracts parameter data when using DeepSpeed Zero Stage 3,
    which partitions model parameters across multiple GPUs. If the parameter is
    partitioned, it gathers the full parameter; otherwise, it returns a copy.

    Args:
        param: Model parameter tensor (potentially partitioned)

    Returns:
        torch.Tensor: Full parameter tensor on CPU
    """
    if hasattr(param, "ds_id"):
        # Parameter is partitioned with DeepSpeed Zero Stage 3
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        # Regular parameter - just detach and copy to CPU
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, lora_args):
    """Extract PEFT (LoRA) state dict with DeepSpeed Zero Stage 3 support.

    This function extracts the state dictionary for PEFT adapters while handling
    DeepSpeed Zero Stage 3 parameter partitioning. It processes LoRA parameters
    and modules marked for saving.

    Args:
        named_params: Iterator of (name, parameter) tuples from model
        lora_args: LoRA configuration arguments

    Returns:
        dict: State dictionary containing PEFT parameters
    """
    bias = lora_args.lora_bias

    def is_module_to_save(name):
        """Check if a module should be saved based on configuration.

        Args:
            name: Module name to check

        Returns:
            bool: True if module should be saved
        """
        if lora_args.modules_to_save is None:
            return False
        for module_name in lora_args.modules_to_save:
            if module_name in name:
                return True
        return False

    if bias == "none":
        to_return = {
            k: t for k, t in named_params if "lora_" in k or is_module_to_save(k)
        }
    elif bias == "all":
        to_return = {
            k: t
            for k, t in named_params
            if "lora_" in k or "bias" in k or is_module_to_save(k)
        }
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
            elif is_module_to_save(k):
                to_return[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def pad_to_length(
    tensor: torch.Tensor,
    length: int,
    pad_value: Union[int, float],
    dim: int = -1,
    padding_side: Literal["right", "left"] = "right",
) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        if padding_side == "right":
            return torch.cat(
                [
                    tensor,
                    pad_value
                    * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                ],
                dim=dim,
            )
        elif padding_side == "left":
            return torch.cat(
                [
                    pad_value
                    * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                    tensor,
                ],
                dim=dim,
            )
        else:
            raise ValueError(f"Unknown padding_side: {padding_side}")


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str, lora_args: None):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    # ? why if-else?
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), lora_args
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def safe_save_model_for_ppo_trainer(
    trainer: PPOTrainer, output_dir: str, lora_args: None
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if not trainer.accelerator.is_main_process:
        return
    print("saving")
    os.makedirs(output_dir, exist_ok=True)
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    if trainer.config.use_lora:
        v_head = unwrapped_model.state_dict()
        torch.save(v_head, os.path.join(output_dir, "pytorch_model.bin"))
        state_dict = get_peft_state_maybe_zero_3(
            unwrapped_model.pretrained_model.named_parameters(), lora_args
        )
        unwrapped_model.pretrained_model.save_pretrained(
            output_dir, state_dict=state_dict
        )
    else:
        trainer._save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    print("model saved")


def flatten_list(l: list):  # noqa
    output = []
    if isinstance(l[0], list):
        for item in l:
            output.extend(item)
    else:
        output = l
    return output


def get_vision_tower(model):
    """Get vision tower from a vision-language model.

    This function extracts the vision tower component from various
    vision-language model architectures. Used for freezing vision
    components during training.

    Args:
        model: Vision-language model instance

    Returns:
        Vision tower component or None if not found
    """
    # Check common vision tower attribute names
    vision_tower_attrs = [
        "vision_tower",
        "visual_encoder",
        "vision_model",
        "visual",
        "clip_model",
    ]

    for attr in vision_tower_attrs:
        if hasattr(model, attr):
            return getattr(model, attr)

    # Check nested attributes (e.g., model.model.vision_tower)
    if hasattr(model, "model"):
        for attr in vision_tower_attrs:
            if hasattr(model.model, attr):
                return getattr(model.model, attr)

    return None
