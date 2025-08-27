"""Resonance Model Utilities and Factories.

This module provides factory functions and utility classes for creating and
managing vision-language models in the Resonance framework. It serves as a
centralized hub for model instantiation, configuration, and management.

🎯 **Core Functionality:**
- Model factory functions for all supported architectures
- Trainer factory functions for different algorithms
- Data collator factory functions for various training types
- Processor factory functions for multi-modal data handling
- Configuration management for model-specific settings

🏭 **Factory Functions:**
- Model creation for LLaVA, Qwen-VL, InstructBLIP, InternLM-XComposer2
- Trainer creation for SFT, DPO, PPO, and Reward Modeling
- Collator creation for different data formatting needs
- Processor creation for vision-language preprocessing

🔧 **Integration:**
This module integrates with the base classes and provides a unified interface
for model creation across the Resonance ecosystem, ensuring consistency and
reducing code duplication.

Author: Frank Chen (Resonance Team)  
Repo: https://github.com/1998frankchen/resonance
"""

from dataclasses import dataclass
from ..base import (
    VLDPODataCollatorWithPadding,
    VLDPOTrainer,
    VLModelWithValueHead,
    VLPPODataCollator,
    VLPPOTrainer,
    VLProcessor,
    VLRewardModel,
    VLRMDataCollatorWithPadding,
    VLRMTrainer,
    VLSFTDataCollatorWithPadding,
    VLSFTTrainer,
)
from transformers import PreTrainedModel


@dataclass
class ModelCoreMapper:
    model: PreTrainedModel
    processor: VLProcessor
    dpo_collator: VLDPODataCollatorWithPadding
    dpo_trainer: VLDPOTrainer
    reward_model: VLRewardModel
    value_model: VLModelWithValueHead
    reward_collator: VLRMDataCollatorWithPadding
    reward_trainer: VLRMTrainer
    sft_collator: VLSFTDataCollatorWithPadding
    sft_trainer: VLSFTTrainer
    ppo_collator: VLPPODataCollator
    ppo_trainer: VLPPOTrainer
