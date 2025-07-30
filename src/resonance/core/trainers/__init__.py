"""Resonance Core Training Framework.

This module provides a unified, high-level training interface for all RLHF
algorithms in Resonance. It abstracts away implementation details and provides
a consistent API for training vision-language models.

The training framework follows these principles:
- Configuration-driven: All settings specified via typed configurations
- Algorithm-agnostic: Same interface for SFT, DPO, PPO, and RM training
- Extensible: Easy to add new algorithms and model architectures
- Robust: Comprehensive validation and error handling
- Scalable: Built-in support for distributed training

Key Components:
    BaseTrainer: Abstract base class for all trainers
    SFTTrainer: Supervised fine-tuning trainer
    DPOTrainer: Direct preference optimization trainer
    PPOTrainer: Proximal policy optimization trainer
    RewardModelTrainer: Reward model training trainer
    TrainerFactory: Factory for creating trainers from configurations

Example:
    ```python
    from resonance.core.config import SFTConfig, ModelConfig, TrainingConfig, DataConfig
    from resonance.core.trainers import create_trainer

    config = SFTConfig(
        model=ModelConfig(model_name_or_path="Qwen/Qwen-VL-Chat"),
        training=TrainingConfig(num_train_epochs=3),
        data=DataConfig(dataset_name="vlquery_json", data_path="data.json")
    )

    trainer = create_trainer(config)
    trainer.train()
    ```

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from .base import BaseTrainer
from .factory import create_trainer, TrainerRegistry

# Note: Specific trainer implementations would be added in future versions
# from .sft import SFTTrainer
# from .dpo import DPOTrainer
# from .ppo import PPOTrainer
# from .reward_model import RewardModelTrainer

__all__ = [
    # Base trainer
    "BaseTrainer",
    # Algorithm-specific trainers (future implementation)
    # "SFTTrainer",
    # "DPOTrainer",
    # "PPOTrainer",
    # "RewardModelTrainer",
    # Factory and registry
    "create_trainer",
    "TrainerRegistry",
]
