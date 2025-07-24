"""Resonance Algorithm-Specific Configuration Classes.

This module defines configuration classes for different training algorithms
supported by Resonance: SFT, DPO, PPO, and Reward Modeling. Each configuration
class combines base configurations with algorithm-specific parameters.

The algorithm configurations follow this pattern:
- Inherit from base configurations for common settings
- Add algorithm-specific hyperparameters
- Provide sensible defaults based on research best practices
- Include validation for algorithm-specific constraints

Classes:
    SFTConfig: Supervised Fine-Tuning configuration
    DPOConfig: Direct Preference Optimization configuration
    PPOConfig: Proximal Policy Optimization configuration
    RewardModelConfig: Reward model training configuration

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .base import BaseConfig, DataConfig, EvalConfig, ModelConfig, TrainingConfig


@dataclass(frozen=True)
class SFTConfig(BaseConfig):
    """Supervised Fine-Tuning (SFT) configuration.

    SFT is the first stage of RLHF training where the model learns to follow
    instructions and have conversations about visual content. This configuration
    combines all necessary settings for effective SFT training.

    The configuration includes:
    - Model loading and architecture settings
    - Training loop and optimization parameters
    - Dataset and preprocessing settings
    - Evaluation configuration
    - SFT-specific hyperparameters

    Attributes:
        model: Model configuration
        training: Training configuration
        data: Data configuration
        evaluation: Evaluation configuration
        output_dir: Directory to save model and results
        resume_from_checkpoint: Resume from checkpoint path
        ignore_bias_buffers: Ignore bias buffers in DDP
    """

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # SFT-specific settings
    output_dir: str = "./output/sft"
    resume_from_checkpoint: Optional[str] = None
    ignore_bias_buffers: bool = False

    def validate(self) -> None:
        """Validate SFT configuration."""
        # Validate all sub-configurations
        self.model.validate()
        self.training.validate()
        self.data.validate()
        self.evaluation.validate()

        # SFT-specific validation
        if not self.output_dir:
            raise ValueError("output_dir is required")

        # Check that model supports instruction following
        if self.model.freeze_language_model and not self.model.use_lora:
            raise ValueError(
                "Cannot freeze language model without LoRA - "
                "model won't learn anything"
            )


@dataclass(frozen=True)
class DPOConfig(BaseConfig):
    """Direct Preference Optimization (DPO) configuration.

    DPO trains models to align with human preferences without requiring
    a separate reward model. This configuration includes DPO-specific
    hyperparameters and settings for preference learning.

    Key DPO parameters:
    - beta: Regularization strength (higher = stay closer to reference)
    - label_smoothing: Smoothing for preference labels
    - loss_type: Type of DPO loss function
    - reference_model_path: Path to reference model (usually SFT model)

    Attributes:
        model: Model configuration
        training: Training configuration
        data: Data configuration
        evaluation: Evaluation configuration
        beta: KL regularization strength (typically 0.1-0.5)
        label_smoothing: Label smoothing factor
        loss_type: DPO loss variant ('sigmoid', 'hinge', 'ipo')
        reference_model_path: Reference model for KL regularization
        score_margin: Minimum score gap for preference pairs
        output_dir: Directory to save model and results
        resume_from_checkpoint: Resume from checkpoint path
        ignore_bias_buffers: Ignore bias buffers in DDP
    """

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # DPO-specific hyperparameters
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_model_path: Optional[str] = None
    score_margin: float = -1.0  # -1 means use largest gap

    # General settings
    output_dir: str = "./output/dpo"
    resume_from_checkpoint: Optional[str] = None
    ignore_bias_buffers: bool = False

    def validate(self) -> None:
        """Validate DPO configuration."""
        # Validate all sub-configurations
        self.model.validate()
        self.training.validate()
        self.data.validate()
        self.evaluation.validate()

        # DPO-specific validation
        if self.beta <= 0:
            raise ValueError("beta must be positive")

        if not 0 <= self.label_smoothing <= 1:
            raise ValueError("label_smoothing must be between 0 and 1")

        if self.loss_type not in ["sigmoid", "hinge", "ipo", "kto_pair"]:
            raise ValueError(
                f"loss_type must be one of: sigmoid, hinge, ipo, kto_pair. "
                f"Got: {self.loss_type}"
            )

        # Check data configuration for preference learning
        if (
            "dpo" not in self.data.dataset_name.lower()
            and "preference" not in self.data.dataset_name.lower()
        ):
            raise ValueError(
                "DPO requires preference dataset. "
                f"Dataset name '{self.data.dataset_name}' doesn't indicate preference data"
            )


@dataclass(frozen=True)
class PPOConfig(BaseConfig):
    """Proximal Policy Optimization (PPO) configuration.

    PPO is an on-policy reinforcement learning algorithm that uses a reward
    model to optimize the policy. This configuration includes PPO-specific
    hyperparameters and settings for RL training.

    Key PPO parameters:
    - learning_rate: Learning rate for policy updates
    - batch_size: Number of experiences per update
    - mini_batch_size: Size of mini-batches for gradient updates
    - ppo_epochs: Number of optimization epochs per batch
    - clip_range: PPO clip range for policy updates
    - vf_coef: Value function loss coefficient
    - ent_coef: Entropy loss coefficient
    - kl_penalty: KL divergence penalty type

    Attributes:
        model: Model configuration
        training: Training configuration
        data: Data configuration
        evaluation: Evaluation configuration
        reward_model_path: Path to trained reward model
        learning_rate: Policy learning rate
        batch_size: PPO batch size
        mini_batch_size: Mini-batch size for updates
        ppo_epochs: Optimization epochs per batch
        clip_range: PPO clipping range
        clip_range_vf: Value function clipping range
        vf_coef: Value function loss coefficient
        ent_coef: Entropy bonus coefficient
        kl_penalty: KL penalty type ('kl', 'abs', 'mse', 'full')
        target_kl: Target KL divergence
        gamma: Reward discount factor
        lam: GAE lambda parameter
        whiten_rewards: Normalize rewards
        output_dir: Directory to save model and results
        resume_from_checkpoint: Resume from checkpoint path
    """

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # PPO-specific hyperparameters
    learning_rate: float = 1e-5
    reward_model_path: str = ""  # Will be validated in validate() method
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    vf_coef: float = 0.1
    ent_coef: float = 0.01
    kl_penalty: str = "kl"
    target_kl: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    whiten_rewards: bool = True

    # General settings
    output_dir: str = "./output/ppo"
    resume_from_checkpoint: Optional[str] = None

    def validate(self) -> None:
        """Validate PPO configuration."""
        # Validate all sub-configurations
        self.model.validate()
        self.training.validate()
        self.data.validate()
        self.evaluation.validate()

        # PPO-specific validation
        if not self.reward_model_path or self.reward_model_path == "":
            raise ValueError("reward_model_path is required for PPO")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be positive")

        if self.batch_size % self.mini_batch_size != 0:
            raise ValueError("batch_size must be divisible by mini_batch_size")

        if self.ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive")

        if not 0 < self.clip_range <= 1:
            raise ValueError("clip_range must be between 0 and 1")

        if self.kl_penalty not in ["kl", "abs", "mse", "full"]:
            raise ValueError(f"kl_penalty must be one of: kl, abs, mse, full")

        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")

        if not 0 <= self.lam <= 1:
            raise ValueError("lam must be between 0 and 1")


@dataclass(frozen=True)
class RewardModelConfig(BaseConfig):
    """Reward Model training configuration.

    Reward models learn to predict human preferences and provide reward
    signals for reinforcement learning. This configuration includes settings
    specific to reward model training on preference data.

    Key aspects:
    - Trained on pairwise preference data
    - Uses ranking loss to predict preferences
    - Can filter data based on score margins
    - Outputs scalar rewards for each input

    Attributes:
        model: Model configuration
        training: Training configuration
        data: Data configuration
        evaluation: Evaluation configuration
        score_margin: Minimum score gap for preference pairs
        max_length: Maximum input sequence length
        output_dir: Directory to save model and results
        resume_from_checkpoint: Resume from checkpoint path
        ignore_bias_buffers: Ignore bias buffers in DDP
    """

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # Reward model specific settings
    score_margin: float = 0.5  # Minimum score difference
    max_length: int = 512

    # General settings
    output_dir: str = "./output/reward_model"
    resume_from_checkpoint: Optional[str] = None
    ignore_bias_buffers: bool = False

    def validate(self) -> None:
        """Validate reward model configuration."""
        # Validate all sub-configurations
        self.model.validate()
        self.training.validate()
        self.data.validate()
        self.evaluation.validate()

        # Reward model specific validation
        if self.score_margin < 0 and self.score_margin != -1:
            raise ValueError("score_margin must be non-negative or -1 for auto")

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        # Check data configuration for preference learning
        if (
            "reward" not in self.data.dataset_name.lower()
            and "preference" not in self.data.dataset_name.lower()
        ):
            raise ValueError(
                "Reward model requires preference dataset. "
                f"Dataset name '{self.data.dataset_name}' doesn't indicate preference data"
            )
