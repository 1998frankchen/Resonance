"""Resonance Trainer Factory.

This module provides factory functions for creating trainers from configurations.

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from typing import Dict, Type

from ..config.algorithms import SFTConfig, DPOConfig, PPOConfig, RewardModelConfig


class TrainerRegistry:
    """Registry for trainer implementations."""

    _trainers: Dict[Type, str] = {}

    @classmethod
    def register(cls, config_type: Type, trainer_name: str):
        """Register a trainer for a configuration type."""
        cls._trainers[config_type] = trainer_name

    @classmethod
    def get_trainer(cls, config_type: Type) -> str:
        """Get trainer name for configuration type."""
        return cls._trainers.get(config_type, "BaseTrainer")


def create_trainer(config):
    """Create trainer from configuration.

    Args:
        config: Training configuration

    Returns:
        Trainer instance
    """
    # For now, return a placeholder since we don't have full implementation
    # This would be implemented with actual trainer classes

    from .base import BaseTrainer

    class MockTrainer(BaseTrainer):
        def __init__(self, config):
            self.config = config

        def train(self):
            print(f"Training with {type(self.config).__name__}")
            # Actual training implementation would go here

    return MockTrainer(config)
