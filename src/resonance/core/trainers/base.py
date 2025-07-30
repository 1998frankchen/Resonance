"""Base Trainer Class for Resonance.

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Base class for all trainers."""

    @abstractmethod
    def train(self):
        """Execute training."""
        pass
