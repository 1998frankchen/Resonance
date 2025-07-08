"""Resonance Command Line Interface.

This module provides a unified command-line interface for all Resonance
training and evaluation operations. It consolidates the previously scattered
training scripts into a cohesive, user-friendly CLI tool.

The CLI follows modern best practices:
- Intuitive subcommand structure (train, eval, export, etc.)
- Rich help documentation and examples
- Configuration file support
- Progress tracking and logging
- Error handling and validation

Key Commands:
    train: Train models using SFT, DPO, PPO, or reward modeling
    eval: Evaluate models on vision-language benchmarks
    export: Export and merge trained models
    config: Generate and validate configuration files
    data: Data processing and validation utilities

Command Structure:
    ```
    resonance <command> <subcommand> [options]

    resonance train sft --config config.yaml
    resonance train dpo --model path/to/sft/model --data preference_data.json
    resonance eval mme --model path/to/model --data eval_data.json
    resonance export merge --model path/to/lora/model --output merged_model
    ```

Features:
    - Type-safe configuration handling
    - Automatic resource detection and optimization
    - Distributed training setup
    - Comprehensive logging and monitoring
    - Progress bars and status updates

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from .main import main, ResonanceCLI
from .train import TrainCommand
from .eval import EvalCommand
from .export import ExportCommand
from .config import ConfigCommand
from .data import DataCommand

__all__ = [
    "main",
    "ResonanceCLI",
    "TrainCommand",
    "EvalCommand",
    "ExportCommand",
    "ConfigCommand",
    "DataCommand",
]
