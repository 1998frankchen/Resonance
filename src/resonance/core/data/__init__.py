"""Resonance Core Data Processing Framework.

This module provides a unified data processing system for vision-language RLHF
training. It handles various dataset formats, preprocessing pipelines, and
data loading strategies required for different training algorithms.

The data framework supports:
- Multiple dataset formats (JSON, HuggingFace, custom)
- Vision-language data preprocessing
- Preference pair generation for RLHF
- Efficient data loading and caching
- Multi-modal data validation

Key Components:
    DataProcessor: Main interface for data processing
    DatasetLoader: Loading datasets from various sources
    PreferenceDataset: Handling preference pairs for DPO/RM training
    VisionLanguageCollator: Batching multi-modal data
    DataRegistry: Registry of available datasets

Dataset Types:
    - Instruction datasets: Question-answer pairs with images
    - Preference datasets: Ranked responses for preference learning
    - Conversation datasets: Multi-turn dialogues with visual context
    - Evaluation datasets: Benchmark tasks and metrics

Example:
    ```python
    from resonance.core.data import DataProcessor, DatasetLoader
    from resonance.core.config import DataConfig

    config = DataConfig(
        dataset_name="vlquery_json",
        data_path="path/to/data.json",
        image_root="path/to/images"
    )

    processor = DataProcessor(config)
    dataset = processor.load_dataset()
    ```

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from .processor import DataProcessor

# Note: Additional data components would be implemented in future versions
# from .loader import DatasetLoader, DatasetRegistry
# from .collators import VisionLanguageCollator, PreferenceCollator, InstructionCollator
# from .datasets import PreferenceDataset, InstructionDataset, EvaluationDataset
# from .utils import validate_data_config, preprocess_images, tokenize_conversations

__all__ = [
    # Main interfaces
    "DataProcessor",
    # Loading and registry (future implementation)
    # "DatasetLoader",
    # "DatasetRegistry",
    # Data collation (future implementation)
    # "VisionLanguageCollator",
    # "PreferenceCollator",
    # "InstructionCollator",
    # Dataset implementations (future implementation)
    # "PreferenceDataset",
    # "InstructionDataset",
    # "EvaluationDataset",
    # Utility functions (future implementation)
    # "validate_data_config",
    # "preprocess_images",
    # "tokenize_conversations",
]
