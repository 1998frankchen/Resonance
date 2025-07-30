"""Resonance Core Data Processor.

This module provides the main DataProcessor class that orchestrates all data
processing operations for vision-language RLHF training. It serves as the
central interface for loading, preprocessing, and preparing datasets.

The DataProcessor handles:
- Dataset loading from various sources and formats
- Image preprocessing and validation
- Text tokenization and formatting
- Preference pair generation and filtering
- Data validation and quality checks
- Caching and performance optimization

The processor is designed to be algorithm-agnostic, supporting all RLHF
training methods (SFT, DPO, PPO, RM) through a unified interface.

Classes:
    DataProcessor: Main data processing orchestrator
    ProcessingStats: Statistics about data processing
    ProcessingError: Custom exception for data processing errors

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Set up logger first
logger = logging.getLogger(__name__)

try:
    from datasets import Dataset
except ImportError:
    Dataset = None
    logger.warning("datasets not available - some features may not work")

try:
    from PIL import Image
except ImportError:
    Image = None
    logger.warning("Pillow not available - image processing may not work")

from ..config.base import DataConfig


class ProcessingError(Exception):
    """Custom exception for data processing errors."""

    pass


@dataclass
class ProcessingStats:
    """Statistics about data processing operations.

    Attributes:
        total_samples: Total number of samples processed
        valid_samples: Number of samples that passed validation
        invalid_samples: Number of samples that failed validation
        image_errors: Number of samples with image loading errors
        text_errors: Number of samples with text processing errors
        processing_time: Total processing time in seconds
    """

    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    image_errors: int = 0
    text_errors: int = 0
    processing_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate of data processing."""
        if self.total_samples == 0:
            return 0.0
        return self.valid_samples / self.total_samples

    def __str__(self) -> str:
        return (
            f"ProcessingStats(total={self.total_samples}, "
            f"valid={self.valid_samples}, "
            f"success_rate={self.success_rate:.2%}, "
            f"time={self.processing_time:.2f}s)"
        )


class DataProcessor:
    """Main data processor for vision-language RLHF training.

    The DataProcessor provides a unified interface for handling all data
    processing operations across different training algorithms and dataset
    formats. It manages the complete pipeline from raw data to training-ready
    batches.

    Key Features:
        - Supports multiple dataset formats (JSON, HuggingFace, custom)
        - Handles vision-language data preprocessing
        - Generates preference pairs for RLHF training
        - Provides data validation and quality checks
        - Optimized for performance with caching and parallel processing

    Args:
        config: Data configuration specifying dataset and processing settings
        cache_dir: Directory for caching processed data
        num_workers: Number of workers for parallel processing
        verbose: Enable verbose logging

    Example:
        ```python
        config = DataConfig(
            dataset_name="vlquery_json",
            data_path="data.json",
            image_root="images/"
        )

        processor = DataProcessor(config)
        dataset = processor.load_dataset()
        ```
    """

    def __init__(
        self,
        config: DataConfig,
        cache_dir: Optional[str] = None,
        num_workers: int = 4,
        verbose: bool = False,
    ):
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.num_workers = num_workers
        self.verbose = verbose

        # Initialize processing statistics
        self.stats = ProcessingStats()

        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.INFO)

        # Validate configuration
        self.config.validate()

        logger.info(f"Initialized DataProcessor for dataset: {config.dataset_name}")

    def load_dataset(self) -> Dataset:
        """Load and process the dataset according to configuration.

        This is the main entry point for data processing. It handles
        loading the raw data, applying preprocessing, and returning
        a training-ready dataset.

        Returns:
            Processed dataset ready for training

        Raises:
            ProcessingError: If data processing fails
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")

        try:
            # Load raw data based on dataset type
            raw_data = self._load_raw_data()

            # Apply preprocessing pipeline
            processed_data = self._preprocess_data(raw_data)

            # Validate processed data
            validated_data = self._validate_data(processed_data)

            # Convert to HuggingFace Dataset
            dataset = Dataset.from_list(validated_data)

            logger.info(f"Successfully loaded {len(dataset)} samples")
            logger.info(f"Processing stats: {self.stats}")

            return dataset

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise ProcessingError(f"Dataset loading failed: {e}") from e

    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw data from the specified source.

        Returns:
            List of raw data samples
        """
        data_path = Path(self.config.data_path)

        if not data_path.exists():
            raise ProcessingError(f"Data path does not exist: {data_path}")

        logger.info(f"Loading data from: {data_path}")

        # Handle different file formats
        if data_path.suffix.lower() == ".json":
            return self._load_json_data(data_path)
        elif data_path.suffix.lower() == ".jsonl":
            return self._load_jsonl_data(data_path)
        else:
            raise ProcessingError(f"Unsupported file format: {data_path.suffix}")

    def _load_json_data(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys for list data
            for key in ["data", "samples", "conversations", "examples"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ProcessingError(f"Could not find data list in JSON file")
        else:
            raise ProcessingError(f"Unexpected JSON structure: {type(data)}")

    def _load_jsonl_data(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")

        return data

    def _preprocess_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply preprocessing pipeline to raw data.

        Args:
            raw_data: List of raw data samples

        Returns:
            List of preprocessed samples
        """
        logger.info("Preprocessing data...")

        processed_data = []
        self.stats.total_samples = len(raw_data)

        # Apply data ratio if specified
        if self.config.data_ratio < 1.0:
            num_samples = int(len(raw_data) * self.config.data_ratio)
            raw_data = raw_data[:num_samples]
            logger.info(
                f"Using {num_samples} samples (ratio: {self.config.data_ratio})"
            )

        for idx, sample in enumerate(raw_data):
            try:
                # Preprocess individual sample
                processed_sample = self._preprocess_sample(sample)
                if processed_sample is not None:
                    processed_data.append(processed_sample)
                    self.stats.valid_samples += 1
                else:
                    self.stats.invalid_samples += 1

            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                self.stats.invalid_samples += 1

        logger.info(f"Preprocessing complete: {len(processed_data)} valid samples")
        return processed_data

    def _preprocess_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single data sample.

        Args:
            sample: Raw data sample

        Returns:
            Preprocessed sample or None if invalid
        """
        try:
            # Extract and validate required fields
            processed = {}

            # Handle image path
            if "image" in sample or "image_path" in sample:
                image_path = sample.get("image") or sample.get("image_path")
                if image_path and self.config.image_root:
                    full_image_path = Path(self.config.image_root) / image_path
                    if full_image_path.exists():
                        processed["image_path"] = str(full_image_path)
                    else:
                        logger.warning(f"Image not found: {full_image_path}")
                        return None

            # Handle text fields
            for field in ["prompt", "question", "conversations", "chosen", "rejected"]:
                if field in sample:
                    processed[field] = sample[field]

            # Preserve other metadata
            for field in ["id", "metadata", "source"]:
                if field in sample:
                    processed[field] = sample[field]

            return processed

        except Exception as e:
            logger.warning(f"Error preprocessing sample: {e}")
            return None

    def _validate_data(
        self, processed_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate processed data samples.

        Args:
            processed_data: List of processed samples

        Returns:
            List of validated samples
        """
        logger.info("Validating processed data...")

        validated_data = []

        for sample in processed_data:
            if self._validate_sample(sample):
                validated_data.append(sample)

        logger.info(f"Validation complete: {len(validated_data)} valid samples")
        return validated_data

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a single processed sample.

        Args:
            sample: Processed sample

        Returns:
            True if valid, False otherwise
        """
        # Check for required fields based on dataset type
        if self.config.dataset_name == "vlquery_json":
            required_fields = ["prompt", "image_path"]
        elif "dpo" in self.config.dataset_name.lower():
            required_fields = ["prompt", "chosen", "rejected"]
        else:
            required_fields = ["prompt"]

        for field in required_fields:
            if field not in sample or not sample[field]:
                return False

        # Validate image if present
        if "image_path" in sample:
            try:
                if not Path(sample["image_path"]).exists():
                    return False
                # Try to open image to verify it's valid
                with Image.open(sample["image_path"]) as img:
                    img.verify()
            except Exception:
                return False

        return True

    def get_stats(self) -> ProcessingStats:
        """Get processing statistics.

        Returns:
            Current processing statistics
        """
        return self.stats
