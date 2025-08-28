"""Resonance Unified Utilities Module.

This module consolidates all utility functions used throughout Resonance into
a well-organized, cohesive package. It provides common functionality for
model operations, data processing, text analysis, and system utilities.

The utilities are organized into logical submodules:
- model_utils: Model loading, saving, and manipulation utilities
- data_utils: Data processing and transformation utilities
- text_utils: Text processing, tokenization, and analysis
- system_utils: System operations, logging, and performance utilities
- visual_utils: Image processing and visualization utilities

Key Features:
- Centralized utility functions with consistent APIs
- Comprehensive error handling and validation
- Performance optimizations and caching
- Cross-platform compatibility
- Extensive documentation and examples

Example Usage:
    ```python
    from resonance.utils import model_utils, data_utils, text_utils

    # Load a model with utilities
    model = model_utils.load_model_safe("path/to/model")

    # Process data
    processed_data = data_utils.preprocess_conversations(raw_data)

    # Analyze text differences
    diff_result = text_utils.compare_texts(text1, text2)
    ```

Migration Notes:
    The utilities in this module replace and consolidate the previous scattered
    utility files (common.py, auto_load.py, diff_lib.py, data.py) with improved
    organization, documentation, and functionality.

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

# Import key utility functions for easy access
from .model_utils import (
    load_model_safe,
    save_model_safe,
    get_vision_tower,
    maybe_zero_3,
    safe_save_model_for_hf_trainer,
    safe_save_model_for_ppo_trainer,
)

from .data import (
    make_vlfeedback_paired_dataset,
    build_dataset_from_vlquery_json,
    make_rlhfv_paired_dataset,
    build_plain_dpo_dataset,
)

from .diff_lib import (
    split_into_words,
    split_into_clauses,
    show_mark_compare_words,
    show_mark_compare_substring,
    get_diff_ids,
    color_print_diff_pair,
    Colors,
)

from .common import maybe_zero_3

# Re-export commonly used classes and constants
from .data import DATASET_MAP

__all__ = [
    # Model utilities
    "load_model_safe",
    "save_model_safe",
    "get_vision_tower",
    "maybe_zero_3",
    "safe_save_model_for_hf_trainer",
    "safe_save_model_for_ppo_trainer",
    # Data utilities
    "make_vlfeedback_paired_dataset",
    "build_dataset_from_vlquery_json",
    "make_rlhfv_paired_dataset",
    "build_plain_dpo_dataset",
    "DATASET_MAP",
    # Text utilities
    "split_into_words",
    "split_into_clauses",
    "show_mark_compare_words",
    "show_mark_compare_substring",
    "get_diff_ids",
    "color_print_diff_pair",
    "Colors",
    # Common utilities
    "maybe_zero_3",
]
