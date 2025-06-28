"""Resonance: Vision-Language RLHF Framework.

Resonance is a comprehensive framework for training vision-language models using
Reinforcement Learning from Human Feedback (RLHF). It provides a unified, extensible
architecture for implementing state-of-the-art multi-modal alignment techniques.

🎵 **What is Resonance?**
Resonance bridges the gap between machine vision and human wisdom by providing
tools to align vision-language models with human preferences. The framework
supports the complete RLHF pipeline from supervised fine-tuning to preference
optimization and reinforcement learning.

🏗️ **Architecture Overview:**
```
resonance/
├── core/              # Framework core components
│   ├── config/        # Type-safe configuration system
│   ├── trainers/      # Training algorithm implementations
│   ├── models/        # Base model abstractions
│   └── data/          # Data processing pipeline
├── contrib/           # Model-specific implementations
│   ├── llava/         # LLaVA model support
│   ├── qwen_vl/       # Qwen-VL model support
│   └── ...            # Other model implementations
├── evaluation/        # Unified evaluation framework
├── cli/               # Command-line interface
├── utils/             # Shared utilities
└── [legacy scripts]   # Backward-compatible training scripts
```

🚀 **Supported Algorithms:**
- **SFT (Supervised Fine-Tuning):** Foundation instruction-following training
- **DPO (Direct Preference Optimization):** Preference learning without reward models
- **PPO (Proximal Policy Optimization):** Reinforcement learning with reward models
- **Reward Modeling:** Training preference prediction models

🤖 **Supported Models:**
- **LLaVA:** Large Language and Vision Assistant variants
- **Qwen-VL:** Qwen Vision-Language models
- **InstructBLIP:** Instruction-aware vision-language models
- **InternLM-XComposer2:** Versatile vision-language composer
- **Extensible:** Easy to add new model architectures

📊 **Key Features:**
- **Type-Safe Configuration:** Comprehensive configuration system with validation
- **Unified Training Interface:** Consistent API across all algorithms
- **Multi-Modal Data Processing:** Robust handling of vision-language datasets
- **Performance Optimizations:** Memory-efficient training with DeepSpeed, LoRA
- **Comprehensive Evaluation:** Unified evaluation across multiple benchmarks
- **Production Ready:** Robust error handling, logging, and monitoring

💡 **Quick Start:**

Using the new configuration-based API:
```python
from resonance.core.config import SFTConfig, ModelConfig, TrainingConfig, DataConfig
from resonance.core.trainers import create_trainer

# Configure training
config = SFTConfig(
    model=ModelConfig(
        model_name_or_path="Qwen/Qwen-VL-Chat",
        use_lora=True,
        lora_r=16
    ),
    training=TrainingConfig(
        num_train_epochs=3,
        learning_rate=2e-5
    ),
    data=DataConfig(
        dataset_name="vlquery_json",
        data_path="data.json",
        image_root="images/"
    ),
    output_dir="./output/sft"
)

# Create and run trainer
trainer = create_trainer(config)
trainer.train()
```

Using the command-line interface:
```bash
# New unified CLI
resonance train sft --model Qwen/Qwen-VL-Chat --data data.json --output ./output

# Legacy script compatibility
python src/resonance/sft.py --model_name_or_path Qwen/Qwen-VL-Chat --data_path data.json
```

🔧 **Development:**
Resonance is designed for extensibility and contribution:
- Clean separation between core framework and model implementations
- Plugin architecture for adding new models and algorithms
- Comprehensive testing and validation
- Rich development tools and documentation

📚 **Documentation:**
- Configuration reference in `core/config/`
- Training guides in `core/trainers/`
- Model integration in `contrib/`
- Evaluation documentation in `evaluation/`

🤝 **Community:**
Resonance is an open-source project welcoming contributions:
- Issue tracking and feature requests
- Model implementation contributions
- Algorithm and technique improvements
- Documentation and example contributions

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
License: Apache 2.0
"""

# Core framework exports
from .core.config import (
    SFTConfig,
    DPOConfig,
    PPOConfig,
    RewardModelConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    EvalConfig,
)
from .core.trainers import create_trainer
from .core.data import DataProcessor

# Evaluation framework
from .evaluation import EvaluationManager

# Utilities
from .utils import (
    load_model_safe,
    save_model_safe,
    get_vision_tower,
    Colors,
    split_into_words,
)

# Version information
__version__ = "2.0.0"
__author__ = "Frank Chen (Resonance Team)"

# Main exports for easy access
__all__ = [
    # Configuration classes
    "SFTConfig",
    "DPOConfig",
    "PPOConfig",
    "RewardModelConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "EvalConfig",
    # Core functionality
    "create_trainer",
    "DataProcessor",
    "EvaluationManager",
    # Utilities
    "load_model_safe",
    "save_model_safe",
    "get_vision_tower",
    "Colors",
    "split_into_words",
    "preprocess_conversations",
    # Version
    "__version__",
    "__author__",
]
