# Contributing to Resonance

Welcome to **Resonance**! We're excited to have you contribute to the project where machine vision meets human wisdom. This guide will help you set up your development environment and understand how to contribute effectively.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Running Demos](#running-demos)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Architecture Overview](#architecture-overview)

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **CUDA-capable GPU** (recommended for training)
- **Git** for version control
- **16GB+ RAM** (32GB+ recommended for large models)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/1998frankchen/resonance.git
cd resonance

# Set up development environment (see detailed instructions below)
make dev-setup  # or follow manual setup instructions
```

## 🛠️ Development Environment Setup

### Option 1: Automated Setup (Recommended)

We provide a Makefile for easy setup:

```bash
make dev-setup    # Install dependencies and set up pre-commit hooks
make test        # Run all tests
make demo        # Run demo tests
```

### Option 2: Manual Setup

#### Step 1: Create Virtual Environment

We recommend using `uv` for fast package management:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env  # or restart your shell

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
```

#### Step 2: Install Dependencies

```bash
# Install the package in editable mode
uv pip install -e .

# Install additional development dependencies
uv pip install pytest black isort pre-commit mypy

# Install pre-commit hooks
pre-commit install
```

#### Step 3: Verify Installation

```bash
# Test that imports work
python -c "import resonance; print('✅ Resonance imported successfully!')"

# Run demo tests
python demos/test_data_loading.py
```

### GPU Setup (Optional but Recommended)

For training on GPU, install additional CUDA dependencies:

```bash
# For CUDA 11.8 (adjust version as needed)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install FlashAttention for memory efficiency
uv pip install flash-attn --no-build-isolation
```

## 📁 Project Structure

Understanding Resonance's structure:

```
resonance/
├── src/resonance/              # Main package
│   ├── base/                   # Base classes and interfaces
│   │   ├── model.py           # VLRewardModel, VLModelWithValueHead
│   │   ├── processor.py       # VLProcessor base class
│   │   ├── trainer.py         # Base trainers
│   │   └── collator.py        # Data collators
│   ├── models/                # Supported model implementations
│   │   ├── QwenVL/            # Qwen-VL model support
│   │   ├── Llava/             # LLaVA model support
│   │   ├── LlavaNext/         # LLaVA-Next support
│   │   └── InstructBlip/      # InstructBLIP support
│   ├── eval/                  # Evaluation benchmarks
│   │   ├── mme/               # MME benchmark
│   │   ├── mmbench/           # MMBench evaluation
│   │   ├── mmvet/             # MMVet benchmark
│   │   └── vqa/               # VQA evaluation
│   ├── utils/                 # Utility functions
│   │   ├── common.py          # Common utilities
│   │   ├── data.py            # Data processing
│   │   └── auto_load.py       # Model auto-loading
│   ├── sft.py                 # Supervised fine-tuning
│   ├── dpo.py                 # Direct Preference Optimization
│   ├── ppo.py                 # Proximal Policy Optimization
│   └── reward_modeling.py     # Reward model training
├── demos/                     # Demo examples and tutorials
│   ├── data/                  # Sample datasets
│   ├── images/                # Demo images
│   ├── scripts/               # Demo training scripts
│   └── test_data_loading.py   # Demo validation tests
├── docs/                      # Documentation
├── scripts/                   # Training scripts
├── accelerate_config/         # Multi-GPU configurations
└── assets/                    # Branding and visual assets
```

## 🎯 Running Demos

The demos are real, working examples that demonstrate Resonance capabilities:

### Demo 1: Supervised Fine-Tuning (SFT)

```bash
# View the demo script (shows training command)
bash demos/scripts/demo_sft.sh

# To actually run training (requires a pretrained model):
# 1. Download a model like Qwen-VL-Chat to ckpts/
# 2. Replace <path-to-pretrained-model> in the script
# 3. Run the accelerate command shown
```

### Demo 2: Direct Preference Optimization (DPO)

```bash
# View DPO demo
bash demos/scripts/demo_dpo.sh

# This shows how to train with preference pairs (chosen vs rejected responses)
```

### Demo 3: Evaluation

```bash
# View evaluation demo
bash demos/scripts/demo_eval.sh

# This demonstrates evaluation on various benchmarks
```

### Validating Demos

```bash
# Run comprehensive demo tests
python demos/test_data_loading.py

# This validates:
# - Dataset format correctness
# - Image file integrity  
# - Script argument validation
```

## 🔄 Development Workflow

### 1. Setting Up for Development

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/yourusername/resonance.git
cd resonance

# Add upstream remote
git remote add upstream https://github.com/1998frankchen/resonance.git

# Set up development environment
make dev-setup
```

### 2. Making Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Test your changes
make test
python demos/test_data_loading.py

# Commit with descriptive messages
git add .
git commit -m "Add support for new model architecture

- Implement LlamaVision processor
- Add corresponding trainer and collator
- Update model auto-loading logic"
```

### 3. Code Quality

Resonance uses several tools to maintain code quality:

```bash
# Format code
make format  # or: black . && isort .

# Type checking
make lint    # or: mypy src/

# Run pre-commit hooks manually
pre-commit run --all-files
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/integration/   # Integration tests
python demos/test_data_loading.py     # Demo tests
```

### Writing Tests

When adding new features:

1. **Unit tests** for individual functions/classes
2. **Integration tests** for end-to-end workflows
3. **Demo tests** for user-facing examples

Example test structure:
```python
# tests/test_processor.py
import pytest
from resonance.base.processor import VLProcessor

def test_processor_initialization():
    # Test processor can be initialized
    pass

def test_data_processing():
    # Test data processing pipeline
    pass
```

## 🎨 Code Style

Resonance follows these style guidelines:

### Python Code Style

- **PEP 8** compliance (enforced by `black`)
- **Type hints** for all public functions
- **Docstrings** in Google format for all public methods
- **Import organization** with `isort`

Example:
```python
def process_multimodal_data(
    texts: List[str], 
    images: List[Path],
    max_length: int = 512
) -> BatchEncoding:
    """Process multimodal input data for training.
    
    Args:
        texts: List of text prompts
        images: List of image file paths
        max_length: Maximum sequence length
        
    Returns:
        BatchEncoding: Processed batch ready for model input
        
    Raises:
        ValueError: If texts and images have different lengths
    """
    if len(texts) != len(images):
        raise ValueError("Texts and images must have same length")
    
    # Implementation...
    return batch_encoding
```

### Documentation Style

- **Comprehensive docstrings** for all public APIs
- **Type hints** in function signatures
- **Example usage** in docstrings when helpful
- **Clear parameter descriptions**

## 📤 Submitting Changes

### Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create pull request**:
   - Use descriptive title and description
   - Reference any related issues
   - Include tests for new features
   - Ensure all CI checks pass

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature  
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Added/updated unit tests
   - [ ] Tested on demo datasets
   - [ ] Manual testing performed
   
   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   ```

### Review Process

- **Automated checks** must pass (formatting, tests, type checking)
- **Code review** by maintainers
- **Demo validation** to ensure changes don't break examples
- **Performance testing** for training-related changes

## 🏗️ Architecture Overview

### Core Components

1. **Base Classes** (`src/resonance/base/`):
   - `VLProcessor`: Unified interface for multimodal processing
   - `VLRewardModel`: Reward modeling for RLHF
   - `VLTrainer`: Base trainer implementations

2. **Model Support** (`src/resonance/models/`):
   - Each model has its own subdirectory
   - Implements model-specific processing logic
   - Registers with auto-loading system

3. **Training Methods** (root of `src/resonance/`):
   - `sft.py`: Supervised fine-tuning
   - `dpo.py`: Direct Preference Optimization
   - `ppo.py`: Proximal Policy Optimization
   - `reward_modeling.py`: Reward model training

### Adding New Models

To add support for a new vision-language model:

1. **Create model directory**: `src/resonance/models/YourModel/`
2. **Implement processor**: Inherit from `VLProcessor`
3. **Add trainer support**: Implement required trainer methods
4. **Register model**: Update `auto_load.py` mapping
5. **Add tests**: Unit and integration tests
6. **Create demo**: Add demo script and dataset

Example implementation structure:
```python
# src/resonance/models/YourModel/__init__.py
from .processor import YourModelProcessor
from .trainer import YourModelTrainer

core_mapper = {
    "processor": YourModelProcessor,
    "sft_trainer": YourModelTrainer,
    "dpo_trainer": YourModelDPOTrainer,
    # ... other components
}
```

### Data Processing Pipeline

1. **Raw Data** → **Processor** → **Tokenized Data** → **Collator** → **Model Input**
2. Each model implements its own processor for format-specific handling
3. Collators batch and pad data for efficient training
4. Trainers orchestrate the complete training loop

## 🤝 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: Check `docs/` directory for detailed guides
- **Examples**: See `demos/` for working examples

## 📚 Additional Resources

- [Training Arguments Guide](docs/TrainingArguments.md)
- [Evaluation Guide](docs/EvaluationGuide.md)
- [Custom Model Guide](docs/CustomizedModel.md)
- [Demo Documentation](demos/README.md)

---

**Happy coding with Resonance!** 🎵

*Where Machine Vision Meets Human Wisdom*