"""Resonance Supervised Fine-Tuning (SFT) Training Script.

This script implements supervised fine-tuning for vision-language models in Resonance.
SFT is the first stage of RLHF training that teaches models to follow instructions 
and engage in conversations about visual content using supervised learning.

🎯 **Purpose:**
SFT trains vision-language models to understand and respond to instructions by learning
from high-quality instruction-following examples. This creates a foundation model that
can then be further improved through preference optimization (DPO) or reinforcement
learning (PPO).

🏗️ **Architecture Integration:**
This script serves as a backward-compatible interface to the new Resonance architecture:
- Converts command-line arguments to typed configurations
- Leverages the unified core.config system for type safety
- Uses the core.trainers framework for consistent training logic
- Integrates with the consolidated utils for shared functionality

📊 **Training Process:**
1. **Data Loading:** Loads instruction datasets with image-text pairs
2. **Model Setup:** Initializes vision-language model with LoRA adaptation
3. **Training Loop:** Supervised learning on instruction-response pairs
4. **Evaluation:** Optional evaluation on held-out data
5. **Model Saving:** Saves trained adapter and merged model

🔧 **Key Features:**
- **Multi-Modal Support:** Handles both vision and language inputs seamlessly
- **Efficient Training:** LoRA/QLoRA for memory-efficient adaptation
- **Flexible Data:** Supports various conversation and instruction formats
- **Robust Training:** Gradient checkpointing, mixed precision, DeepSpeed
- **Model Agnostic:** Works with LLaVA, Qwen-VL, InstructBLIP, and more

📈 **Performance Optimizations:**
- Gradient accumulation for large effective batch sizes
- Mixed precision training (fp16/bf16) for speed and memory
- Gradient checkpointing for memory efficiency
- Vision tower freezing to focus adaptation on language

🎛️ **Configuration Options:**
- Model selection and LoRA parameters
- Dataset paths and preprocessing settings
- Training hyperparameters and optimization
- Evaluation and logging configuration
- Output and checkpointing options

💡 **Migration Path:**
This script maintains full backward compatibility with existing workflows while
internally using the improved Resonance architecture. Users can gradually migrate
to the new configuration-based API for enhanced type safety and functionality.

Usage Examples:
    # Basic SFT training
    python src/resonance/sft.py \\
        --model_name_or_path Qwen/Qwen-VL-Chat \\
        --dataset_name vlquery_json \\
        --data_path conversations.json \\
        --image_root images/ \\
        --output_dir ./output/sft \\
        --use_lora True
    
    # Advanced configuration  
    python src/resonance/sft.py \\
        --model_name_or_path Qwen/Qwen-VL-Chat \\
        --data_path large_dataset.json \\
        --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 8 \\
        --learning_rate 2e-5 \\
        --num_train_epochs 3 \\
        --lora_r 64 \\
        --lora_alpha 128 \\
        --bf16 True \\
        --gradient_checkpointing True

🔗 **Related Components:**
- core/config/algorithms.py: SFTConfig type definitions
- core/trainers/sft.py: Core SFT training implementation  
- core/data/: Unified data processing pipeline
- utils/: Consolidated utility functions

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
License: Apache 2.0
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import HfArgumentParser
from transformers.trainer_callback import TrainerCallback

from resonance.utils.auto_load import (
    MyAutoSFTCollator,
    MyAutoSFTTrainer,
    MyAutoProcessor,
    auto_load_rlmodel,
)
from resonance.utils.common import safe_save_model_for_hf_trainer
from resonance.utils.data import DATASET_MAP


# Enable detailed logging for debugging
# transformers.logging.set_verbosity_info()
@dataclass
class ScriptArguments:
    """Configuration arguments for SFT training script.

    This class contains all the hyperparameters and configuration options
    specific to supervised fine-tuning of vision-language models.
    """

    # Dataset Configuration
    data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to custom dataset JSON file. Required for custom datasets."
        },
    )
    data_ratio: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Fraction of dataset to use for training (0.0-1.0). Useful for quick experiments."
        },
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={
            "help": "Root directory containing images. Will be joined with image paths in dataset."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset identifier. Options: 'vlquery_json' for custom conversation data."
        },
    )

    # Model Configuration
    model_name_or_path: Optional[str] = field(
        default="llava-hf/llava-1.5-7b-hf",
        metadata={
            "help": "HuggingFace model identifier or local path to pre-trained vision-language model."
        },
    )

    # Training Configuration
    max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum sequence length for input text. Longer sequences will be truncated."
        },
    )
    label_pad_token_id: Optional[int] = field(
        default=-100,
        metadata={
            "help": "Token ID used for padding labels. Tokens with this ID are ignored in loss computation."
        },
    )

    # Model-specific Configuration
    freeze_vision_tower: bool = field(
        default=True,
        metadata={
            "help": "Whether to freeze the vision encoder during training. Recommended for stability."
        },
    )
    merge_peft_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to merge LoRA weights into base model after training. Creates full model."
        },
    )

    # Advanced/Debug Configuration
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Fix for DDP issues with LM bias/mask buffers. Set True if encountering distributed training issues."
        },
    )


@dataclass
class LoraArguments:
    """Configuration for Low-Rank Adaptation (LoRA) fine-tuning.

    LoRA enables efficient fine-tuning by updating only a small number of
    parameters while keeping the original model frozen.
    """

    lora_r: int = field(
        default=64,
        metadata={
            "help": "LoRA rank. Higher values = more parameters but potentially better quality."
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={
            "help": "LoRA alpha parameter for scaling. Typically lora_alpha = lora_r or lora_r/2."
        },
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout rate for regularization."}
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Target modules for LoRA. Comma-separated list or 'auto' for automatic detection."
        },
    )
    lora_bias: str = field(
        default="none",
        metadata={
            "help": "LoRA bias configuration. Options: 'none', 'all', 'lora_only'."
        },
    )
    q_lora: bool = field(
        default=False,
        metadata={
            "help": "Enable QLoRA (quantized LoRA) for even more efficient training."
        },
    )
    bits: int = field(
        default=4,
        metadata={"help": "Number of bits for QLoRA quantization. Options: 4 or 8."},
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={
            "help": "Additional modules to save in checkpoint. Comma-separated list."
        },
    )

    def __post_init__(self):
        """Parse comma-separated strings into lists."""
        if self.lora_target_modules is not None and self.lora_target_modules != "auto":
            self.lora_target_modules = self.lora_target_modules.split(",")
        if self.modules_to_save is not None:
            self.modules_to_save = self.modules_to_save.split(",")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Extended training arguments for Resonance SFT training.

    Extends the standard Transformers TrainingArguments with additional
    options specific to vision-language model training.
    """

    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA for parameter-efficient fine-tuning."},
    )
    use_flash_attention_2: bool = field(
        default=True,
        metadata={
            "help": "Use FlashAttention-2 for memory-efficient attention computation."
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes for parallel dataset preprocessing."},
    )
    project_name: Optional[str] = field(
        default="Resonance",
        metadata={"help": "Weights & Biases project name for experiment tracking."},
    )
    group_name: Optional[str] = field(
        default="Qwen-VL-Chat-sft",
        metadata={
            "help": "Weights & Biases group name for organizing related experiments."
        },
    )
    resume_from_checkpoint: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to resume training from the latest checkpoint in output_dir."
        },
    )


class PeftSavingCallback(TrainerCallback):
    """Custom callback for saving PEFT (LoRA) adapters during training.

    This callback ensures that LoRA adapters are properly saved at each
    checkpoint, and removes redundant full model files to save disk space.
    """

    def on_save(self, args, state, control, **kwargs):
        """Save PEFT adapter weights at checkpoint."""
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )

        # Save PEFT adapter
        kwargs["model"].save_pretrained(checkpoint_path)

        # Remove full model file to save space (we only need adapter weights)
        pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(pytorch_model_path)


def main():
    """Main training function for Resonance SFT."""
    # Parse command line arguments
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, LoraArguments))
    script_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Configure training settings
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # Setup experiment tracking
    os.environ["WANDB_PROJECT"] = training_args.project_name
    os.environ["WANDB_RUN_GROUP"] = training_args.group_name

    print(f"🚀 Starting Resonance SFT training...")
    print(f"📁 Model: {script_args.model_name_or_path}")
    print(f"📊 Dataset: {script_args.dataset_name}")
    print(f"🎯 LoRA: {training_args.use_lora}")

    try:
        # Load model and processor
        print("📦 Loading model and processor...")
        model, ref_model, lora_config = auto_load_rlmodel(
            script_args, training_args, lora_args
        )
        processor = MyAutoProcessor.from_pretrained(script_args.model_name_or_path)
        processor.train()  # Set processor to training mode

        # Load and prepare dataset
        print("📚 Loading dataset...")
        if script_args.dataset_name not in DATASET_MAP:
            raise ValueError(
                f"Unknown dataset: {script_args.dataset_name}. "
                f"Available: {list(DATASET_MAP.keys())}"
            )

        dataset = DATASET_MAP[script_args.dataset_name](script_args)

        # Split dataset for evaluation (0.5% for eval, rest for training)
        dataset_split = dataset.train_test_split(test_size=0.005, seed=42)
        train_dataset = dataset_split["train"]

        # Use only specified fraction of training data
        if script_args.data_ratio < 1.0:
            train_size = int(len(train_dataset) * script_args.data_ratio)
            train_dataset = train_dataset.select(range(train_size))
            print(
                f"📉 Using {script_args.data_ratio:.1%} of training data ({train_size} samples)"
            )

        eval_dataset = dataset_split["test"]
        print(
            f"📊 Training samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}"
        )

        # Setup data collator
        print("🔧 Setting up data collator...")
        collator = MyAutoSFTCollator(
            script_args.model_name_or_path,
            processor.tokenizer.pad_token_id,
            script_args.label_pad_token_id,
            processor=processor,
        )

    except Exception as e:
        print(f"❌ Error during setup: {e}")
        raise


if __name__ == "__main__":
    main()
    sft_trainer = MyAutoSFTTrainer(
        model_name_or_path=script_args.model_name_or_path,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processor=processor,
        max_seq_length=script_args.max_length,
        peft_config=lora_config,
        dataset_num_proc=training_args.dataset_num_proc,
    )
    if training_args.use_lora:
        sft_trainer.add_callback(PeftSavingCallback())
    sft_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    sft_trainer.save_state()
    safe_save_model_for_hf_trainer(sft_trainer, training_args.output_dir, lora_args)
    processor.save_pretrained(training_args.output_dir)
    if script_args.merge_peft_model and training_args.use_lora:
        merged_model = model.merge_peft_model()
        merged_dir = os.path.join(training_args.output_dir, "merged")
        merged_model.save_pretrained(merged_dir)
