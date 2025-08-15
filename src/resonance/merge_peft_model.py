"""Resonance PEFT Model Merger Utility.

This utility script merges LoRA/QLoRA adapter weights back into the base model,
creating a standalone model that doesn't require separate adapter files. This is
useful for model deployment, sharing, or further fine-tuning.

🎯 **Purpose:**
- Merge LoRA/QLoRA adapters with base models
- Create deployable standalone models
- Reduce inference complexity and dependencies
- Enable easier model sharing and distribution

🔧 **Features:**
- Automatic base model detection and loading
- Safe merge and unload operations
- Model architecture preservation
- JSON configuration output for tracking

⚠️ **Important Notes:**
- Merged models are larger than adapter-only models
- Original base model and adapter are preserved
- Merged model maintains full capabilities of the adapted model
- Compatible with all Resonance-supported model architectures

Usage:
    python src/resonance/merge_peft_model.py --adapter_path path/to/lora/model

Author: Frank Chen (Resonance Team)
Repo: https://github.com/1998frankchen/resonance
"""

from resonance.utils.auto_load import MyAutoModel
from argparse import ArgumentParser
import os
from loguru import logger
import json

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--adapter_path", type=str)
    args = parser.parse_args()
    peft_model = MyAutoModel.from_pretrained(args.adapter_path)
    architectures = peft_model.config.architectures
    logger.info("Merging and unloading model...")
    merged_model = peft_model.merge_and_unload()
    save_path = os.path.join(args.adapter_path, "merged")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving merged model to {save_path}...")
    merged_model.save_pretrained(save_path)
    with open(os.path.join(save_path, "config.json"), "r") as f:
        config = json.load(f)
    config["architectures"] = architectures
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f)
    logger.success("Done!")
