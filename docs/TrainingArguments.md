# Resonance Training Arguments Guide

This comprehensive guide covers all training arguments available in **Resonance** (formerly Resonance). Resonance builds on [Transformers](https://github.com/huggingface/transformers), so all standard Transformers training arguments are also available.

## 📋 Table of Contents

- [Common Arguments](#common-arguments)
- [Model & Architecture](#model--architecture)
- [Dataset Configuration](#dataset-configuration)  
- [LoRA & Efficient Training](#lora--efficient-training)
- [Training Hyperparameters](#training-hyperparameters)
- [DPO-Specific Arguments](#dpo-specific-arguments)
- [PPO-Specific Arguments](#ppo-specific-arguments)
- [Evaluation & Logging](#evaluation--logging)
- [Advanced Options](#advanced-options)

---

## Common Arguments

These arguments are shared across all Resonance trainers (SFT, DPO, PPO, etc.):

### Model & Architecture

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name_or_path` | str | **Required** | Path or HuggingFace Hub ID of the pretrained model |
| `--freeze_vision_tower` | bool | `True` | Whether to freeze the vision encoder during training |
| `--use_flash_attention_2` | bool | `False` | Enable FlashAttention-2 for memory-efficient training |

### Dataset Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_name` | str | **Required** | Dataset type identifier (see [Dataset Types](#dataset-types)) |
| `--data_path` | str | `None` | Path to custom dataset JSON file |
| `--image_root` | str | `None` | Root directory for images (for custom datasets) |
| `--data_ratio` | float | `1.0` | Ratio of data to use for training |
| `--dataset_num_proc` | int | `4` | Number of processes for data preprocessing |

#### Dataset Types

| `dataset_name` | Description | Use Case |
|----------------|-------------|----------|
| `vlfeedback_paired` | [VLFeedback](https://huggingface.co/datasets/MMInstruction/VLFeedback) dataset | DPO training with preference pairs |
| `rlhfv` | [RLHF-V](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) dataset | RLHF training |
| `vlquery_json` | Custom conversation data in JSON format | SFT training |
| `plain_dpo` | Custom comparison data in JSON format | DPO training |

### Length Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_length` | int | `2048` | Maximum sequence length for the entire input |
| `--max_prompt_length` | int | `1024` | Maximum length for the prompt portion |
| `--max_target_length` | int | `1024` | Maximum length for the target/response portion |

---

## LoRA & Efficient Training

Resonance supports Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) for parameter-efficient training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_lora` | bool | `False` | Enable LoRA fine-tuning |
| `--lora_r` | int | `64` | LoRA rank (higher = more parameters, better quality) |
| `--lora_alpha` | int | `16` | LoRA scaling parameter |
| `--lora_dropout` | float | `0.05` | LoRA dropout rate |
| `--lora_target_modules` | str | `"auto"` | Target modules for LoRA (comma-separated or "auto") |
| `--lora_bias` | str | `"none"` | LoRA bias strategy: "none", "all", or "lora_only" |
| `--q_lora` | bool | `False` | Enable QLoRA (quantized LoRA) |
| `--bits` | int | `4` | Quantization bits for QLoRA (4 or 8) |
| `--modules_to_save` | str | `None` | Additional modules to save (comma-separated) |

### LoRA Target Modules Examples

```bash
# For attention layers only
--lora_target_modules "c_attn,attn.c_proj"

# For attention and MLP layers  
--lora_target_modules "c_attn,attn.c_proj,w1,w2"

# Let Resonance auto-detect optimal targets
--lora_target_modules "auto"
```

---

## Training Hyperparameters

Standard training arguments from Transformers. Key recommendations:

| Argument | Recommended | Notes |
|----------|-------------|-------|
| `--per_device_train_batch_size` | `2-8` | Adjust based on GPU memory |
| `--gradient_accumulation_steps` | `4-16` | Increase for larger effective batch size |
| `--learning_rate` | `1e-5` to `5e-5` | Lower for stability |
| `--weight_decay` | `0.01-0.05` | Regularization |
| `--warmup_ratio` | `0.1` | 10% warmup steps |
| `--lr_scheduler_type` | `"cosine"` | Learning rate decay |
| `--num_train_epochs` | `1-3` | Few epochs usually sufficient |
| `--bf16` | `True` | Use bfloat16 if supported |
| `--gradient_checkpointing` | `True` | Save memory at cost of speed |

---

## DPO-Specific Arguments

For Direct Preference Optimization training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--beta` | float | `0.1` | DPO temperature parameter (higher = more conservative) |
| `--score_margin` | float | `0.0` | Margin for preference scores |

---

## PPO-Specific Arguments

For Proximal Policy Optimization training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--kl_coef` | float | `0.1` | KL divergence coefficient |
| `--cliprange` | float | `0.2` | PPO clipping range |
| `--vf_coef` | float | `0.5` | Value function coefficient |

---

## Evaluation & Logging

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--eval_strategy` | str | `"steps"` | Evaluation strategy: "no", "steps", "epoch" |
| `--eval_steps` | int | `500` | Evaluate every N steps |
| `--save_strategy` | str | `"steps"` | Save strategy: "no", "steps", "epoch" |  
| `--save_steps` | int | `1000` | Save every N steps |
| `--save_total_limit` | int | `3` | Maximum number of checkpoints to keep |
| `--logging_steps` | int | `50` | Log every N steps |
| `--report_to` | str | `"wandb"` | Logging service: "wandb", "tensorboard", "none" |
| `--project_name` | str | `"Resonance"` | Project name for logging |
| `--group_name` | str | `None` | Experiment group name |
| `--run_name` | str | `None` | Specific run name |

---

## Advanced Options

### Memory Optimization

```bash
# For large models on limited memory
--gradient_checkpointing True \
--use_lora True \
--q_lora True \
--bits 4 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```

### Multi-GPU Training

```bash
# Use with accelerate
accelerate launch --config_file accelerate_config/zero2.yaml \
    --num_processes 4 \
    src/resonance/dpo.py \
    # ... other arguments
```

### Custom Datasets

For custom datasets, prepare JSON files following these formats:

**SFT Dataset (`vlquery_json`):**
```json
[
  {
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "user", "value": "What's in this image?"},
      {"from": "assistant", "value": "I can see..."}
    ]
  }
]
```

**DPO Dataset (`plain_dpo`):**
```json
[
  {
    "image": "path/to/image.jpg", 
    "prompt": "Describe this image",
    "chosen": "High quality response",
    "rejected": "Low quality response"
  }
]
```

---

## Example Configurations

### Quick Demo Training
```bash
python src/resonance/sft.py \
    --model_name_or_path microsoft/DialoGPT-medium \
    --dataset_name vlquery_json \
    --data_path demos/data/demo_sft.json \
    --image_root demos/images \
    --max_length 512 \
    --per_device_train_batch_size 2 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --use_lora True \
    --lora_r 16 \
    --report_to none
```

### Production DPO Training  
```bash
accelerate launch --config_file accelerate_config/zero2.yaml \
    src/resonance/dpo.py \
    --model_name_or_path Qwen/Qwen-VL-Chat \
    --dataset_name vlfeedback_paired \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 16 \
    --beta 0.1 \
    --max_length 1024 \
    --bf16 True \
    --gradient_checkpointing True \
    --project_name "Resonance" \
    --group_name "Qwen-VL-DPO"
```

For more examples, check out the [demos](../demos/) directory!