# Resonance 训练参数指南

本综合指南涵盖了 **Resonance**（原Resonance）中所有可用的训练参数。Resonance基于 [Transformers](https://github.com/huggingface/transformers) 构建，因此所有标准Transformers训练参数也可用。

## �� 目录

- [通用参数](#通用参数)
- [模型与架构](#模型与架构)
- [数据集配置](#数据集配置)
- [LoRA与高效训练](#lora与高效训练)
- [训练超参数](#训练超参数)
- [DPO特定参数](#dpo特定参数)
- [PPO特定参数](#ppo特定参数)
- [评估与日志](#评估与日志)
- [高级选项](#高级选项)

---

## 通用参数

这些参数在所有Resonance训练器（SFT、DPO、PPO等）中共享：

### 模型与架构

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model_name_or_path` | str | **必需** | 预训练模型的路径或HuggingFace Hub ID |
| `--freeze_vision_tower` | bool | `True` | 训练期间是否冻结视觉编码器 |
| `--use_flash_attention_2` | bool | `False` | 启用FlashAttention-2进行内存高效训练 |

### 数据集配置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--dataset_name` | str | **必需** | 数据集类型标识符（参见[数据集类型](#数据集类型)） |
| `--data_path` | str | `None` | 自定义数据集JSON文件路径 |
| `--image_root` | str | `None` | 图像根目录（用于自定义数据集） |
| `--data_ratio` | float | `1.0` | 用于训练的数据比例 |
| `--dataset_num_proc` | int | `4` | 数据预处理的进程数 |

#### 数据集类型

| `dataset_name` | 描述 | 使用场景 |
|----------------|------|----------|
| `vlfeedback_paired` | [VLFeedback](https://huggingface.co/datasets/MMInstruction/VLFeedback) 数据集 | 带偏好对的DPO训练 |
| `rlhfv` | [RLHF-V](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) 数据集 | RLHF训练 |
| `vlquery_json` | JSON格式的自定义对话数据 | SFT训练 |
| `plain_dpo` | JSON格式的自定义比较数据 | DPO训练 |

### 长度配置

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--max_length` | int | `2048` | 整个输入的最大序列长度 |
| `--max_prompt_length` | int | `1024` | 提示部分的最大长度 |
| `--max_target_length` | int | `1024` | 目标/响应部分的最大长度 |

---

## LoRA与高效训练

Resonance支持低秩适应（LoRA）和量化LoRA（QLoRA）进行参数高效训练：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--use_lora` | bool | `False` | 启用LoRA微调 |
| `--lora_r` | int | `64` | LoRA秩（越高=更多参数，更好质量） |
| `--lora_alpha` | int | `16` | LoRA缩放参数 |
| `--lora_dropout` | float | `0.05` | LoRA dropout率 |
| `--lora_target_modules` | str | `"auto"` | LoRA目标模块（逗号分隔或"auto"） |
| `--lora_bias` | str | `"none"` | LoRA偏置策略："none"、"all"或"lora_only" |
| `--q_lora` | bool | `False` | 启用QLoRA（量化LoRA） |
| `--bits` | int | `4` | QLoRA量化位数（4或8） |
| `--modules_to_save` | str | `None` | 要保存的额外模块（逗号分隔） |

### LoRA目标模块示例

```bash
# 仅用于注意力层
--lora_target_modules "c_attn,attn.c_proj"

# 用于注意力和MLP层
--lora_target_modules "c_attn,attn.c_proj,w1,w2"

# 让Resonance自动检测最优目标
--lora_target_modules "auto"
```

---

## 训练超参数

来自Transformers的标准训练参数。关键建议：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--per_device_train_batch_size` | `2-8` | 根据GPU内存调整 |
| `--gradient_accumulation_steps` | `4-16` | 增加以获得更大的有效批次大小 |
| `--learning_rate` | `1e-5` 到 `5e-5` | 较低值更稳定 |
| `--weight_decay` | `0.01-0.05` | 正则化 |
| `--warmup_ratio` | `0.1` | 10%预热步数 |
| `--lr_scheduler_type` | `"cosine"` | 学习率衰减 |
| `--num_train_epochs` | `1-3` | 通常几个epoch就足够 |
| `--bf16` | `True` | 如果支持则使用bfloat16 |
| `--gradient_checkpointing` | `True` | 以速度为代价节省内存 |

---

## DPO特定参数

用于直接偏好优化训练：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--beta` | float | `0.1` | DPO温度参数（越高=越保守） |
| `--score_margin` | float | `0.0` | 偏好分数边距 |

---

## PPO特定参数

用于近端策略优化训练：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--kl_coef` | float | `0.1` | KL散度系数 |
| `--cliprange` | float | `0.2` | PPO裁剪范围 |
| `--vf_coef` | float | `0.5` | 价值函数系数 |

---

## 评估与日志

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--eval_strategy` | str | `"steps"` | 评估策略："no"、"steps"、"epoch" |
| `--eval_steps` | int | `500` | 每N步评估一次 |
| `--save_strategy` | str | `"steps"` | 保存策略："no"、"steps"、"epoch" |
| `--save_steps` | int | `1000` | 每N步保存一次 |
| `--save_total_limit` | int | `3` | 保留的最大检查点数 |
| `--logging_steps` | int | `50` | 每N步记录一次 |
| `--report_to` | str | `"wandb"` | 日志服务："wandb"、"tensorboard"、"none" |
| `--project_name` | str | `"Resonance"` | 日志项目名称 |
| `--group_name` | str | `None` | 实验组名称 |
| `--run_name` | str | `None` | 特定运行名称 |

---

## 高级选项

### 内存优化

```bash
# 用于有限内存上的大型模型
--gradient_checkpointing True \
--use_lora True \
--q_lora True \
--bits 4 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```

### 多GPU训练

```bash
# 与accelerate一起使用
accelerate launch --config_file accelerate_config/zero2.yaml \
    --num_processes 4 \
    src/resonance/dpo.py \
    # ... 其他参数
```

### 自定义数据集

对于自定义数据集，请按照以下格式准备JSON文件：

**SFT数据集（`vlquery_json`）：**
```json
[
  {
    "image": "path/to/image.jpg",
    "conversations": [
      {"from": "user", "value": "这张图片里有什么？"},
      {"from": "assistant", "value": "我可以看到..."}
    ]
  }
]
```

**DPO数据集（`plain_dpo`）：**
```json
[
  {
    "image": "path/to/image.jpg",
    "prompt": "描述这张图片",
    "chosen": "高质量回答",
    "rejected": "低质量回答"
  }
]
```

---

## 配置示例

### 快速演示训练
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

### 生产DPO训练
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

更多示例，请查看 [demos](../demos/) 目录！

## 参数调优建议

### 1. 内存优化策略
- **小内存GPU**：使用QLoRA + 梯度检查点
- **中等内存GPU**：使用LoRA + 混合精度
- **大内存GPU**：全参数微调 + 大批次

### 2. 学习率选择
- **SFT训练**：1e-5 到 5e-5
- **DPO训练**：1e-5 到 2e-5（更保守）
- **PPO训练**：5e-6 到 1e-5（最保守）

### 3. 批次大小配置
- **单GPU**：per_device_train_batch_size = 2-4
- **多GPU**：per_device_train_batch_size = 1-2
- **梯度累积**：gradient_accumulation_steps = 4-16

### 4. 训练轮数建议
- **SFT**：1-3个epoch
- **DPO**：1-2个epoch
- **PPO**：1个epoch（通常足够）

## 常见问题

### Q: 如何选择LoRA参数？
A: 从r=16开始，根据任务复杂度调整。复杂任务使用r=64，简单任务使用r=16。

### Q: 什么时候使用QLoRA？
A: 当GPU内存不足时，QLoRA可以显著减少内存使用，但可能略微影响性能。

### Q: 如何确定学习率？
A: 从1e-5开始，如果训练不稳定则降低，如果收敛太慢则适当提高。

### Q: 批次大小如何选择？
A: 在内存允许的情况下尽可能大，使用梯度累积来模拟更大的批次。


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
