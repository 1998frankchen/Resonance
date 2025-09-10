# Resonance: Where Machine Vision Meets Human Wisdom

*Built for Vision-Language Models with Wisdom*

<p align="center">
    <br>
    <img src="assets/resonance_banner.png" style="width: 75%">
    <br>
    <img src="assets/performance.svg" style="width: 75%">
    <br>
    <a href="CODE_OF_CONDUCT.md"><img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg"></a>
    <img alt="python version" src="https://img.shields.io/badge/python-%3E%3D3.10-orange">
    <img alt="Static Badge" src="https://img.shields.io/badge/pytorch-%3E%3D2.0-orange">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/1998frankchen/resonance">
    <img alt="GitHub License" src="https://img.shields.io/github/license/1998frankchen/resonance">
    <br>
</p>

> **Welcome to Resonance!** This project represents the evolution of Resonance, actively maintained by Frank Chen with enhanced features, comprehensive documentation, robust implementations, and demo examples to get you started quickly.

## 📝 Introduction
由于视觉语言模型（VLM、LVLM或MLLM）社区缺乏统一的模型架构，视觉语言模型的偏好学习研究和开发具有挑战性。像LLaVA、Qwen-VL和InternLM-XComposer这样的最先进视觉语言模型具有不同的实现风格，这使得很难将它们包含在单一的训练框架中。
或者更流畅的表达：
视觉语言模型（VLM、LVLM或MLLM）的偏好学习研究开发面临挑战，主要原因是VLM社区缺乏统一的模型架构。像LLaVA、Qwen-VL和InternLM-XComposer等最先进的视觉语言模型采用了不同的实现方式，这使得很难将它们整合到一个统一的训练框架中。   


Research and development on preference learning of Vision-Language Models (VLM, LVLM or MLLM) is challenging due to the lack of unified model architectures in the VLM community. State-of-the-art VLMs like LLaVA, Qwen-VL, and InternLM-XComposer have different implementation styles, making it difficult to include them in a single training framework.

**Resonance** provides a perfect solution to abstract VLMs in a unified framework where machine vision meets human wisdom through preference learning. Key features include:

- **Supports popular VLMs** : LLaVA-1.5, InstructBLIP, LLaVA-Next, Qwen-VL, InternLM-XComposer2, etc.
- **Supports popular fine-tuning methods** : SFT, DPO (DDPO & KTO still in precss), etc.
- **Evaluation on popular benchmarks** : MME, MMBench, SEEDBench, MMVet, MMMU, etc.
- **Easy to expand** : Customize your own dataset and model with few code.

## 🤖 Supported Models

- [InstructBLIP (Decoder-only)](https://huggingface.co/Salesforce/instructblip-vicuna-13b)
- [LLaVA-1.5](https://huggingface.co/collections/llava-hf/llava-15-65f762d5b6941db5c2ba07e0)
- [LLaVA-Next](https://huggingface.co/collections/llava-hf/llava-next-65f75c4afac77fd37dbbe6cf)
- [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [InternLM-XComposer2](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b)

## ⚙️ Supported Methods

- Supervised Fine-tuning (SFT)
- [Direct Preference Optimization (DPO)](https://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)
- [Proximal Policy Optimization (PPO)](http://arxiv.org/abs/1707.06347)
- - [Dense Direct Preference Optimization (DDPO)(in process)](http://arxiv.org/abs/2312.00849)
- [Kahneman-Tversky Optimization (KTO) (coming soon)](http://arxiv.org/abs/2402.01306)

## 🛠️ Installation

To install from source code (convenient for running the training and evaluation scripts), please run the following commands:

```bash
git clone https://github.com/1998frankchen/resonance.git
cd resonance
pip install -e .
```

We recommend to install FlashAttention for effective training and inference:

```bash
pip install flash-attn==2.5.8 --no-build-isolation
```

## 🎯 Quick Start with Demos

Get started quickly with our comprehensive demo examples:

```bash
# Check out the demos directory
ls demos/

# Run demo scripts to see training commands
bash demos/scripts/demo_sft.sh      # Supervised fine-tuning demo
bash demos/scripts/demo_dpo.sh      # Direct preference optimization demo
bash demos/scripts/demo_eval.sh     # Evaluation demo
```

The demos include:

- **Sample datasets** in JSON format for SFT and DPO training
- **Generated demo images** for visual question answering
- **Ready-to-use training scripts** with proper configurations
- **Evaluation examples** on various benchmarks

See [demos/README.md](demos/README.md) for detailed instructions.

## 🚀 Training

### Training Scripts

You can run the following command to launch DPO training of QwenVL-Chat using [VLFeedback](https://huggingface.co/datasets/MMInstruction/VLFeedback) dataset:

```bash
#model weights should exist in ckpts/Qwen-VL-Chat
bash scripts/dpo_qwenvl.sh
```

Or run the python file directly:

```bash
accelerate launch --config_file accelerate_config/zero2.yaml --num_processes 8 \
        src/resonance/dpo.py \
        --model_name_or_path ckpts/Qwen-VL-Chat \
        --output_dir ckpts/Qwen-VL-Chat-dpo/ \
        --dataset_name VLFeedback \
        --data_ratio 1.0 \
        --freeze_vision_tower True \
        --use_flash_attention_2 False \
        --use_lora True \
        --lora_r 64 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules auto \
        --lora_bias "none" \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --learning_rate 1e-5 \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --bf16 True \
        --tf32 True \
        --remove_unused_columns False \
        --beta 0.1 \
        --max_length 1024 \
        --max_prompt_length 512 \
        --max_target_length 512 \
        --eval_strategy "steps" \
        --eval_steps 200 \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 10 \
        --logging_first_step False \
        --logging_steps 10 \
        --report_to wandb \
        --run_name  "bs256_lr1e-5" \
        --project_name "Resonance" \
        --group_name "Qwen-VL-Chat-dpo"
```

To train other models with other methods, you can refer to the related scripts in `scripts/` directory.

Please refer to [arguments.md](docs/TrainingArguments.md) for detailed explanation of each arguments used in the scripts.

### Data Preparation


Resonance处理数据集时使用三个参数，在所有示例训练脚本中都可以找到。运行前请确保正确设置：     
--dataset_name 数据集名称。可选值：vlfeedback_paired（VLFeedback数据集）、rlhfv（RLHF-V数据集）、vlquery_json（自定义多模态对话数据）、plain_dpo（自定义多模态比较数据）。  
--data_path json文件路径。仅自定义数据集需要。VLFeedback和RLHF-V数据集会自动从huggingface下载。  
--image_root 图像根目录。仅自定义数据集需要。会与json中的图像路径拼接。  

Resonance uses three arguments when processing the given dataset, which can be found in all the example training scripts. Please make sure they are properly set in the script before running it:

- `--dataset_name` The name of the dataset. It can be `vlfeedback_paired` for the [VLFeedback](https://huggingface.co/datasets/MMInstruction/VLFeedback) dataset, `rlhfv` for the [RLHF-V](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) dataset, `vlquery_json` for customized multimodal conversation data stored in json format, `plain_dpo` for customized multimodal comparison data stored in json format.
- `--data_path` Path to the json file. **Only needed for customized dataset**. If you use VLFeedback or RLHF-V, these datasets will be automatically downloaded from huggingface and loaded via the `datasets` package.
- `--image_root` Root directory of the images. **Only needed for customized dataset**. It will be joined with the image path of each sample in the json file.

### Customized Dataset

For methods that need comparison data, e.g. DPO, please prepare your json data in the following format:

```json
[
    {
        "image":"example.jpg",
        "prompt":"Describe this image in detail.",
        "chosen":"This is a cat.",
        "rejected":"This is a dog."
    },
    ...
]
```

And set `--dataset_name` to `plain_dpo` in the training command.

For SFT, please prepare your conversation data in the following format:

```json
[
    {
        "image":"example.jpg",
        "conversations":[
            {
                "from": "user",
                "value": "<prompt>",
            },
            {
                "from": "assistant",
                "value": "<answer>",
            },
            ...
        ]
    },
    ...
]
```

And set `--dataset_name` to `vlquery_json` in the training command.

### Customized Model

You can easily add your own model to Resonance framework by implementing some APIs. Please refer to [CustomizedModel.md](docs/CustomizedModel.md)

## 📊 Evaluation

Resonance supports to evaluate VLMs on popular multimodal benchmarks like MME, MMVet, Seedbench, MMBench and so on. Please refer to the [Evaluation Guide](docs/EvaluationGuide.md) for details.

## 🥇 Performance

For reference, we report the performance of some models before and after DPO training on VLFeedback .

| Model | MMBench | MMVet | SEEDBench-Img | MMMU | MathVista |
| :-----: | :-----:  | :-----:  | :-----: | :-----:  | :-----:  |
| [InternLM-Xcomposer2-VL-7b](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b) | 76.37 | 46.5 | 74.19 | 40.33 | 56.7 |
| [InternLM-Xcomposer2-VL-7b-DPO](https://huggingface.co/1998frankchen/internlm-xcomposer2-vl-7b-DPO) | 78.18 | 49.7 | 75.18 |39.67 | 56.6 |
| [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) | 56.53 | 48.5 | 59.63 | 35.67 | 35.6 |
| [Qwen-VL-Chat-DPO](https://huggingface.co/1998frankchen/Qwen-VL-Chat-DPO) | 57.56 | 49.1 | 60.67 | 37.89 | 35.6 |
| [LLaVA-Next-Mistral-7b](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)     | 67.70 | 43.8 | 71.7 | 37.00 | 35.1 |
| [LLaVA-Next-Mistral-7b-DPO](https://huggingface.co/1998frankchen/llava-v1.6-mistral-7b-hf-DPO) | 68.30 | 44.2 | 71.7 | 36.89 | 36.2 |
| [LLaVA-Next-Vicuna-7b](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) | 62.71 | 38.2 | 68.17 | 34.00 | 31.3 |
| [LLaVA-Next-Vicuna-7b-DPO](https://huggingface.co/1998frankchen/llava-v1.6-vicuna-7b-hf-DPO) | 64.52 | 44.1 | 69.75 | 33.11 | 32.0 |

## 📄 License

This project is licensed under the Apache License. See `LICENSE` for details.
