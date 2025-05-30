#!/bin/bash
# Demo SFT Training Script for Resonance
# This script demonstrates supervised fine-tuning on a small demo dataset

echo "Starting Demo SFT Training for Resonance..."

# Demo configuration - smaller parameters for quick testing
per_device_train_batch_size=1
gradient_accumulation_steps=2
epoch=1
lr=5e-5

# Use CPU or single GPU for demo
num_processes=1

# Demo dataset paths
dataset_name="vlquery_json"
data_path="demos/data/demo_sft.json"
image_root="demos/images"

# Output directory
output_dir="demos/checkpoints/demo-sft"
mkdir -p $output_dir

echo "Configuration:"
echo "  Dataset: $data_path"
echo "  Images: $image_root"
echo "  Output: $output_dir"
echo "  Batch size: $per_device_train_batch_size"
echo "  Epochs: $epoch"
echo "  Learning rate: $lr"
echo ""

# Note: This is a demo script. In practice, you would need:
# 1. A pre-trained vision-language model (like Qwen-VL-Chat)
# 2. Proper GPU setup
# 3. Larger dataset

echo "Demo script created. To run actual training, you would execute:"
echo ""
echo "accelerate launch --num_processes $num_processes \\"
echo "    src/resonance/sft.py \\"
echo "    --model_name_or_path <path-to-pretrained-model> \\"
echo "    --output_dir $output_dir \\"
echo "    --dataset_name $dataset_name \\"
echo "    --data_path $data_path \\"
echo "    --image_root $image_root \\"
echo "    --freeze_vision_tower True \\"
echo "    --use_flash_attention_2 False \\"
echo "    --use_lora True \\"
echo "    --lora_r 16 \\"
echo "    --lora_alpha 32 \\"
echo "    --lora_dropout 0.05 \\"
echo "    --per_device_train_batch_size $per_device_train_batch_size \\"
echo "    --gradient_accumulation_steps $gradient_accumulation_steps \\"
echo "    --num_train_epochs $epoch \\"
echo "    --learning_rate $lr \\"
echo "    --weight_decay 0.01 \\"
echo "    --warmup_ratio 0.1 \\"
echo "    --lr_scheduler_type cosine \\"
echo "    --gradient_checkpointing False \\"
echo "    --bf16 False \\"
echo "    --fp16 True \\"
echo "    --remove_unused_columns False \\"
echo "    --max_length 512 \\"
echo "    --save_strategy steps \\"
echo "    --save_steps 50 \\"
echo "    --logging_steps 10 \\"
echo "    --report_to none \\"
echo "    --run_name demo-sft-run"

echo ""
echo "Demo SFT script ready! Check demos/data/demo_sft.json for the sample dataset."