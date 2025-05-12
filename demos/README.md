# Resonance Demos

This directory contains demo scripts, datasets, and examples to help you get started with Resonance (formerly Resonance) quickly.

## 📁 Directory Structure

```
demos/
├── README.md              # This file
├── create_sample_images.py # Script to generate demo images
├── data/                   # Demo datasets
│   ├── demo_sft.json      # Supervised fine-tuning demo data
│   ├── demo_dpo.json      # Direct preference optimization demo data
│   └── demo_eval_vqa.json # Evaluation demo data (VQA)
├── images/                 # Demo images (auto-generated)
│   ├── demo_01.jpg        # Sample image: Cat sitting on chair
│   ├── demo_02.jpg        # Sample image: Dog playing in park
│   └── ...                # More demo images
└── scripts/               # Demo training/evaluation scripts
    ├── demo_sft.sh        # Supervised fine-tuning demo
    ├── demo_dpo.sh        # Direct preference optimization demo
    └── demo_eval.sh       # Evaluation demo
```

## 🚀 Quick Start

### 1. Setup Environment

Ensure you have activated the Resonance virtual environment:

```bash
source .venv/bin/activate
```

### 2. Generate Demo Images (Optional)

The demo images are already created, but you can regenerate them:

```bash
python demos/download_real_images.py
```

### 3. Run Demo Scripts

#### Supervised Fine-Tuning (SFT)
```bash
bash demos/scripts/demo_sft.sh
```

This will show you the command structure for SFT training with a small conversational dataset.

#### Direct Preference Optimization (DPO)
```bash  
bash demos/scripts/demo_dpo.sh
```

This demonstrates DPO training with preference pairs (chosen vs rejected responses).

#### Evaluation
```bash
bash demos/scripts/demo_eval.sh
```

This shows how to evaluate models on various benchmarks.

## 📊 Demo Datasets

### SFT Dataset (`demo_sft.json`)
- Format: Conversational format with image, user questions, and assistant responses
- Size: 5 examples
- Use case: Supervised fine-tuning for instruction following

### DPO Dataset (`demo_dpo.json`)  
- Format: Preference pairs with chosen and rejected responses
- Size: 5 examples
- Use case: Direct preference optimization to align model outputs

### Evaluation Dataset (`demo_eval_vqa.json`)
- Format: Visual question answering pairs
- Size: 3 examples  
- Use case: Model evaluation on VQA tasks

## 🎯 Next Steps

1. **Download a pre-trained model** (e.g., Qwen-VL-Chat, LLaVA-1.5)
2. **Prepare your own dataset** following the demo formats
3. **Modify the scripts** with your model path and dataset paths
4. **Run training** with proper GPU setup
5. **Evaluate** your fine-tuned model

## 📝 Notes

- These demos use minimal configurations suitable for quick testing
- For production training, increase batch sizes, epochs, and use multi-GPU setup
- Ensure you have sufficient GPU memory for the models you want to train
- The demo images are synthetic and created for illustration purposes

## 🔗 Related Documentation

- [Training Arguments](../docs/TrainingArguments.md)
- [Evaluation Guide](../docs/EvaluationGuide.md)  
- [Customized Model](../docs/CustomizedModel.md)

Happy training with Resonance! 🎉