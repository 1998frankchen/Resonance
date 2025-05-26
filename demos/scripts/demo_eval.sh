#!/bin/bash
# Demo Evaluation Script for Resonance
# This script demonstrates how to evaluate vision-language models

echo "Starting Demo Evaluation for Resonance..."

# Demo configuration
model_path="<path-to-fine-tuned-model>"
eval_batch_size=1

echo "Configuration:"
echo "  Model: $model_path"
echo "  Batch size: $eval_batch_size"
echo ""

echo "Available evaluation benchmarks in Resonance:"
echo "  - MME: Multi-Modal Evaluation"
echo "  - MMBench: Multi-Modal Benchmark"  
echo "  - SEEDBench: Seed Benchmark for Images"
echo "  - MMVet: Multi-Modal Veterinary Test"
echo "  - MMMU: Multi-Modal Multi-University"
echo "  - MathVista: Mathematical Visual Question Answering"
echo "  - POPE: Polling-based Object Probing Evaluation"
echo "  - VQA: Visual Question Answering"
echo ""

echo "Demo evaluation commands:"
echo ""
echo "# Example 1: MME Evaluation"
echo "bash scripts/eval/mme.sh \$MODEL_PATH \$OUTPUT_DIR"
echo ""
echo "# Example 2: MMBench Evaluation"  
echo "bash scripts/eval/mmbench.sh \$MODEL_PATH \$OUTPUT_DIR"
echo ""
echo "# Example 3: Custom VQA Evaluation"
echo "python src/resonance/eval/vqa/generate.py \\"
echo "    --model_name_or_path \$MODEL_PATH \\"
echo "    --dataset_name vqa_demo \\"
echo "    --data_path demos/data/demo_eval_vqa.json \\"
echo "    --image_root demos/images \\"
echo "    --output_path demos/results/vqa_results.json"

# Create a small eval dataset
mkdir -p demos/data
cat > demos/data/demo_eval_vqa.json << 'EOF'
[
    {
        "image": "demo_01.jpg",
        "question": "What animal is in this image?",
        "answer": "cat"
    },
    {
        "image": "demo_02.jpg", 
        "question": "Where is the dog playing?",
        "answer": "park"
    },
    {
        "image": "demo_03.jpg",
        "question": "What time of day is shown?",
        "answer": "sunset"
    }
]
EOF

echo ""
echo "Created demo evaluation dataset: demos/data/demo_eval_vqa.json"
echo "Demo evaluation script ready!"