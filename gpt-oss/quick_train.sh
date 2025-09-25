#!/bin/bash
# Quick start training script for GPT-OSS 20B

echo "🚀 Starting GPT-OSS 20B fine-tuning..."

# Check if sample data exists
if [ ! -f "sample_training_data.json" ]; then
    echo "📊 Creating sample training dataset..."
    /home/murr/Project/.venv/bin/python finetune.py --create_sample
fi

# Start training with memory-efficient settings
echo "🎯 Starting fine-tuning with LoRA..."
/home/murr/Project/.venv/bin/python finetune.py \
    --dataset_path sample_training_data.json \
    --output_dir gpt-oss-20b-finetuned \
    --epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --use_lora

echo "✅ Fine-tuning completed! Model saved to: gpt-oss-20b-finetuned"
