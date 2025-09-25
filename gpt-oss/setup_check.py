#!/usr/bin/env python3
"""
GPT-OSS Setup and Environment Check Script
"""

import subprocess
import sys
import os
import json
from pathlib import Path
import importlib.util

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_python_packages():
    """Check if required Python packages are installed"""
    required_packages = [
        "torch", "transformers", "datasets", "accelerate", 
        "huggingface_hub", "peft", "trl", "bitsandbytes"
    ]
    
    missing_packages = []
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def check_model_files():
    """Check if model files are downloaded"""
    model_dir = Path("gpt-oss-20b")
    required_files = [
        "config.json",
        "tokenizer.json",
        "original/model.safetensors",
        "model-00000-of-00002.safetensors"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (model_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All model files are present")
        return True

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"✅ GPU available: {gpu_name} ({memory_gb}GB)")
            
            if memory_gb >= 16:
                print("✅ Sufficient GPU memory for GPT-OSS 20B")
                return "optimal"
            elif memory_gb >= 8:
                print("⚠️  Limited GPU memory - quantization recommended")
                return "limited"
            else:
                print("❌ Insufficient GPU memory for GPT-OSS 20B")
                return "insufficient"
        else:
            print("❌ No GPU available - training will be very slow")
            return "none"
    except ImportError:
        print("❌ PyTorch not installed")
        return "error"

def create_quick_start_scripts():
    """Create quick start scripts"""
    
    # Training script
    training_script = """#!/bin/bash
# Quick start training script for GPT-OSS 20B

echo "🚀 Starting GPT-OSS 20B fine-tuning..."

# Check if sample data exists
if [ ! -f "sample_training_data.json" ]; then
    echo "📊 Creating sample training dataset..."
    python finetune.py --create_sample
fi

# Start training with memory-efficient settings
echo "🎯 Starting fine-tuning with LoRA..."
python finetune.py \\
    --dataset_path sample_training_data.json \\
    --output_dir gpt-oss-20b-finetuned \\
    --epochs 3 \\
    --learning_rate 1e-4 \\
    --batch_size 1 \\
    --gradient_accumulation_steps 8 \\
    --use_lora

echo "✅ Fine-tuning completed! Model saved to: gpt-oss-20b-finetuned"
"""
    
    with open("quick_train.sh", "w") as f:
        f.write(training_script)
    os.chmod("quick_train.sh", 0o755)
    
    # Test script
    test_script = """#!/bin/bash
# Quick test script for GPT-OSS 20B

echo "🧪 Testing GPT-OSS 20B installation..."

# Test with memory optimization
echo "Testing with memory optimizations..."
python test_memory_optimized.py

echo "✅ Test completed!"
"""
    
    with open("quick_test.sh", "w") as f:
        f.write(test_script)
    os.chmod("quick_test.sh", 0o755)
    
    print("✅ Created quick start scripts: quick_train.sh, quick_test.sh")

def main():
    print("🔍 GPT-OSS 20B Setup Check")
    print("=" * 50)
    
    # Check Python packages
    packages_ok = check_python_packages()
    
    # Check model files
    model_files_ok = check_model_files()
    
    # Check GPU
    gpu_status = check_gpu()
    
    # Create quick start scripts
    create_quick_start_scripts()
    
    print("\n📋 SETUP SUMMARY")
    print("=" * 50)
    print(f"Python packages: {'✅' if packages_ok else '❌'}")
    print(f"Model files: {'✅' if model_files_ok else '❌'}")
    print(f"GPU status: {gpu_status}")
    
    if packages_ok and model_files_ok:
        print("\n🎉 GPT-OSS 20B is ready!")
        print("\n🚀 Quick start options:")
        print("1. Test installation: ./quick_test.sh")
        print("2. Start fine-tuning: ./quick_train.sh")
        print("3. Create custom dataset: python finetune.py --create_sample")
        print("4. Run custom training: python finetune.py --dataset_path your_data.json")
        
        if gpu_status == "limited":
            print("\n⚠️  Memory optimization tips:")
            print("- Use quantization: Add --load_in_8bit flag")
            print("- Reduce batch size: --batch_size 1")
            print("- Increase gradient accumulation: --gradient_accumulation_steps 16")
            
    else:
        print("\n❌ Setup incomplete. Please address the issues above.")
        
        if not packages_ok:
            print("\n📦 To install missing packages:")
            print("pip install torch transformers datasets accelerate huggingface_hub peft trl bitsandbytes")
            
        if not model_files_ok:
            print("\n📥 To download model files:")
            print("huggingface-cli download openai/gpt-oss-20b --local-dir gpt-oss-20b")

if __name__ == "__main__":
    main()