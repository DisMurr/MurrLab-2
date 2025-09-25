#!/usr/bin/env python3
"""
Simple test for GPT-OSS 20B native implementation
"""

import sys
import os

def test_native_gpt_oss():
    """Test the native GPT-OSS chat functionality"""
    print("Testing native GPT-OSS implementation...")
    
    try:
        # Test import
        import gpt_oss
        from gpt_oss.chat import main as chat_main
        print("✅ GPT-OSS native module imported successfully")
        
        # Test harmony
        from openai_harmony import (
            HarmonyEncodingName,
            load_harmony_encoding,
            Conversation,
            Message,
            Role,
            SystemContent,
        )
        print("✅ Harmony format module imported successfully")
        
        # Test model path
        if os.path.exists("gpt-oss-20b/original/model.safetensors"):
            print("✅ Model files found")
            model_size_gb = os.path.getsize("gpt-oss-20b/original/model.safetensors") // (1024**3)
            print(f"✅ Model size: {model_size_gb}GB")
        else:
            print("❌ Model files not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing native GPT-OSS: {e}")
        return False

def test_transformers_import():
    """Test transformers and related imports"""
    print("\nTesting Transformers and fine-tuning libraries...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Transformers imported")
        
        from peft import LoraConfig, TaskType, get_peft_model
        print("✅ PEFT imported")
        
        from datasets import Dataset
        print("✅ Datasets imported")
        
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        else:
            print("⚠️  CUDA not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing libraries: {e}")
        return False

def main():
    print("🧪 GPT-OSS 20B Simple Test")
    print("=" * 50)
    
    # Test native implementation
    native_ok = test_native_gpt_oss()
    
    # Test transformers and libraries
    libs_ok = test_transformers_import()
    
    print("\n" + "=" * 50)
    print("SIMPLE TEST SUMMARY")
    print("=" * 50)
    print(f"Native GPT-OSS: {'✅ PASSED' if native_ok else '❌ FAILED'}")
    print(f"Libraries: {'✅ PASSED' if libs_ok else '❌ FAILED'}")
    
    if native_ok and libs_ok:
        print("\n🎉 GPT-OSS 20B is ready for fine-tuning!")
        print("\nNext steps:")
        print("1. Run: ./quick_train.sh")
        print("2. Or: python finetune.py --dataset_path sample_training_data.json")
        print("\nFor native GPT-OSS chat:")
        print("python -m gpt_oss.chat gpt-oss-20b/original/ -r low")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()