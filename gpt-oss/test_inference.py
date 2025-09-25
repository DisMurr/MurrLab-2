#!/usr/bin/env python3
"""
Test script to verify GPT-OSS 20B installation and basic inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

def test_transformers_inference():
    """Test inference using Transformers library"""
    print("Testing GPT-OSS 20B inference with Transformers...")
    
    try:
        model_path = "gpt-oss-20b"
        
        # Load tokenizer and model
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"Loading model from {model_path}...")
        print("Note: This will use significant GPU memory (16GB+)")
        
        # Use device_map for better memory management
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Test prompt
        messages = [
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        
        # Apply chat template (uses harmony format automatically)
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs.to(model.device),
                max_new_tokens=256,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print("\n" + "="*60)
        print("GPT-OSS 20B Response:")
        print("="*60)
        print(response)
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Error during inference test: {e}")
        return False

def test_native_inference():
    """Test inference using native GPT-OSS implementation"""
    print("\nTesting GPT-OSS native implementation...")
    
    try:
        from gpt_oss.chat import main as chat_main
        print("Native GPT-OSS chat module imported successfully!")
        return True
    except Exception as e:
        print(f"Error importing native GPT-OSS: {e}")
        return False

def main():
    print("GPT-OSS 20B Installation Test")
    print("="*50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("WARNING: CUDA not available. This will be very slow!")
    
    # Test native implementation import
    native_success = test_native_inference()
    
    # Test Transformers inference (only if user confirms)
    print("\n" + "="*50)
    response = input("Run Transformers inference test? This will load the full 20B model (y/N): ").strip().lower()
    
    if response == 'y':
        transformers_success = test_transformers_inference()
    else:
        print("Skipping Transformers inference test.")
        transformers_success = None
    
    # Summary
    print("\n" + "="*50)
    print("INSTALLATION TEST SUMMARY")
    print("="*50)
    print(f"Native GPT-OSS import: {'✅ PASSED' if native_success else '❌ FAILED'}")
    if transformers_success is not None:
        print(f"Transformers inference: {'✅ PASSED' if transformers_success else '❌ FAILED'}")
    else:
        print("Transformers inference: ⏭️  SKIPPED")
    print("="*50)

if __name__ == "__main__":
    main()