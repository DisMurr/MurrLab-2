#!/usr/bin/env python3
"""
Memory-optimized test script for GPT-OSS 20B with 8-bit quantization and optimizations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import os
import gc

def test_memory_optimized_inference():
    """Test inference using 8-bit quantization and memory optimizations"""
    print("Testing GPT-OSS 20B inference with memory optimizations...")
    
    try:
        model_path = "gpt-oss-20b"
        
        # Setup quantization config for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Loading quantized model from {model_path}...")
        print("Using 8-bit quantization for memory efficiency...")
        
        # Load model with aggressive memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory={0: "20GB", "cpu": "30GB"}
        )
        
        # Test prompt
        messages = [
            {"role": "user", "content": "What is 2+2?"}
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
                max_new_tokens=50,  # Shorter response to save memory
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print("\n" + "="*60)
        print("GPT-OSS 20B Response:")
        print("="*60)
        print(response)
        print("="*60)
        
        # Clean up
        del model
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"Error during inference test: {e}")
        # Clean up on error
        gc.collect()
        torch.cuda.empty_cache()
        return False

def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Get memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_cached = torch.cuda.memory_reserved(0)
        memory_free = memory_total - memory_allocated
        
        print(f"Total GPU Memory: {memory_total // 1024**3:.1f} GB")
        print(f"Allocated Memory: {memory_allocated // 1024**3:.1f} GB")
        print(f"Cached Memory: {memory_cached // 1024**3:.1f} GB")
        print(f"Free Memory: {memory_free // 1024**3:.1f} GB")
        
        return memory_free > 10 * 1024**3  # Need at least 10GB
    return False

def main():
    print("GPT-OSS 20B Memory-Optimized Test")
    print("="*50)
    
    # Check GPU memory
    if not check_gpu_memory():
        print("WARNING: Insufficient GPU memory. Consider using CPU or smaller batch sizes.")
    
    # Test with memory optimizations
    print("\n" + "="*50)
    print("Testing with 8-bit quantization and memory optimizations...")
    success = test_memory_optimized_inference()
    
    # Summary
    print("\n" + "="*50)
    print("MEMORY-OPTIMIZED TEST SUMMARY")
    print("="*50)
    print(f"Quantized inference: {'✅ PASSED' if success else '❌ FAILED'}")
    print("="*50)
    
    if success:
        print("\n✅ GPT-OSS 20B is ready for fine-tuning!")
        print("You can now run: python finetune.py --dataset_path sample_training_data.json")
    else:
        print("\n❌ Consider using a machine with more GPU memory or try CPU inference.")

if __name__ == "__main__":
    main()