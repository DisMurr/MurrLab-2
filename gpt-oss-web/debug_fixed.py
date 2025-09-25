#!/usr/bin/env python3
"""
FIXED: Simple model loading test without conflicting quantization
The model is already MXFP4 quantized, so we load it directly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

def test_model_loading_fixed():
    print("🧪 Testing GPT-OSS model loading (FIXED - no conflicting quantization)...")
    
    # Step 1: Check CUDA
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
    
    # Step 2: Clear memory
    print("🧹 Clearing GPU cache...")
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 3: Load tokenizer
    try:
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("../gpt-oss/gpt-oss-20b")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return False
    
    # Step 4: Load model WITHOUT conflicting quantization
    try:
        print("🤖 Loading model (already MXFP4 quantized, takes 2-5 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            "../gpt-oss/gpt-oss-20b",
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,  # Use dtype instead of torch_dtype
            low_cpu_mem_usage=True,
            max_memory={0: "20GB", "cpu": "30GB"}
        )
        print("✅ Model loaded successfully!")
        
        # Step 5: Quick test
        print("🧪 Testing inference...")
        messages = [{"role": "user", "content": "What is 2+2?"}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.to(model.device),
                max_new_tokens=20,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"🎉 Model response: {response}")
        
        # Cleanup
        del model, outputs
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        torch.cuda.empty_cache()
        return False

if __name__ == "__main__":
    success = test_model_loading_fixed()
    if success:
        print("\n🎉 All tests passed! The model should work in the web interface.")
    else:
        print("\n❌ Tests failed. There's still an issue with the model loading.")