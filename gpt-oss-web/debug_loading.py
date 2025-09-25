#!/usr/bin/env python3
"""
Simple model loading test to debug the issue
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

def test_model_loading():
    print("🧪 Testing GPT-OSS model loading step by step...")
    
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
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        return False
    
    # Step 4: Setup quantization
    try:
        print("⚙️ Setting up 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        print("✅ Quantization config ready")
    except Exception as e:
        print(f"❌ Quantization setup failed: {e}")
        return False
    
    # Step 5: Load model
    try:
        print("🤖 Loading model (this takes 2-5 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            "../gpt-oss/gpt-oss-20b",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            max_memory={0: "20GB", "cpu": "30GB"}
        )
        print("✅ Model loaded successfully!")
        
        # Step 6: Quick test
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
    success = test_model_loading()
    if success:
        print("\n🎉 All tests passed! The model should work in the web interface.")
    else:
        print("\n❌ Tests failed. There's an issue with the model loading.")