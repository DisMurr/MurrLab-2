#!/usr/bin/env python3
"""
Simple GPT-OSS 20B Gradio Interface - Minimal working version
Back to basics approach that definitely works
"""

import sys
import os
import torch
import gradio as gr
from pathlib import Path
import time

# Add the parent directory to access gpt-oss modules
sys.path.append(str(Path(__file__).parent.parent / "gpt-oss"))

def load_model_simple():
    """Simple model loading - exactly like the working test"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = "../gpt-oss/gpt-oss-20b"
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Simple loading - exactly like what worked before
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def chat_simple(message, history, model, tokenizer):
    """Simple chat function - exactly like the working test"""
    if not model or not tokenizer:
        return "❌ Model not loaded"
    
    if not message.strip():
        return "Please enter a message."
    
    try:
        # Simple generation - like the test that worked
        inputs = tokenizer(message, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Clean up
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return response.strip()
        
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"❌ Error: {str(e)}"

# Load model once
print("🚀 Starting Simple GPT-OSS Interface...")
model, tokenizer = load_model_simple()

def create_simple_interface():
    """Create the simplest possible working interface"""
    
    def respond(message, history):
        if not message.strip():
            return history, ""
        
        # Get bot response
        bot_message = chat_simple(message, history, model, tokenizer)
        
        # Add to history
        history.append([message, bot_message])
        return history, ""
    
    with gr.Blocks(title="GPT-OSS 20B Simple") as demo:
        gr.Markdown("# 🤖 GPT-OSS 20B Simple Interface")
        
        if model:
            gr.Markdown("✅ **Model Status**: Loaded and ready!")
        else:
            gr.Markdown("❌ **Model Status**: Not loaded")
        
        chatbot = gr.Chatbot(label="Chat with GPT-OSS 20B")
        msg = gr.Textbox(label="Your message", placeholder="Type your message here...")
        clear = gr.Button("Clear Chat")
        
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: [], None, chatbot, queue=False)
        
    return demo

if __name__ == "__main__":
    demo = create_simple_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )