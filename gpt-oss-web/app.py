#!/usr/bin/env python3
"""
GPT-OSS 20B Web Interface using Flask
Simple web interface to interact with the fine-tuned GPT-OSS model
"""

import sys
import os
import json
import torch
import gc
from flask import Flask, render_template, request, jsonify, stream_template
from flask_cors import CORS
import threading
import queue
import time
from pathlib import Path

# Add the parent directory to the path to access gpt-oss modules
sys.path.append(str(Path(__file__).parent.parent / "gpt-oss"))

app = Flask(__name__)
CORS(app)

# Global variables
model = None
tokenizer = None
model_lock = threading.Lock()

class GPTOSSWebInterface:
    def __init__(self, model_path="../gpt-oss/gpt-oss-20b"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the GPT-OSS model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            print(f"Loading model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Setup quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            ) if torch.cuda.is_available() else None
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                max_memory={0: "20GB", "cpu": "30GB"} if torch.cuda.is_available() else None
            )
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            # Fallback to unquantized model if quantization fails
            print(f"Quantized loading failed: {e}")
            print("Falling back to standard loading...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                print("✅ Model loaded successfully (standard loading)")
            except Exception as e2:
                print(f"❌ Failed to load model: {e2}")
                self.model = None
    
    def generate_response(self, user_input, max_length=256, temperature=1.0, top_p=0.9):
        """Generate a response from the model"""
        if not self.model or not self.tokenizer:
            return "❌ Model not loaded. Please check the setup."
        
        try:
            # Format the input as a conversation
            messages = [
                {"role": "user", "content": user_input}
            ]
            
            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode the response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up GPU memory
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return f"❌ Error generating response: {str(e)}"

# Initialize the interface
gpt_interface = GPTOSSWebInterface()

@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    """API endpoint for generating responses"""
    try:
        data = request.get_json()
        user_input = data.get('message', '')
        max_length = data.get('max_length', 256)
        temperature = data.get('temperature', 1.0)
        top_p = data.get('top_p', 0.9)
        
        if not user_input.strip():
            return jsonify({'error': 'Empty message'}), 400
        
        with model_lock:
            response = gpt_interface.generate_response(
                user_input, max_length, temperature, top_p
            )
        
        return jsonify({
            'response': response,
            'model_loaded': gpt_interface.model is not None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """Get system status"""
    return jsonify({
        'model_loaded': gpt_interface.model is not None,
        'device': gpt_interface.device,
        'cuda_available': torch.cuda.is_available(),
        'gpu_memory': torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0
    })

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')

if __name__ == '__main__':
    print("🚀 Starting GPT-OSS Web Interface...")
    print(f"Model status: {'✅ Loaded' if gpt_interface.model else '❌ Not loaded'}")
    print("🌐 Web interface will be available at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)