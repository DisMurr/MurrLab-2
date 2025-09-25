#!/usr/bin/env python3
"""
GPT-OSS 20B Working Interface
Simple, reliable version that loads the model only when needed
"""

import sys
import os
import torch
import gradio as gr
from pathlib import Path
import gc

# Add the parent directory to access gpt-oss modules
sys.path.append(str(Path(__file__).parent.parent / "gpt-oss"))

class WorkingGPTInterface:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = "../gpt-oss/gpt-oss-20b"
        self.loading = False
    
    def clear_gpu_cache(self):
        """Clear GPU cache thoroughly"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    def load_model_now(self):
        """Load model when user requests it"""
        if self.loading:
            return "⏳ Model is already loading..."
        
        if self.model is not None:
            return "✅ Model is already loaded and ready!"
        
        self.loading = True
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Clear any existing cache first
            self.clear_gpu_cache()
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with minimal memory usage
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            self.loading = False
            return "✅ Model loaded successfully! You can now start chatting."
            
        except Exception as e:
            self.loading = False
            error_msg = f"❌ Failed to load model: {str(e)}"
            self.clear_gpu_cache()
            return error_msg
    
    def generate_response(self, message, max_tokens=256, temperature=1.0, top_p=0.9):
        """Generate response from the loaded model"""
        if self.model is None:
            return "❌ Model not loaded. Click 'Load Model' first."
        
        if not message.strip():
            return "Please enter a message."
        
        try:
            # Format as conversation
            messages = [{"role": "user", "content": message}]
            
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
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up
            del outputs
            self.clear_gpu_cache()
            
            return response.strip()
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            self.clear_gpu_cache()
            return error_msg
    
    def get_status(self):
        """Get current status"""
        if self.loading:
            return "⏳ Loading model..."
        elif self.model is None:
            return "⭕ Model not loaded"
        else:
            return "✅ Model ready"

# Initialize the interface
gpt_interface = WorkingGPTInterface()

def create_interface():
    """Create the working Gradio interface"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="GPT-OSS 20B Working Interface") as demo:
        
        gr.Markdown("""
        # 🤖 GPT-OSS 20B - Working Interface
        
        This is a reliable interface that loads the model only when you're ready.
        """)
        
        with gr.Row():
            status_display = gr.Textbox(
                value=gpt_interface.get_status(),
                label="Status",
                interactive=False
            )
            load_btn = gr.Button("🔄 Load Model", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Your Message",
                        lines=2
                    )
                
                with gr.Row():
                    send_btn = gr.Button("📤 Send", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear")
                
                response_output = gr.Textbox(
                    label="Response",
                    lines=10,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## Settings")
                
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=512,
                    value=256,
                    label="Max Tokens"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    label="Temperature"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    label="Top P"
                )
        
        # Event handlers
        def load_model_handler():
            result = gpt_interface.load_model_now()
            status = gpt_interface.get_status()
            return result, status
        
        def send_message(message, max_tokens, temperature, top_p):
            if not message.strip():
                return "Please enter a message."
            
            response = gpt_interface.generate_response(
                message, max_tokens, temperature, top_p
            )
            return response
        
        def clear_input():
            return "", ""
        
        def refresh_status():
            return gpt_interface.get_status()
        
        # Wire up events
        load_btn.click(
            load_model_handler,
            outputs=[response_output, status_display]
        )
        
        send_btn.click(
            send_message,
            inputs=[user_input, max_tokens, temperature, top_p],
            outputs=[response_output]
        )
        
        clear_btn.click(
            clear_input,
            outputs=[user_input, response_output]
        )
        
        # Auto-refresh status
        demo.load(refresh_status, outputs=[status_display])
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting GPT-OSS Working Interface...")
    print("This version loads the model only when you click 'Load Model'")
    
    demo = create_interface()
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True,
        show_error=True
    )