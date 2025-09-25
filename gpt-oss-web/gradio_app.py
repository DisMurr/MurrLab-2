#!/usr/bin/env python3
"""
GPT-OSS 20B Gradio Interface
Simple and elegant web interface using Gradio
"""

import sys
import os
import torch
import gradio as gr
from pathlib import Path
import time
import gc

# Add the parent directory to access gpt-oss modules
sys.path.append(str(Path(__file__).parent.parent / "gpt-oss"))

class GPTOSSGradioInterface:
    def __init__(self, model_path="../gpt-oss/gpt-oss-20b"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load the GPT-OSS model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Simple direct loading - let the model use its native quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_response(self, message, history, max_tokens, temperature, top_p):
        """Generate response for Gradio chat interface"""
        if not self.model or not self.tokenizer:
            yield "❌ Model not loaded. Please check the setup."
            return
        
        if not message.strip():
            yield "Please enter a message."
            return
        
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
            
            # Generate response with streaming
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Stream the response word by word for better UX
            words = response.split()
            current_response = ""
            for word in words:
                current_response += word + " "
                yield current_response.strip()
                time.sleep(0.05)  # Small delay for streaming effect
                
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            print(error_msg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield error_msg
    
    def get_system_info(self):
        """Get system information for display"""
        info = []
        info.append(f"🤖 **Model**: GPT-OSS 20B")
        info.append(f"📍 **Device**: {self.device.upper()}")
        info.append(f"🔧 **Model Loaded**: {'✅ Yes' if self.model else '❌ No'}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            info.append(f"🎮 **GPU**: {gpu_name}")
            info.append(f"💾 **GPU Memory**: {gpu_memory}GB")
        
        return "\n".join(info)

# Initialize the interface
gpt_interface = GPTOSSGradioInterface()

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="GPT-OSS 20B Interface",
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .chat-bubble {
            max-width: 80% !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 GPT-OSS 20B Web Interface
        
        Welcome to your personal GPT-OSS 20B assistant! This is a powerful 21B parameter model optimized for reasoning and conversation.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    show_label=False,
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=4,
                        container=False
                    )
                    send_btn = gr.Button("📤 Send", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
                    retry_btn = gr.Button("🔄 Retry Last", scale=1)
            
            with gr.Column(scale=1):
                # Settings panel
                gr.Markdown("## ⚙️ Settings")
                
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=1024,
                    value=256,
                    step=1,
                    label="Max Tokens",
                    info="Maximum response length"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="Controls creativity (lower = more focused)"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P",
                    info="Nucleus sampling threshold"
                )
                
                gr.Markdown("## 📊 System Info")
                system_info = gr.Markdown(gpt_interface.get_system_info())
                
                refresh_btn = gr.Button("🔄 Refresh Info", size="sm")
        
        # Example prompts
        gr.Markdown("### 💡 Try these examples:")
        examples = gr.Examples(
            examples=[
                ["What is machine learning and how does it work?"],
                ["Write a Python function to calculate fibonacci numbers"],
                ["Explain quantum computing in simple terms"],
                ["Create a short story about a robot learning to paint"],
                ["What are the pros and cons of renewable energy?"]
            ],
            inputs=msg
        )
        
        # Event handlers
        def user_message(message, history):
            return "", history + [[message, None]]
        
        def bot_response(history, max_tokens, temperature, top_p):
            if not history:
                return history
            
            message = history[-1][0]
            history[-1][1] = ""
            
            for partial_response in gpt_interface.generate_response(
                message, history, max_tokens, temperature, top_p
            ):
                history[-1][1] = partial_response
                yield history
        
        def clear_chat():
            return []
        
        def retry_last(history, max_tokens, temperature, top_p):
            if not history:
                return history
            
            # Remove the last response and regenerate
            if history and history[-1][1] is not None:
                history[-1][1] = ""
                message = history[-1][0]
                for partial_response in gpt_interface.generate_response(
                    message, history, max_tokens, temperature, top_p
                ):
                    history[-1][1] = partial_response
                    yield history
            return history
        
        def refresh_system_info():
            return gpt_interface.get_system_info()
        
        # Wire up the interface
        msg.submit(
            user_message, 
            [msg, chatbot], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot_response, 
            [chatbot, max_tokens, temperature, top_p], 
            chatbot
        )
        
        send_btn.click(
            user_message, 
            [msg, chatbot], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot_response, 
            [chatbot, max_tokens, temperature, top_p], 
            chatbot
        )
        
        clear_btn.click(clear_chat, None, chatbot, queue=False)
        
        retry_btn.click(
            retry_last,
            [chatbot, max_tokens, temperature, top_p],
            chatbot
        )
        
        refresh_btn.click(refresh_system_info, None, system_info)
        
    return demo

if __name__ == "__main__":
    print("🚀 Starting GPT-OSS Gradio Interface...")
    print(f"Model status: {'✅ Loaded' if gpt_interface.model else '❌ Not loaded'}")
    
    demo = create_interface()
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )