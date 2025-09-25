#!/usr/bin/env python3
"""
GPT-OSS 20B Streamlit Interface
Clean and modern web interface using Streamlit
"""

import sys
import os
import torch
import streamlit as st
from pathlib import Path
import time

# Add the parent directory to access gpt-oss modules
sys.path.append(str(Path(__file__).parent.parent / "gpt-oss"))

# Page config
st.set_page_config(
    page_title="GPT-OSS 20B Interface",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the GPT-OSS model and tokenizer (cached)"""
    model_path = "../gpt-oss/gpt-oss-20b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        with st.spinner("Loading GPT-OSS 20B model... This may take a few minutes."):
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Try quantized loading first
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                ) if torch.cuda.is_available() else None
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    max_memory={0: "20GB", "cpu": "30GB"} if torch.cuda.is_available() else None
                )
                st.success("✅ Model loaded with 8-bit quantization")
                
            except Exception as e:
                st.warning(f"Quantized loading failed, trying standard loading: {e}")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                st.success("✅ Model loaded with standard loading")
            
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None, device

def generate_response(model, tokenizer, device, message, max_tokens=256, temperature=1.0, top_p=0.9):
    """Generate response from the model"""
    if not model or not tokenizer:
        return "❌ Model not loaded. Please check the setup."
    
    if not message.strip():
        return "Please enter a message."
    
    try:
        # Format as conversation
        messages = [{"role": "user", "content": message}]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:], 
            skip_special_tokens=True
        )
        
        # Clean up
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()
        
    except Exception as e:
        error_msg = f"❌ Error generating response: {str(e)}"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return error_msg

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🤖 GPT-OSS 20B Web Interface")
    st.markdown("Welcome to your personal GPT-OSS 20B assistant! A powerful 21B parameter model optimized for reasoning and conversation.")
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=10,
            max_value=1024,
            value=256,
            step=1,
            help="Maximum response length"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Controls creativity (lower = more focused)"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling threshold"
        )
        
        st.markdown("---")
        st.header("📊 System Info")
        
        # System information
        info_lines = []
        info_lines.append(f"🤖 **Model**: GPT-OSS 20B")
        info_lines.append(f"📍 **Device**: {device.upper()}")
        info_lines.append(f"🔧 **Model Status**: {'✅ Loaded' if model else '❌ Not loaded'}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            info_lines.append(f"🎮 **GPU**: {gpu_name}")
            info_lines.append(f"💾 **GPU Memory**: {gpu_memory}GB")
        
        for line in info_lines:
            st.markdown(line)
        
        if st.button("🔄 Refresh System Info"):
            st.rerun()
        
        st.markdown("---")
        st.header("💡 Example Prompts")
        
        examples = [
            "What is machine learning and how does it work?",
            "Write a Python function to calculate fibonacci numbers",
            "Explain quantum computing in simple terms",
            "Create a short story about a robot learning to paint",
            "What are the pros and cons of renewable energy?"
        ]
        
        for example in examples:
            if st.button(f"📝 {example[:30]}...", key=example, use_container_width=True):
                st.session_state.user_input = example
                st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    # Get user input
    if prompt := st.chat_input("Type your message here...", key="chat_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    model, tokenizer, device, prompt, 
                    max_tokens, temperature, top_p
                )
            
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Handle example button clicks
    if st.session_state.user_input and st.session_state.user_input not in [msg["content"] for msg in st.session_state.messages[-2:]]:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
        with st.chat_message("user"):
            st.markdown(st.session_state.user_input)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(
                    model, tokenizer, device, st.session_state.user_input, 
                    max_tokens, temperature, top_p
                )
            
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear the input
        st.session_state.user_input = ""
    
    # Control buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Retry Last", use_container_width=True) and st.session_state.messages:
            # Remove last assistant message and regenerate
            if st.session_state.messages[-1]["role"] == "assistant":
                last_user_msg = None
                if len(st.session_state.messages) >= 2:
                    last_user_msg = st.session_state.messages[-2]["content"]
                
                # Remove the last assistant message
                st.session_state.messages.pop()
                
                if last_user_msg:
                    # Generate new response
                    with st.spinner("Regenerating..."):
                        response = generate_response(
                            model, tokenizer, device, last_user_msg, 
                            max_tokens, temperature, top_p
                        )
                    
                    # Add new response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()

if __name__ == "__main__":
    main()