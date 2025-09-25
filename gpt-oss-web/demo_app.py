#!/usr/bin/env python3
"""
GPT-OSS 20B Demo Web Interface
Lightweight demo version that shows the web interface without loading the full model
Perfect for testing the interface before committing GPU memory
"""

import gradio as gr
import time
import random

class GPTOSSDemoInterface:
    def __init__(self):
        self.model_loaded = False
        self.demo_responses = [
            "I'm a demo response! The actual GPT-OSS 20B model would provide much more sophisticated answers.",
            "This is just a demonstration. When the real model is loaded, you'll get powerful AI responses here!",
            "Demo mode active! The interface is working perfectly. Load the actual model for real AI conversations.",
            "Interface test successful! This shows how the chat will work with the real GPT-OSS 20B model.",
            "Everything is working great! The web interface is ready for the actual model when you load it."
        ]
    
    def generate_demo_response(self, message, history, max_tokens, temperature, top_p):
        """Generate a demo response to show interface functionality"""
        if not message.strip():
            yield "Please enter a message to test the interface!"
            return
        
        # Simulate thinking time
        time.sleep(0.5)
        
        # Choose a demo response
        base_response = random.choice(self.demo_responses)
        
        # Add some context about the user's message
        response = f"You said: '{message}'\n\n{base_response}\n\nSettings used: Max Tokens={max_tokens}, Temperature={temperature}, Top-P={top_p}"
        
        # Stream the response word by word for realistic effect
        words = response.split()
        current_response = ""
        for word in words:
            current_response += word + " "
            yield current_response.strip()
            time.sleep(0.03)  # Small delay for streaming effect
    
    def get_system_info(self):
        """Get demo system information"""
        info = []
        info.append(f"🤖 **Model**: GPT-OSS 20B (Demo Mode)")
        info.append(f"📍 **Status**: Interface Ready ✅")
        info.append(f"🔧 **Mode**: Demo (not loading actual model)")
        info.append(f"💾 **Purpose**: Test web interface functionality")
        info.append(f"🚀 **Ready**: Yes! Interface is fully functional")
        
        return "\n".join(info)

# Initialize the demo interface
demo_interface = GPTOSSDemoInterface()

def create_demo_interface():
    """Create the demo Gradio interface"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="GPT-OSS 20B Demo Interface",
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .demo-banner {
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            color: white;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="demo-banner">
            <h2>🎭 DEMO MODE - GPT-OSS 20B Web Interface</h2>
            <p>This is a demonstration of the web interface. The actual AI model is not loaded to save GPU memory.</p>
        </div>
        """)
        
        gr.Markdown("""
        # 🤖 GPT-OSS 20B Web Interface (Demo)
        
        This demo shows you exactly how the web interface works! The chat, buttons, and settings all function normally.
        When you're ready to use the real model, simply run `gradio_app.py` instead of this demo version.
        
        **Features being demonstrated:**
        - ✅ Chat interface with streaming responses
        - ✅ Parameter controls (temperature, top-p, max tokens)  
        - ✅ System information display
        - ✅ Example prompts and chat management
        - ✅ All buttons and interface elements
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    type="messages",
                    show_label=False,
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here to test the interface...",
                        show_label=False,
                        scale=4,
                        container=False
                    )
                    send_btn = gr.Button("📤 Send", scale=1, variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", scale=1)
                    retry_btn = gr.Button("🔄 Retry Last", scale=1)
                    demo_info_btn = gr.Button("ℹ️ Demo Info", scale=1, variant="secondary")
            
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
                    info="Controls creativity"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P",
                    info="Nucleus sampling"
                )
                
                gr.Markdown("## 📊 System Info")
                system_info = gr.Markdown(demo_interface.get_system_info())
                
                refresh_btn = gr.Button("🔄 Refresh Info", size="sm")
                
                gr.Markdown("""
                ## 🚀 Next Steps
                
                **To use the real model:**
                1. Stop this demo
                2. Free up GPU memory
                3. Run: `python gradio_app.py`
                
                **Current Status:**
                - Interface: ✅ Working
                - Model: 🎭 Demo mode
                - GPU Memory: 💾 Preserved
                """)
        
        # Example prompts
        gr.Markdown("### 💡 Test these interface features:")
        examples = gr.Examples(
            examples=[
                ["Test the streaming response feature"],
                ["Check how parameter controls work"],
                ["See how the chat interface handles longer messages"],
                ["Verify the retry and clear functions work"],
                ["Demonstrate the example prompt system"]
            ],
            inputs=msg
        )
        
        # Event handlers
        def user_message(message, history):
            return "", history + [{"role": "user", "content": message}]
        
        def bot_response(history, max_tokens, temperature, top_p):
            if not history:
                return history
            
            message = history[-1]["content"] if history[-1]["role"] == "user" else ""
            if not message:
                return history
            
            # Add empty assistant message
            history.append({"role": "assistant", "content": ""})
            
            for partial_response in demo_interface.generate_demo_response(
                message, history[:-1], max_tokens, temperature, top_p
            ):
                history[-1]["content"] = partial_response
                yield history
        
        def clear_chat():
            return []
        
        def show_demo_info():
            return [{"role": "assistant", "content": """🎭 **Demo Mode Information**

This is a fully functional demonstration of the GPT-OSS 20B web interface! Here's what's working:

✅ **Streaming Responses**: Messages appear word-by-word like the real model
✅ **Parameter Controls**: Temperature, Top-P, and Max Tokens all affect demo responses  
✅ **Chat Management**: Clear chat, retry last message, message history
✅ **Example Prompts**: Click any example to test it
✅ **System Info**: Live status and configuration display
✅ **Responsive Design**: Works on desktop and mobile

**Why demo mode?**
- Saves GPU memory (20GB) for when you really need it
- Lets you test the interface without waiting for model loading
- Perfect for development and testing

**To use the real GPT-OSS 20B model:**
1. Make sure you have 20GB+ free GPU memory
2. Close this demo interface  
3. Run: `python gradio_app.py`
4. Wait for model loading (2-5 minutes)
5. Enjoy powerful AI conversations!

The interface will look and work exactly like this demo, but with real AI responses! 🤖"""}]
        
        def retry_last(history, max_tokens, temperature, top_p):
            if not history:
                return history
            
            # Remove the last assistant message and regenerate
            if history and history[-1]["role"] == "assistant":
                # Find the last user message
                user_message = None
                for i in range(len(history) - 2, -1, -1):
                    if history[i]["role"] == "user":
                        user_message = history[i]["content"]
                        break
                
                # Remove the last assistant message
                history = history[:-1]
                
                if user_message:
                    # Add empty assistant message
                    history.append({"role": "assistant", "content": ""})
                    
                    # Generate new response
                    for partial_response in demo_interface.generate_demo_response(
                        user_message, history[:-1], max_tokens, temperature, top_p
                    ):
                        history[-1]["content"] = partial_response
                        yield history
            
            return history
        
        def refresh_system_info():
            return demo_interface.get_system_info()
        
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
        
        demo_info_btn.click(lambda: show_demo_info(), None, chatbot, queue=False)
        
        retry_btn.click(
            retry_last,
            [chatbot, max_tokens, temperature, top_p],
            chatbot
        )
        
        refresh_btn.click(refresh_system_info, None, system_info)
        
    return demo

if __name__ == "__main__":
    print("🎭 Starting GPT-OSS Demo Interface...")
    print("This demo shows the web interface without loading the actual model")
    print("Perfect for testing before committing 20GB of GPU memory!")
    
    demo = create_demo_interface()
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )