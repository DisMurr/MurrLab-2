# GPT-OSS 20B Quick Usage Guide

## 🚀 Your Complete GPT-OSS Setup is Ready!

You now have a fully functional GPT-OSS 20B installation with multiple web interfaces and fine-tuning capabilities.

## 📁 What You Have

- **GPT-OSS 20B Model**: 39GB fully downloaded and ready
- **Python Environment**: Virtual environment with all dependencies  
- **Web Interfaces**: 4 different ways to interact with the model
- **Fine-tuning Scripts**: Ready for LoRA fine-tuning
- **Demo Mode**: Test interfaces without loading the model

## 🎯 Quick Start Options

### Option 1: Demo Interface (Recommended First Step)
```bash
cd /home/murr/Project/gpt-oss-web
python demo_app.py
```
- **Purpose**: Test the web interface without using GPU memory
- **URL**: http://localhost:7860
- **Memory**: Uses minimal resources
- **Perfect for**: Seeing how everything works before committing GPU memory

### Option 2: Full Web Interface
```bash
cd /home/murr/Project/gpt-oss-web
./launch.sh
```
- **Purpose**: Interactive menu to choose your preferred interface
- **Options**: Gradio (recommended), Flask, or Streamlit
- **Memory**: Requires ~20GB GPU memory

### Option 3: Direct Model Testing
```bash
cd /home/murr/Project/gpt-oss
source ../.venv/bin/activate
python test_model.py
```

### Option 4: Fine-tuning
```bash
cd /home/murr/Project/gpt-oss
source ../.venv/bin/activate
python finetune.py
```

## 💡 Recommended Workflow

1. **Start with Demo**: `cd gpt-oss-web && python demo_app.py`
   - See the web interface in action
   - Test all features without GPU memory usage

2. **Check GPU Memory**: `nvidia-smi`
   - Make sure you have ~20GB free GPU memory
   - Close other GPU applications if needed

3. **Load Real Model**: `cd gpt-oss-web && ./launch.sh`
   - Choose Gradio interface (option 2)
   - Wait 2-5 minutes for model loading
   - Start chatting with the real AI!

4. **Fine-tune (Optional)**: `cd gpt-oss && python finetune.py`
   - Use your own data to customize the model

## 🌐 Web Interface Comparison

| Interface | Port | Best For | Features |
|-----------|------|----------|----------|
| **Gradio** | 7860 | General use | Streaming, examples, easy controls ⭐ |
| **Flask** | 5000 | Development | REST API, customizable, CORS |
| **Streamlit** | 8501 | Presentations | Professional UI, sidebar controls |
| **Demo** | 7860 | Testing | No model loading, shows interface |

## 🔧 Troubleshooting

**GPU Memory Issues:**
```bash
# Check memory usage
nvidia-smi

# Free up memory
pkill -f python

# Use memory optimization
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python gradio_app.py
```

**Model Loading Problems:**
- Make sure you have 20GB+ free GPU memory
- Try the demo interface first to test everything else
- Close other applications using GPU memory
- The model will fallback to CPU if needed (slower but works)

## 📊 System Requirements Met

✅ **GPU**: NVIDIA RTX 4090 (22GB) - Perfect for this model!  
✅ **Model**: 21B parameters, MXFP4 quantized  
✅ **Memory**: ~20GB GPU memory needed for full model  
✅ **Environment**: Python 3.12 virtual environment  
✅ **Dependencies**: PyTorch, Transformers, PEFT, web frameworks  

## 🎉 You're All Set!

Your GPT-OSS 20B setup is complete and ready to use. Start with the demo interface to see everything in action, then move to the full model when you're ready to experience the powerful AI capabilities.

**Happy AI chatting!** 🤖✨