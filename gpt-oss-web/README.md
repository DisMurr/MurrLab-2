# gpt-oss-web

Local web UI for GPT-OSS 20B with two modes:
- Backend API (recommended): Calls the Responses API served by gpt-oss backend.
- Local Transformers: Loads the model in the UI process with offload options.

## Quickstart

1) Start backend server (recommended)

```bash
# from gpt-oss-web folder
chmod +x run_backend.sh
./run_backend.sh
```

Optional environment variables:
- CHECKPOINT: path to model (default: ../gpt-oss/gpt-oss-20b)
- PORT: server port (default: 8000)
- GPU_MEM_CAP: e.g., 18GiB
- CPU_MEM_CAP: e.g., 48GiB
- OFFLOAD_DIR: disk offload folder (default: ./offload)

2) Launch frontend UI

```bash
# from gpt-oss-web folder
python fixed_app.py
```

In the UI, keep Mode = Backend API and Backend URL = http://127.0.0.1:8000, then click Load Model.

If you prefer to load locally, switch Mode to Local Transformers, set GPU VRAM cap and/or GPU-only, then click Load Model.

## Notes
- GPU-only requires substantial free VRAM. If you hit OOM, use the offload mode (default) or reduce tokens.
- The backend uses memory-safe defaults for Transformers, including device_map="auto" and offload.
- Installing Triton with MXFP4 kernels can avoid dequantize-to-bf16 spikes.# GPT-OSS 20B Web Interface 🤖

This directory contains multiple web interfaces for interacting with your GPT-OSS 20B model through a browser. Choose the interface that best fits your needs!

## 🚀 Quick Start

1. **Make sure the GPT-OSS model is set up** (should be in `../gpt-oss/gpt-oss-20b/`)
2. **Run the launcher script**:
   ```bash
   ./launch.sh
   ```
3. **Choose your preferred interface** from the menu
4. **Open your browser** to the provided URL

## 📱 Available Interfaces

### 🎨 Gradio Interface (RECOMMENDED)
**File**: `gradio_app.py` | **Port**: 7860 | **URL**: http://localhost:7860

**Features**:
- ✨ Beautiful, modern chat interface
- 🔄 Real-time streaming responses
- ⚙️ Easy parameter controls (temperature, top-p, max tokens)
- 💡 Built-in example prompts
- 📊 System information display
- 🎯 Perfect for general use and experimentation

**Best for**: Most users - it's simple, elegant, and full-featured.

### 🌐 Flask Interface
**File**: `app.py` | **Port**: 5000 | **URL**: http://localhost:5000

**Features**:
- 🏗️ Full REST API with JSON endpoints
- 🎨 Custom HTML/CSS/JavaScript frontend
- 🔗 CORS enabled for external integrations
- ⚡ Fast response times
- 🔧 Highly customizable

**Best for**: Developers who need API access or want to customize the interface.

**API Endpoints**:
- `GET /` - Main chat interface
- `POST /api/chat` - Send message and get response
- `GET /api/status` - Check model status
- `GET /settings` - Configuration page

### ✨ Streamlit Interface
**File**: `streamlit_app.py` | **Port**: 8501 | **URL**: http://localhost:8501

**Features**:
- 💎 Clean, professional interface
- 📊 Sidebar with controls and system info
- 🔄 Real-time parameter adjustment
- 💡 Clickable example prompts
- 🎪 Modern chat experience

**Best for**: Professional demos, presentations, and users who prefer Streamlit's aesthetic.

## 🛠️ Manual Setup

If you prefer to run interfaces manually:

### Prerequisites
```bash
# Make sure you're in the project directory and virtual environment is activated
source ../gpt-oss/.venv/bin/activate

# Install web interface dependencies (should already be installed)
pip install flask flask-cors gradio streamlit
```

### Running Individual Interfaces

**Gradio** (Recommended):
```bash
python gradio_app.py
```

**Flask**:
```bash
python app.py
```

**Streamlit**:
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## ⚙️ Configuration Options

All interfaces support these parameters:

- **Max Tokens** (10-1024): Maximum length of the response
- **Temperature** (0.1-2.0): Controls creativity (lower = more focused)
- **Top P** (0.1-1.0): Nucleus sampling threshold

## 🔧 Troubleshooting

### Model Not Loading
- ✅ Check that the model exists at `../gpt-oss/gpt-oss-20b/`
- ✅ Ensure you have enough GPU memory (20GB+ recommended)
- ✅ Try the 8-bit quantized loading (automatic fallback)

### Memory Issues
- 🔄 Restart the interface to clear GPU memory
- ⚡ Use lower max_tokens values
- 💾 Close other GPU-intensive applications

### Port Already in Use
```bash
# Find and kill process using the port (example for port 7860)
sudo lsof -ti:7860 | xargs sudo kill -9
```

### Permission Errors
```bash
# Make sure the launch script is executable
chmod +x launch.sh
```

## 🎯 Performance Tips

1. **GPU Memory**: The model works best with 20GB+ GPU memory
2. **Quantization**: 8-bit quantization is automatically tried first
3. **Batch Size**: Keep conversations reasonably short for best performance
4. **Temperature**: Use 0.7-1.0 for balanced responses

## 🚀 Advanced Usage

### Custom Model Path
Edit the model path in any interface file:
```python
model_path = "path/to/your/model"
```

### API Integration (Flask)
Use the Flask interface's REST API:
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, GPT-OSS!", "max_tokens": 256}'
```

### Network Access
To allow access from other devices on your network, the interfaces are configured to bind to `0.0.0.0`. Access via:
- `http://YOUR_IP_ADDRESS:PORT`

## 📁 File Structure

```
gpt-oss-web/
├── launch.sh           # Interactive launcher script
├── app.py             # Flask interface
├── gradio_app.py      # Gradio interface (recommended)
├── streamlit_app.py   # Streamlit interface
├── templates/         # HTML templates for Flask
│   ├── index.html     # Main chat page
│   └── settings.html  # Settings page
└── README.md          # This file
```

## 🆘 Getting Help

1. **Check the terminal output** for error messages
2. **Verify system requirements**: GPU memory, Python version, dependencies
3. **Try different interfaces** if one doesn't work
4. **Restart the interface** to clear any memory issues

## 🎉 Enjoy Your GPT-OSS Experience!

You now have three different ways to interact with your powerful GPT-OSS 20B model. Each interface has its strengths:

- **Just want to chat?** → Use Gradio
- **Need API access?** → Use Flask  
- **Want something sleek?** → Use Streamlit

Happy chatting! 🤖✨