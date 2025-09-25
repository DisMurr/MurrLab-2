#!/bin/bash

# GPT-OSS 20B Web Interface Launcher
# This script helps you launch different web interfaces for the GPT-OSS model

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
VENV_PATH="$PROJECT_DIR/.venv"

echo -e "${BLUE}🚀 GPT-OSS 20B Web Interface Launcher${NC}"
echo -e "${BLUE}======================================${NC}"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${YELLOW}Please run the setup script first or activate your Python environment${NC}"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check if model exists
MODEL_PATH="$PROJECT_DIR/gpt-oss/gpt-oss-20b"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}❌ GPT-OSS model not found at $MODEL_PATH${NC}"
    echo -e "${YELLOW}Please download and set up the model first${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo -e "${GREEN}✅ Model found at $MODEL_PATH${NC}"
echo

# Function to show menu
show_menu() {
    echo -e "${BLUE}Choose a web interface to launch:${NC}"
    echo -e "${YELLOW}1) Flask Interface${NC}     - Full-featured with REST API"
    echo -e "${YELLOW}2) Gradio Interface${NC}    - Simple and elegant (recommended)"
    echo -e "${YELLOW}3) Streamlit Interface${NC} - Modern and interactive"
    echo -e "${YELLOW}4) All interfaces info${NC}  - Show details about each option"
    echo -e "${YELLOW}5) Exit${NC}"
    echo
}

# Function to show interface information
show_info() {
    echo -e "${BLUE}📱 Interface Comparison:${NC}"
    echo
    echo -e "${YELLOW}Flask Interface:${NC}"
    echo "  • Full REST API with JSON responses"
    echo "  • Custom HTML/CSS/JavaScript frontend"
    echo "  • CORS enabled for external integration"
    echo "  • Port: 5000"
    echo "  • Best for: Custom integrations and development"
    echo
    echo -e "${YELLOW}Gradio Interface:${NC}"
    echo "  • Built-in chat interface with streaming responses"
    echo "  • Easy-to-use sliders for model parameters"
    echo "  • Example prompts and system information"
    echo "  • Port: 7860"
    echo "  • Best for: General use and experimentation (RECOMMENDED)"
    echo
    echo -e "${YELLOW}Streamlit Interface:${NC}"
    echo "  • Modern chat interface with sidebar controls"
    echo "  • Real-time parameter adjustment"
    echo "  • Built-in example prompts"
    echo "  • Port: 8501"
    echo "  • Best for: Clean UI and professional demos"
    echo
}

# Function to launch Flask interface
launch_flask() {
    echo -e "${GREEN}🌐 Starting Flask interface...${NC}"
    echo -e "${BLUE}Access at: http://localhost:5000${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo
    cd "$SCRIPT_DIR"
    python app.py
}

# Function to launch Gradio interface
launch_gradio() {
    echo -e "${GREEN}🎨 Starting Gradio interface...${NC}"
    echo -e "${BLUE}Access at: http://localhost:7860${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo
    cd "$SCRIPT_DIR"
    python gradio_app.py
}

# Function to launch Streamlit interface
launch_streamlit() {
    echo -e "${GREEN}✨ Starting Streamlit interface...${NC}"
    echo -e "${BLUE}Access at: http://localhost:8501${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo
    cd "$SCRIPT_DIR"
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
}

# Main menu loop
while true; do
    show_menu
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            launch_flask
            ;;
        2)
            launch_gradio
            ;;
        3)
            launch_streamlit
            ;;
        4)
            show_info
            echo
            read -p "Press Enter to continue..."
            echo
            ;;
        5)
            echo -e "${GREEN}👋 Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Invalid option. Please choose 1-5.${NC}"
            echo
            ;;
    esac
done