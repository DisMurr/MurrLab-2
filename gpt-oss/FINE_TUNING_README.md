# GPT-OSS 20B Fine-tuning Setup

This directory contains a complete setup for fine-tuning OpenAI's GPT-OSS 20B model with all necessary tools and scripts.

## 🚀 Quick Start

### 1. Installation Verification
```bash
# Test the installation
python test_inference.py
```

### 2. Create Sample Training Data
```bash
# Generate a sample dataset
python finetune.py --create_sample
```

### 3. Fine-tune the Model
```bash
# Basic fine-tuning with LoRA
python finetune.py --dataset_path sample_training_data.json --epochs 3

# Advanced fine-tuning with custom parameters
python finetune.py \
  --dataset_path your_data.json \
  --output_dir custom-model \
  --epochs 5 \
  --learning_rate 2e-4 \
  --batch_size 2 \
  --use_wandb
```

## 📁 Directory Structure

```
gpt-oss/
├── gpt-oss-20b/              # Downloaded model files
│   ├── original/             # Original model weights
│   ├── model-*.safetensors   # Transformers-compatible weights
│   ├── config.json           # Model configuration
│   └── tokenizer.json        # Tokenizer
├── finetune.py              # Main fine-tuning script
├── test_inference.py        # Installation test script
├── FINE_TUNING_README.md    # This file
└── sample_training_data.json # Sample dataset (created when needed)
```

## 🛠 Scripts Overview

### `test_inference.py`
- Tests the GPT-OSS installation
- Verifies model loading and basic inference
- Supports both native GPT-OSS and Transformers implementations

### `finetune.py`
- Complete fine-tuning pipeline
- LoRA (Low-Rank Adaptation) support for efficient training
- Automatic dataset formatting for harmony response format
- Weights & Biases integration
- Customizable training parameters

## 📊 Model Information

- **Model**: GPT-OSS 20B
- **Parameters**: 21B total (3.6B active)
- **Memory**: ~16GB GPU memory required
- **Format**: MXFP4 quantized weights
- **License**: Apache 2.0

## 🔧 Training Configuration

### Default Parameters
- **Epochs**: 3
- **Learning Rate**: 1e-4
- **Batch Size**: 1
- **Gradient Accumulation**: 8 steps
- **LoRA Rank**: 16
- **LoRA Alpha**: 32

### Hardware Requirements
- **GPU**: 16GB+ VRAM (RTX 4090, A100, H100)
- **RAM**: 32GB+ recommended
- **Storage**: ~50GB for model + training data

## 📚 Usage Examples

### Basic Fine-tuning
```python
from finetune import GPTOSSFineTuner

# Initialize fine-tuner
fine_tuner = GPTOSSFineTuner(
    model_path="gpt-oss-20b",
    output_dir="my-finetuned-model",
    use_lora=True
)

# Load model
fine_tuner.load_model_and_tokenizer()

# Prepare dataset
dataset = fine_tuner.prepare_dataset(dataset_path="my_data.json")

# Train
trainer = fine_tuner.train(dataset)
```

### Custom Data Format
Your training data should be in JSON format:
```json
[
  {
    "text": "What is machine learning? Machine learning is a method of data analysis that automates analytical model building."
  },
  {
    "text": "Explain neural networks. Neural networks are computing systems inspired by biological neural networks."
  }
]
```

### Using Hugging Face Datasets
```bash
python finetune.py --dataset_name "squad" --epochs 2
```

## 🎯 Fine-tuning Tips

### Memory Optimization
1. **Use LoRA**: Reduces trainable parameters by ~99%
2. **Gradient Checkpointing**: Trades computation for memory
3. **Mixed Precision**: Use bfloat16 for better efficiency
4. **Smaller Batch Sizes**: Use gradient accumulation instead

### Training Best Practices
1. **Data Quality**: High-quality, diverse training data
2. **Learning Rate**: Start with 1e-4, adjust based on loss
3. **Monitoring**: Use Weights & Biases for experiment tracking
4. **Validation**: Keep a held-out dataset for evaluation

### Common Issues
- **OOM Errors**: Reduce batch size or use gradient accumulation
- **Slow Training**: Enable gradient checkpointing and mixed precision
- **Poor Results**: Check data formatting and learning rate

## 🔄 Inference After Fine-tuning

### Using Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt-oss-20b-finetuned")
model = AutoModelForCausalLM.from_pretrained(
    "gpt-oss-20b-finetuned",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate text
messages = [{"role": "user", "content": "Your question here"}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0][inputs.shape[1]:])
```

### Using Native GPT-OSS
```bash
cd gpt-oss
python -m gpt_oss.chat gpt-oss-20b-finetuned/original/
```

## 📈 Advanced Features

### Experiment Tracking
```bash
# Enable Weights & Biases
python finetune.py --use_wandb --dataset_path data.json
```

### Multi-GPU Training
```bash
# Use accelerate for multi-GPU
accelerate launch --multi_gpu finetune.py --dataset_path data.json
```

### Custom LoRA Configuration
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                    # Higher rank for more capacity
    lora_alpha=64,          # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
)
```

## 🐛 Troubleshooting

### Installation Issues
- Ensure CUDA is properly installed
- Use Python 3.12 or compatible version
- Install in a virtual environment

### Memory Issues
- Monitor GPU memory usage with `nvidia-smi`
- Reduce batch size or enable gradient checkpointing
- Use LoRA for parameter-efficient training

### Training Issues
- Check data formatting matches expected structure
- Monitor loss curves for signs of overfitting
- Validate on held-out data regularly

## 📝 License

This project uses the Apache 2.0 license, same as GPT-OSS. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📞 Support

For issues specific to:
- GPT-OSS model: Check the [official repository](https://github.com/openai/gpt-oss)
- Fine-tuning: Create an issue in this repository
- Transformers library: Check the [HuggingFace documentation](https://huggingface.co/docs/transformers)