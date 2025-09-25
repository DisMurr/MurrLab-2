#!/usr/bin/env python3
"""
GPT-OSS 20B Fine-tuning Setup and Training Script

This script provides a complete fine-tuning pipeline for GPT-OSS 20B model
using the Transformers library with LoRA (Low-Rank Adaptation) for efficiency.
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import argparse
from pathlib import Path
import logging
import wandb
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTOSSFineTuner:
    def __init__(self, 
                 model_path: str = "gpt-oss-20b",
                 output_dir: str = "gpt-oss-20b-finetuned",
                 use_lora: bool = True):
        """
        Initialize the fine-tuning setup
        
        Args:
            model_path: Path to the GPT-OSS model
            output_dir: Directory to save fine-tuned model
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
        
        if self.use_lora:
            logger.info("Setting up LoRA configuration")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,  # Rank
                lora_alpha=32,  # Scaling parameter
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def prepare_dataset(self, dataset_path: str = None, dataset_name: str = None):
        """
        Prepare training dataset
        
        Args:
            dataset_path: Path to local dataset (JSON/JSONL)
            dataset_name: Name of HuggingFace dataset
        """
        if dataset_path:
            logger.info(f"Loading dataset from {dataset_path}")
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
            elif dataset_path.endswith('.jsonl'):
                data = []
                with open(dataset_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                raise ValueError("Dataset must be JSON or JSONL format")
            
            dataset = Dataset.from_list(data)
        elif dataset_name:
            logger.info(f"Loading dataset {dataset_name} from HuggingFace")
            dataset = load_dataset(dataset_name, split="train")
        else:
            # Create a sample dataset for demonstration
            logger.info("Creating sample dataset for demonstration")
            sample_data = [
                {"text": "What is machine learning? Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from data."},
                {"text": "Explain quantum computing. Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations."},
                {"text": "What is the capital of France? The capital of France is Paris, which is located in the north-central part of the country."},
            ]
            dataset = Dataset.from_list(sample_data)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            # For GPT-OSS, we need to use the harmony format
            texts = []
            for text in examples["text"]:
                # Simple conversation format - you may want to adapt this
                messages = [
                    {"role": "user", "content": text.split("?")[0] + "?" if "?" in text else text.split(".")[0]},
                    {"role": "assistant", "content": text.split("?")[1].strip() if "?" in text else text}
                ]
                
                # Apply chat template
                formatted_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                texts.append(formatted_text)
            
            return self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=2048,
                return_overflowing_tokens=False,
            )
        
        logger.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, 
              dataset,
              num_epochs: int = 3,
              learning_rate: float = 1e-4,
              batch_size: int = 1,
              gradient_accumulation_steps: int = 8,
              warmup_steps: int = 100,
              save_steps: int = 500,
              logging_steps: int = 10,
              use_wandb: bool = False):
        """
        Train the model
        
        Args:
            dataset: Tokenized dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            use_wandb: Whether to use Weights & Biases for logging
        """
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            bf16=torch.cuda.is_available(),  # Use bfloat16 if available
            gradient_checkpointing=True,
            report_to="wandb" if use_wandb else None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer

def create_sample_dataset(output_path: str = "sample_training_data.json"):
    """Create a sample training dataset"""
    sample_data = [
        {
            "text": "What is artificial intelligence?",
            "response": "Artificial intelligence (AI) is a branch of computer science that aims to create machines and systems capable of performing tasks that typically require human intelligence, such as learning, reasoning, problem-solving, and understanding language."
        },
        {
            "text": "How does machine learning work?",
            "response": "Machine learning works by training algorithms on large datasets to identify patterns and make predictions or decisions without being explicitly programmed for each specific task. The system learns from data and improves its performance over time."
        },
        {
            "text": "What is deep learning?",
            "response": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition."
        },
        # Add more examples as needed
    ]
    
    # Convert to the format expected by the trainer
    formatted_data = []
    for item in sample_data:
        formatted_data.append({
            "text": f"Q: {item['text']}\nA: {item['response']}"
        })
    
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Sample dataset created: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS 20B")
    parser.add_argument("--model_path", default="gpt-oss-20b", help="Path to GPT-OSS model")
    parser.add_argument("--dataset_path", help="Path to training dataset (JSON/JSONL)")
    parser.add_argument("--dataset_name", help="HuggingFace dataset name")
    parser.add_argument("--output_dir", default="gpt-oss-20b-finetuned", help="Output directory")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--create_sample", action="store_true", help="Create sample dataset")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset()
        return
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="gpt-oss-finetuning")
    
    # Initialize fine-tuner
    fine_tuner = GPTOSSFineTuner(
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_lora=args.use_lora
    )
    
    # Load model and tokenizer
    fine_tuner.load_model_and_tokenizer()
    
    # Prepare dataset
    dataset = fine_tuner.prepare_dataset(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name
    )
    
    # Train the model
    trainer = fine_tuner.train(
        dataset=dataset,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.use_wandb
    )
    
    print(f"Fine-tuning completed! Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()