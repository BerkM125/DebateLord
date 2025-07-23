#!/usr/bin/env python3
"""
File: tune_debatelord_colab.py
Description: Colab-optimized version for fine-tuning the Mistral model.
Includes Google Drive integration and proper version management.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from google.colab import drive
import subprocess

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_colab_environment():
    """Setup the Colab environment with correct package versions."""
    logger.info("Setting up Colab environment...")
    setup_commands = [
        "pip install --upgrade pip",
        "pip install torch==2.1.0",
        "pip install transformers==4.35.2",
        "pip install peft==0.5.0",
        "pip install accelerate==0.24.1",
        "pip install bitsandbytes==0.41.1",
        "pip install datasets"
    ]
    
    for cmd in setup_commands:
        subprocess.run(cmd.split(), check=True)
    
    logger.info("Environment setup complete!")

def mount_drive():
    """Mount Google Drive and set up directories."""
    logger.info("Mounting Google Drive...")
    drive.mount('/content/drive')
    base_dir = "/content/drive/MyDrive/DebateLord"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def load_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """Load the model with GPU optimization for Colab."""
    logger.info(f"Loading model {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with GPU optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically use GPU
        torch_dtype=torch.float16,  # Use mixed precision
        low_cpu_mem_usage=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Add LoRA adaptor
    model = get_peft_model(model, lora_config)
    
    logger.info("Model loaded successfully")
    return model, tokenizer

def load_training_data(base_dir: str) -> List[Dict]:
    """Load all training data from JSON files."""
    data_dir = os.path.join(base_dir, "training/local_finetuning/training_data")
    all_patterns = []
    
    for json_file in Path(data_dir).glob("debate_*_patterns.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
                all_patterns.extend(patterns)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
    
    return all_patterns

def prepare_training_dataset(patterns: List[Dict], tokenizer) -> Dataset:
    """Prepare the dataset for training."""
    logger.info("Preparing dataset...")
    
    def format_prompt(example: Dict) -> str:
        return f"Opponent: {example['loser']}\nResponse: {example['winner']}"
    
    # Format prompts
    texts = [format_prompt(p) for p in patterns]
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
    return tokenized_dataset

def train_model(
    model,
    tokenizer,
    dataset,
    base_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """Fine-tune the model with Colab GPU optimization."""
    output_dir = os.path.join(base_dir, "models/debatelord_model")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.001,
        logging_dir=os.path.join(base_dir, "logs"),
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=True,
        fp16=True,  # Use mixed precision
        optim="adamw_torch",
        logging_first_step=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

def main():
    logger.info("Starting Colab training setup...")
    setup_colab_environment()
    base_dir = mount_drive()
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    logger.info("Loading training data...")
    patterns = load_training_data(base_dir)
    if not patterns:
        logger.error("No training data found!")
        return
    
    logger.info(f"Found {len(patterns)} training examples")
    dataset = prepare_training_dataset(patterns, tokenizer)
    
    logger.info("Starting fine-tuning process...")
    train_model(model, tokenizer, dataset, base_dir)
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()
