#!/usr/bin/env python3
"""
File: tune_debatelord.py
Description: Fine-tunes a local LLaMA model using debate pattern data prepared by
prepare_tuning_content.py. The model learns effective debate strategies from winning arguments.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """Load the Mistral model and tokenizer optimized for CPU training."""
    logger.info(f"Loading model {model_name} in CPU mode")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with CPU optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        low_cpu_mem_usage=True,
        use_cache=False,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Basic memory optimizations
    model.config.use_cache = False
    
    # Configure LoRA for CPU
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Reduced rank for CPU
        lora_alpha=16,  # Reduced alpha
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    # Add LoRA adaptor
    model = get_peft_model(model, lora_config)
    
    logger.info("Model loaded successfully in CPU mode")
    return model, tokenizer

def load_training_data() -> List[Dict]:
    """Load all training data from JSON files."""
    data_dir = Path(__file__).parent / "training_data"
    all_patterns = []
    
    for json_file in data_dir.glob("debate_*_patterns.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
                all_patterns.extend(patterns)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
    
    return all_patterns

def format_prompt(example: Dict) -> str:
    """Format a training example into a prompt."""
    return f"Opponent: {example['loser']}\nResponse: {example['winner']}"

def prepare_training_dataset(patterns: List[Dict], tokenizer) -> Dataset:
    """Prepare the dataset for training."""
    logger.info("Preparing dataset...")
    
    # Format prompts
    texts = [format_prompt(p) for p in patterns]
    
    def tokenize_function(examples):
        # Tokenize texts with proper padding and truncation
        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None  # Return lists instead of tensors
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    # Create initial dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Tokenize all texts
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
    output_dir: str = "debatelord_model",
    num_epochs: int = 3,
    batch_size: int = 2,  # Reduced batch size
    learning_rate: float = 1e-4  # Adjusted learning rate
):
    """Fine-tune the model on the debate patterns using LoRA."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    logger.info(f"Starting training with {len(dataset)} examples")
    logger.info(f"Model device: {model.device}")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,  # Smaller batch size for CPU
        gradient_accumulation_steps=8,  # Increased for CPU
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        weight_decay=0.001,
        logging_dir="./logs",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=True,
        no_cuda=True,  # Force CPU usage
        fp16=False,  # Disable mixed precision
        bf16=False,  # Disable bfloat16
        gradient_checkpointing=False,  # Disable for CPU
        optim="adamw_torch",  # Use standard AdamW
        ddp_find_unused_parameters=False,
        logging_first_step=True,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

def main():
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    logger.info("Loading training data...")
    patterns = load_training_data()
    if not patterns:
        logger.error("No training data found. Run prepare_tuning_content.py first.")
        return
    
    logger.info(f"Found {len(patterns)} training examples")
    dataset = prepare_training_dataset(patterns, tokenizer)
    
    logger.info("Starting fine-tuning process...")
    train_model(model, tokenizer, dataset)
    
    logger.info("Fine-tuning complete!")

if __name__ == "__main__":
    main()
