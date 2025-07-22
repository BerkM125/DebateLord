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
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from accelerate import dispatch_model
import bitsandbytes as bnb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """Load the Mistral model and tokenizer with 4-bit quantization for efficiency."""
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
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
    # Format prompts
    texts = [format_prompt(p) for p in patterns]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["input_ids"].clone()
    })
    
    return dataset

def train_model(
    model,
    tokenizer,
    dataset,
    output_dir: str = "debatelord_model",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5
):
    """Fine-tune the model on the debate patterns."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=4,
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
