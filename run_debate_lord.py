#!/usr/bin/env python3
"""
File: run_debate_lord.py
Description: Interactive chat interface for the fine-tuned DebateLord model.
Allows users to engage in debates with the model through a command-line interface.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import logging
from pathlib import Path
import colorama
from colorama import Fore, Style

# Initialize colorama for colored output
colorama.init()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebateLord:
    def __init__(self, model_path: str = "debatelord_model"):
        """Initialize DebateLord with the fine-tuned model."""
        logger.info("Loading DebateLord model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Model loaded successfully!")
        
        # Set up chat history
        self.chat_history: List[Dict[str, str]] = []
        
    def generate_response(self, user_input: str) -> str:
        """Generate a response to the user's input."""
        # Format the prompt with chat history
        prompt = self._format_chat_history()
        prompt += f"Opponent: {user_input}\nResponse:"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Response:")[-1].strip()
        
        # Update chat history
        self.chat_history.append({
            "user": user_input,
            "assistant": response
        })
        
        return response
    
    def _format_chat_history(self) -> str:
        """Format the chat history for the model prompt."""
        formatted = ""
        for exchange in self.chat_history[-3:]:  # Keep last 3 exchanges for context
            formatted += f"Opponent: {exchange['user']}\nResponse: {exchange['assistant']}\n\n"
        return formatted
    
    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []

def main():
    # Check if model exists
    if not os.path.exists("debatelord_model"):
        logger.error(
            "Model not found! Please run tune_debatelord.py first to train the model."
        )
        return
    
    # Initialize the model
    debate_lord = DebateLord()
    
    print(f"{Fore.GREEN}DebateLord initialized! Ready to debate.{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'quit' to exit, 'clear' to clear chat history.{Style.RESET_ALL}")
    
    while True:
        # Get user input
        user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            debate_lord.clear_history()
            print(f"{Fore.GREEN}Chat history cleared!{Style.RESET_ALL}")
            continue
        elif not user_input:
            continue
        
        # Generate and print response
        try:
            response = debate_lord.generate_response(user_input)
            print(f"{Fore.RED}DebateLord: {Style.RESET_ALL}{response}\n")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            print(f"{Fore.RED}Error: Failed to generate response. Please try again.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
