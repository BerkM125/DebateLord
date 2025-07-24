"""
File: compile_llama_context.py
Description: Compiles context for DebateLord's prompt engineering.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""

import asyncio
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

# Get initial user's debate topic
def initiate_conversation() -> str:
    """
    Get initial user's debate topic.

    Returns:
        str: The user's input for the debate topic.
    """
    return str(input("Welcome to DebateLord. What would you like to debate about today?\n"))

# Prepare all context for a well made prompt for DebateLord
def prepare_debate_context(user_input: str) -> str:
    """
    Prepare all context for a well-made prompt for DebateLord.

    Args:
        user_input (str): The user's input for the debate topic.

    Returns:
        str: The compiled context for the prompt.
    """
    training_data = ""
    with open(os.path.join(module_dir, "./gemini_engineered_context.md"), "r+") as context_file:
        context = context_file.read()
        pre_data_ctx = context[0:context.index("[Insert here]")]
        post_data_ctx = context[context.index("[Insert here]"):len(context)]

        # Change this to get all files in the directory instead of a hardcoded 10
        for findex in range(10):
            new_content = open(os.path.join(module_dir, f"../local_finetuning/training_data/debate_{findex}_patterns.json")).read()
            training_data += f"\n{new_content}"
        
        context_file.close()
    
    return f"SYSTEM:\n{pre_data_ctx}\n{training_data}\n{post_data_ctx}\n\nUSER: {user_input}"