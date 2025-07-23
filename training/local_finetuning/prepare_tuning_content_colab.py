#!/usr/bin/env python3
"""
File: prepare_tuning_content_colab.py
Description: Colab-optimized version for preparing training data for fine-tuning.
Includes Google Drive mounting and proper version management.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.
"""

import os
import json
import re
from typing import List, Dict
from google.colab import drive
import subprocess
from pathlib import Path

def setup_colab_environment():
    """Setup the Colab environment with correct package versions."""
    print("Setting up Colab environment...")
    
    # Install specific versions of packages to avoid compatibility issues
    setup_commands = [
        "pip install --upgrade pip",
        "pip install torch==2.1.0",
        "pip install transformers==4.35.2",  # Known compatible version
        "pip install peft==0.5.0",           # Known compatible version
        "pip install accelerate==0.24.1",
        "pip install bitsandbytes==0.41.1",
        "pip install datasets"
    ]
    
    for cmd in setup_commands:
        subprocess.run(cmd.split(), check=True)
    
    print("Environment setup complete!")

def mount_drive():
    """Mount Google Drive and create necessary directories."""
    print("Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Create necessary directories in Google Drive
    base_dir = "/content/drive/MyDrive/DebateLord"
    dirs = [
        "data/raw/debate_transcripts",
        "data/processed/debate_winners",
        "training/local_finetuning/training_data"
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    
    return base_dir

def get_transcript_path(index: int, base_dir: str) -> str:
    """Get the path to a transcript file by its index."""
    return os.path.join(
        base_dir,
        "data", "raw", "debate_transcripts",
        f"{index}_dialogue.txt"
    )

def get_winner_path(index: int, base_dir: str) -> str:
    """Get the path to a winner file by its index."""
    return os.path.join(
        base_dir,
        "data", "processed", "debate_winners",
        f"winner_of_{index}.txt"
    )

def read_winner(index: int, base_dir: str) -> int:
    """Read the winner number from a winner file."""
    try:
        with open(get_winner_path(index, base_dir), 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        print(f"Warning: No winner file found for debate {index}")
        return None
    except ValueError:
        print(f"Warning: Invalid winner number in file for debate {index}")
        return None

def parse_transcript(transcript_path: str) -> List[Dict]:
    """Parse a transcript file into a list of utterances."""
    utterances = []
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Parse lines like: Party 1: "text here", 2:34
                match = re.match(r'Party (\d+): "(.*?)", (\d+:\d+)', line.strip())
                if match:
                    speaker, text, timestamp = match.groups()
                    utterances.append({
                        "speaker": int(speaker),
                        "text": text.strip(),
                        "timestamp": timestamp
                    })
    except FileNotFoundError:
        print(f"Warning: Transcript file not found: {transcript_path}")
        return []
    
    return utterances

def extract_debate_patterns(utterances: List[Dict], winner: int) -> List[Dict]:
    """Extract winning debate patterns from utterances."""
    patterns = []
    
    # Look for pairs of consecutive utterances where winner responds to loser
    for i in range(len(utterances) - 1):
        current = utterances[i]
        next_utterance = utterances[i + 1]
        
        # If current speaker is loser and next is winner
        if current["speaker"] != winner and next_utterance["speaker"] == winner:
            patterns.append({
                "loser": current["text"],
                "winner": next_utterance["text"]
            })
    
    return patterns

def save_training_data(patterns: List[Dict], index: int, base_dir: str):
    """Save training patterns to a JSON file."""
    output_dir = os.path.join(
        base_dir,
        "training", "local_finetuning", "training_data"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"debate_{index}_patterns.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, indent=4, ensure_ascii=False)
    print(f"Saved patterns to {output_path}")

def process_debate(index: int, base_dir: str):
    """Process a single debate by its index."""
    winner = read_winner(index, base_dir)
    if winner is None:
        return
    
    transcript_path = get_transcript_path(index, base_dir)
    utterances = parse_transcript(transcript_path)
    if not utterances:
        return
    
    patterns = extract_debate_patterns(utterances, winner)
    if patterns:
        save_training_data(patterns, index, base_dir)
        print(f"Processed debate {index}: extracted {len(patterns)} patterns")

def main():
    print("Setting up Colab environment...")
    setup_colab_environment()
    
    print("Mounting Google Drive...")
    base_dir = mount_drive()
    
    print("Starting data processing...")
    index = 0
    while True:
        if not os.path.exists(get_transcript_path(index, base_dir)):
            break
        process_debate(index, base_dir)
        index += 1
    
    print(f"Finished processing {index} debates")
    print(f"All data saved to {base_dir}/training/local_finetuning/training_data/")

if __name__ == "__main__":
    main()
