#!/usr/bin/env python3
"""
File: prepare_tuning_content.py
Description: Prepares training data for fine-tuning a local LLaMA model by extracting
winning debate patterns from transcripts and organizing them into training examples.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""

import os
import json
import re
from typing import List, Dict, Tuple

def get_transcript_path(index: int) -> str:
    """Get the path to a transcript file by its index."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "raw", "debate_transcripts",
        f"{index}_dialogue.txt"
    )

def get_winner_path(index: int) -> str:
    """Get the path to a winner file by its index."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "processed", "debate_winners",
        f"winner_of_{index}.txt"
    )

def read_winner(index: int) -> int:
    """Read the winner number from a winner file."""
    try:
        with open(get_winner_path(index), 'r') as f:
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

def save_training_data(patterns: List[Dict], index: int):
    """Save training patterns to a JSON file."""
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "training_data"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"debate_{index}_patterns.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(patterns, f, indent=4, ensure_ascii=False)

def process_debate(index: int):
    """Process a single debate by its index."""
    winner = read_winner(index)
    if winner is None:
        return
    
    transcript_path = get_transcript_path(index)
    utterances = parse_transcript(transcript_path)
    if not utterances:
        return
    
    patterns = extract_debate_patterns(utterances, winner)
    if patterns:
        save_training_data(patterns, index)
        print(f"Processed debate {index}: extracted {len(patterns)} patterns")

if __name__ == "__main__":
    print("Entering main...")
    # Process all debates that have both transcript and winner files
    index = 0
    while True:
        if not os.path.exists(get_transcript_path(index)):
            break
        process_debate(index)
        index += 1
    
    print(f"Finished processing {index} debates")