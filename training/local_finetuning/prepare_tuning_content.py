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
# extract SOME examples of back-and-forths INSIDE OF THOSE TRANSCRIPTS and store them into
# the fine tuning data format described within README.md. Ultimately, the code in this file should be able to
# take the contents of a winner file, for example:
# winner_of_0.txt:
# 1
# 
# and the contents of the matching  transcript file, for example:
# 0_dialogue.txt:
# ... Party 2: "charged language in politics which i'm generally in favor of i like a robust", 2:34
# Party 2: "public debate in a very loud and and", 2:38
# Party 2: "spirited public debate i have no problem with that whatsoever", 2:40
# Party 2: "what i'm talking about is the assumption", 2:43
# Party 2: "that people with whom we disagree politically are inherently of bad character or", 2:45
# Party 2: "in your words want to bring us back to the dark ages", 2:49
# Party 1: "but again it was your description of the state of the union address in twenty", 2:51
# Party 1: "twelve as fascist", 2:55
# Party 2: "the wording of the president from twenty twelve address", 2:57
# Party 2: "was bad and wrong that's all the plenty of things are bad and wrong but it doesn't", 3:00
# Party 1: "make them fascist", 3:05
# Party 2: "well", 3:07
# Party 2: "i suppose that's true but if you would like to again if you'd like to read me the", 3:09
# Party 2: "column out loud i suppose i can critique it for you", 3:12
# Party 1: "or well again with mister robama you said", 3:15
# Party 1: "jiu you're you're just yourself i only mentioned that because for this in", 3:18
# Party 1: "context", 3:21
# Party 1: "the jews who vote for robama are by large", 3:22
# Party 1: "jews in name only jinals you call them", 3:25
# Party 2: "my statement was based on the fact that juzin the united state as an ethnic group", 3:30
# Party 2: "are largely irreligious which is true by", 3:34
# Party 2: "every single poll juzin most irreligious group in the united states as an", 3:36
# Party 2: "orthodox jiu who actually takes juzin seriously the point that i am making is", 3:40
# Party 2: "that most juz who are ethnically juz are not religiously juzin in any context", 3:43
# Party 1: "no no no the point you were making is that juz who vote for a bama", 3:48
# Party 1: "are juzin named only", 3:52
# Party 2: "i said i said that yes that is correct that juz who voted for barak obama a", 3:54
# ...
# 
# And turn all of this into this kind of fine-tuning data:
# [
#     {
#         "loser": "my statement was based on the fact that juzin the united state as an ethnic group...",
#         "winner": "no no no the point you were making is that juz who vote for a bama"
#     },
#     {
#         "loser": "what i'm talking about is the assumption that people with whom we disagree politically are inherently of bad character or in your words want to bring us back to the dark ages",
#         "winner": "but again it was your description of the state of the union address in twenty twelve as fascist"
#     },
#     ...
# ]
# Then, by INDEX OF THE DEBATE, this fine-tuning data should be exported to json files in an appropriate directory.