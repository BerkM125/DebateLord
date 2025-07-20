#!/usr/bin/env python3
"""
File: debate_assessment.py
Description: Utility script for assessing debate transcripts using AI. Analyzes transcripts
from the raw/debate_transcripts directory and provides an assessment of which party won
the debate based on various factors including tone, evidence, and debate strategies.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""

from google import genai
from dotenv import load_dotenv
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv()
client = genai.Client()

if __name__ == "__main__":
    # Get path to debate transcripts directory (going up from utils to project root, then to transcripts)
    transcripts_dir = os.path.join(os.path.dirname(module_dir), 
                                 "data", "raw", "debate_transcripts")
    
    # Get all files in the debate_transcripts directory
    filenames = []
    for file in os.listdir(transcripts_dir):
        if file.endswith("_dialogue.txt"):  # Only include debate dialogue files
            filenames.append(os.path.join("../data/raw/debate_transcripts", file))

    file_index = 0
    for fn in filenames:
        with open(os.path.join(module_dir, fn), "r") as curr_file:
            transcript = curr_file.read()
            post_context_prompt = "Given this transcript of an approximately 10 minute long debate, assess which party in this debate won out over the other, and explain why they were the winner. Cite tone, evidence-backed arguments, good counterarguments and strategies, among other debate factors in your assessment."

            full_prompt = f"{transcript}\n\n{post_context_prompt}"

            ai_assessment = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "parts": [
                            {"text" : full_prompt}
                        ]
                    }
                ]
            )

            print(f"First debate, id {fn} assessment below: ")
            print(ai_assessment.text)
            print("\n\n")

            with open(os.path.join(module_dir, f"../data/processed/debate_assessments/debate{file_index}.txt"), 'w') as outf:
                outf.write(f"VERDICT ON DEBATE ID {fn}: \n\n{ai_assessment.text}")
            
            file_index += 1

    print("Assessment finished.")
            
