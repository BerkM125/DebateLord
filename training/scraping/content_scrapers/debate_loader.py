#!/usr/bin/env python3
"""
File: debate_loader.py
Description: Populates the debates.csv data file with YouTube IDs of debates

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""

# Standard library imports
import os
from typing import Optional

# Third-party imports
import pandas as pd

# Local imports
import jubilee_loader as jl

module_dir = os.path.dirname(os.path.abspath(__file__))

def default_dataframe_loader():
    """
    Create a default DataFrame with a sample debate video.
    
    Returns:
        pd.DataFrame: DataFrame with default debate video ID and title.
    """
    yt_df = pd.DataFrame({
        'ID': ["WHLzNIGeQfA"],
        'title': ["Charlie Kirk vs Hasan Piker"],
        'numSpeakers': [3]
    })
    yt_df.to_csv("../../../data/datasets/debates.csv", index=False)

def load_debate_videos():
    """
    Load regular debate videos and save them to debates.csv."""
    print("Initiating regular debate video loading...")
    yt_df = jl.find_debate_videos(model="Gemini")
    csv_path = os.path.join(module_dir, "../../../data/datasets/debates.csv")

    if not yt_df.empty:
        # If the CSV exists, append without duplicating IDs
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, yt_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["ID"])
            combined_df.to_csv(csv_path, index=False)
        else:
            yt_df.to_csv(csv_path, index=False)
        print("Success. Debates added to debates.csv.")
    else:
        print("No debate videos found.")


if __name__ == "__main__":
    print("Initiating debate video loading...")
    load_debate_videos()
    print("Success.")