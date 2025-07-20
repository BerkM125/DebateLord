#!/usr/bin/env python3
"""
File: full_content_cycle.py
Description: Populates the debates.csv data file with updated YouTube videos of debates,
then extracts dialogue and content from these videos, splitting the debate into dialogue
spoken by each party. Dialogue is recorded by timestamp and also the speaker's ID.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""
# Standard library imports
import os
import sys
from typing import Optional

# Third-party imports
import pandas as pd

# Add project root to path for local imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    from DebateLord.training.scraping.content_scrapers.jubilee_loader import get_youtube_video_title, load_jubilee_videos
    from DebateLord.training.scraping.content_scrapers.youtube_extraction import process_youtube_video

    # First get a new batch of YouTube videos from Jubilee or elsewhere, using Jubilee for now
    load_jubilee_videos()

    # Next, load YouTube video IDs and titles
    yt_df = pd.read_csv("../../data/datasets/debates.csv")

    # Process each video ID
    for index, row in yt_df.iterrows():
        video_id = row['ID'].strip()
        num_speakers = row['numSpeakers']
        print(f"Processing video ID: {video_id} with {num_speakers} speakers")
        
        # Process the YouTube video to extract dialogue
        process_youtube_video(video_id, num_speakers, f"../../data/raw/debate_transcripts/{video_id}_dialogue.txt")