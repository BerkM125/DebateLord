#!/usr/bin/env python3
"""
File: jubilee_loader.py
Description: Populates the debates.csv data file with YouTube IDs of Jubilee debates, no reliance on AI except for
sentiment analysis and simple reasoning

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""
# Standard library imports
import os
import json
from typing import List, Optional, Dict, Any
import uuid
import hashlib
import time

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from google import genai
from tavily import TavilyClient

# Use absolute file paths
module_dir = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv()

# Prime Gemini
client = genai.Client()

def get_youtube_num_speakers(video_title: str) -> int:
    """
    Get the number of speakers in a YouTube video using Gemini API.
    Args:
        video_title (str): Title of the YouTube video.
    Returns:
        int: Number of speakers in the video.
    """
    speaker_ext_prompt = open(os.path.join(module_dir, "./system_instructions/info_extraction_instructions.md"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role":"user",
                "parts":[{"text":f"SYSTEM: {speaker_ext_prompt}\n\nUSER: Determine the number of speakers in the video called {video_title}"}]
            }
        ],
        config = {
            "response_mime_type": "application/json",
            "response_schema": { "type" : "integer" }
        }
    )
    return int(response.text)

def get_youtube_video_id(video_url: str) -> str:
    """
    Extract the YouTube video ID from a URL.
    
    Args:
        video_url (str): URL of the YouTube video.
    
    Returns:
        str: YouTube video ID.
    """
    if "youtube.com/watch?v=" in video_url:
        return video_url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("/")[-1]
    else:
        return ""
    
def get_youtube_video_title(video_url: str) -> str:
    """
    Get the title of a YouTube video using pytubefix library.
    
    Args:
        video_url (str): Complete URL of the YouTube video
        
    Returns:
        str: Title of the YouTube video, or error message if retrieval fails
        
    Note:
        Uses pytubefix instead of pytube to handle YouTube API changes better
    """
    print(f"Fetching title for video: {video_url}")
    from pytubefix import YouTube
    try:
        yt = YouTube(video_url)
        return yt.title
    except Exception as e:
        return f"Error: {str(e)}"

def request_to_batch_job(system_instruction: str, video_url: str) -> Dict[str, Any]:
    """
    Create a batch job request for the Gemini API, saves it into the batch request jsonl file
    
    Args:
        system_instruction (str): System instruction to guide the model.
        video_url (str): URL of the YouTube video to analyze.
        
    Returns:
        Dict[str, Any]: JSON object representing the batch job request.
    """
    
    # Create a deterministic UUID based on video URL
    namespace_uuid = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # UUID namespace for URLs
    url_hash = hashlib.sha1(video_url.encode()).digest()
    unique_id = str(uuid.uuid5(namespace_uuid, video_url))

    batch_json = {
        "contents": [
            {
                "parts": [
                    {"text": f"SYSTEM: {system_instruction}\n\nUSER: {get_youtube_video_title(video_url)}"}
                ]
            }
        ]
    }

    with open(os.path.join(module_dir, "./batch_instructions/batch_requests.jsonl"), "a") as f:
        f.write(json.dumps(batch_json) + "\n")

    return batch_json

def load_system_instructions(file_path: str) -> str:
    """
    Load and read system instructions from a file.
    
    Args:
        file_path (str): Path to the instruction file to read
        
    Returns:
        str: Contents of the instruction file with whitespace stripped
        
    Note:
        Uses context manager to ensure proper file handling
    """
    with open(file_path, 'r') as instructions:
        return instructions.read().strip()

# Sentiment analysis on YouTube titles 
def is_debate_video(video_url: str, model: str, classifier: str='./system_instructions/content_classifier_instructions.md') -> bool:
    response = ""
    sys_instruct = load_system_instructions(os.path.join(module_dir, classifier))
    if model == "Gemini":
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": f"SYSTEM: {sys_instruct}\n\nUSER: {get_youtube_video_title(video_url)}"}]
                }
            ],
            config = {
                "response_mime_type": "application/json",
                "response_schema": { "type" : "boolean" }
            }
        )
        print(f"Gemini's response to video called {get_youtube_video_title(video_url)}: {response.text}")
    return (response.text.lower() == "true" or response.text.lower() == "yes" or response.text.lower() == "1")

def find_debate_videos(model: str, classifier: str='./system_instructions/content_classifier_instructions.md') -> pd.DataFrame:
    """
    Find debate videos using the Gemini API.
    
    Args:
        model (str): The model to use for finding debate videos.
    
    Returns:
        list[str]: List of YouTube video IDs that are debates.
    """
    tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = tavily_client.search(query=open(os.path.join(module_dir, "./system_instructions/tavily_scraping_instructions.md")).read().strip(),
                         search_depth="advanced",
                         max_results=3,
                         include_domains=["youtube.com"])
    
    debate_df = {
        'ID': [],
        'title': [],
        'numSpeakers' : []
    }

    # Old method, being replaced by batch job
    for video in response['results']:
        # Ensure that it's a valid debate video
        if is_debate_video(video['url'], model, classifier):
            # Extract video contents
            video_title = get_youtube_video_title(video['url'])
            video_id = get_youtube_video_id(video['url'])
            video_speakers = get_youtube_num_speakers(video_title)

            # Add to DataFrame
            debate_df["ID"].append(video_id)
            debate_df["title"].append(video_title)
            debate_df["numSpeakers"].append(video_speakers)

    # sys_instruct = load_system_instructions(os.path.join(module_dir, classifier))
    # large_job = None
    # # New method will first create a batch request to Gemini to determine if content is 
    # # high quality and worth processing, and then process the batch job
    # for video in response['results']:
    #     all_batch_requests = []
    #     all_batch_requests.append(request_to_batch_job(
    #         system_instruction=sys_instruct,
    #         video_url=video['url']
    #     ))
    
    # print(f"All batch requests: {all_batch_requests}")

    # large_job = client.batches.create(
    #     model="models/gemini-2.5-flash",
    #     src=all_batch_requests,
    #     config={
    #         'display_name': 'Debate Video Content Classification Job'
    #     }
    # )
    # print(f"Created batch job for video: {video['url']} with ID: {large_job.id}, called {large_job.name}")

    # while large_job.state not in [genai.types.BatchJob.State.SUCCEEDED, genai.types.BatchJob.State.FAILED]:
    #     print("Job still running, waiting 3 seconds...")
    #     time.sleep(3)
    #     large_job = client.batches.get(large_job.name)

    # if large_job.state == genai.types.BatchJob.State.SUCCEEDED:
    #     results = client.batches.get_output(large_job.name)
    #     print(f"Batch job {large_job.name} completed successfully, processing results...")
    #     print(f"BATCH RESULTS: {results}")
    # else:
    #     print(f"Batch job {large_job.name} failed with state: {large_job.state}")

    return pd.DataFrame(debate_df)

def load_jubilee_videos():
    """
    Load Jubilee debate videos and save them to debates.csv."""
    print("Initiating Jubilee video loading...")
    yt_df = find_debate_videos(model="Gemini", classifier="./system_instructions/jubilee_classifier_instructions.md")
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
    load_jubilee_videos()