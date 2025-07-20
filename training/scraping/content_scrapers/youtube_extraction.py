#!/usr/bin/env python3
"""
File: youtube_extraction.py
Description: Extracts the dialogue and content from YouTube videos of debates and splits
the debate into the dialogue spoken by each party. Dialogue is recorded by timestamp
and also the speaker's ID.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""
# Standard library imports
import json
import os
import tempfile
from typing import Dict, List, Optional, Union

# Third-party imports
import pandas as pd
import torch
import whisper
import yt_dlp
from pyannote.audio import Pipeline
from pyannote.core import Segment

# Global module directory
module_dir = os.path.dirname(os.path.abspath(__file__))

def load_youtube_video_ids(data_file_path: str) -> pd.DataFrame:
    """
    Load and return DataFrame of YouTube IDs and associated metadata from dataset.
    
    Args:
        data_file_path (str): Path to the CSV file containing YouTube video data
        
    Returns:
        pd.DataFrame: DataFrame containing video IDs and metadata columns like 
                     'ID', 'title', and 'numSpeakers'
    """
    yt_df = pd.read_csv(data_file_path)
    print(f"Stored debates: {yt_df}")

    return yt_df

def download_youtube_audio(video_id: str, output_dir: str="temp_audio") -> str:
    """
    Download audio from YouTube video and extract 10-second clip from middle
   
    Args:
        video_id (str): YouTube video ID
        output_dir (str): Directory to save audio file
   
    Returns:
        str: Path to downloaded audio file (10-second clip from middle)
    """
    import subprocess
    
    os.makedirs(output_dir, exist_ok=True)
   
    # Configure yt-dlp options to download full audio first
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s_full.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
   
    url = f"https://www.youtube.com/watch?v={video_id}"
   
    try:
        # Download full audio and get video info
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 0)
        
        # Calculate middle segment (10 seconds centered around middle)
        start_time = max(0, (duration / 2) - 5)
        
        # Define file paths
        input_file = f"{output_dir}/{video_id}_full.wav"
        output_file = f"{output_dir}/{video_id}.wav"
        
        # Trim the audio file to 10-second clip from middle
        result = subprocess.run([
            'ffmpeg', '-i', input_file,
            '-ss', str(start_time),
            '-t', '600',
            '-c', 'copy',
            output_file,
            '-y'  # Overwrite output file
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return None
        
        # Remove the full file to save space
        os.remove(input_file)
        
        return output_file
   
    except Exception as e:
        print(f"Error downloading audio: {e}")
        # Clean up any partial files
        try:
            full_file = f"{output_dir}/{video_id}_full.wav"
            if os.path.exists(full_file):
                os.remove(full_file)
        except:
            pass
        return None
    
def transcribe_audio(audio_path: str) -> Optional[Dict]:
    """
    Transcribe audio using OpenAI's Whisper model.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
    
    Returns:
        Optional[Dict]: Dictionary containing transcription results with segments,
                       including text, timestamps, and confidence scores.
                       Returns None if transcription fails.
    """
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def perform_speaker_diarization(audio_path: str, num_speakers: int = 2) -> Optional[List[Dict[str, Union[float, str]]]]:
    """
    Perform speaker diarization using pyannote.audio to identify speaker segments.
    
    Args:
        audio_path (str): Path to the audio file to analyze
        num_speakers (int, optional): Expected number of speakers in the audio. Defaults to 2.
    
    Returns:
        Optional[List[Dict[str, Union[float, str]]]]: List of speaker segments, where each segment
            contains 'start' time (float), 'end' time (float), and 'speaker' label (str).
            Returns None if diarization fails.
    
    Note:
        Requires authentication with Hugging Face CLI before use.
    """
    try:
        # Load the speaker diarization pipeline
        # Ensure that you're logged into HuggingFace CLI before running this.
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        
        # Run diarization, first ensuring that the number of expected speakers is inputted
        diarization = pipeline(audio_path, num_speakers=num_speakers)
        
        # Convert to list of segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        return speaker_segments
    
    except Exception as e:
        print(f"Error in speaker diarization: {e}")
        print("Note: You may need to authenticate with Hugging Face:")
        print("huggingface-cli login")
        return None

def format_timestamp(seconds: Union[int, float]) -> str:
    """
    Convert seconds to MM:SS format.
    
    Args:
        seconds (Union[int, float]): Time in seconds to format
        
    Returns:
        str: Formatted time string in 'MM:SS' format
    """
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes}:{seconds:02d}"

def assign_speakers_to_transcription(transcription, speaker_segments):
    """
    Assign speakers to transcription segments
    
    Args:
        transcription (dict): Whisper transcription result
        speaker_segments (list): Speaker diarization segments
    
    Returns:
        list: Combined transcription with speaker assignments
    """
    result = []
    
    # Create speaker mapping
    unique_speakers = list(set(seg['speaker'] for seg in speaker_segments))
    speaker_mapping = {speaker: f"Party {i+1}" for i, speaker in enumerate(unique_speakers)}
    
    for segment in transcription['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()
        
        # Find the speaker for this segment
        assigned_speaker = "Unknown"
        for speaker_seg in speaker_segments:
            # Check if transcription segment overlaps with speaker segment
            if (start_time >= speaker_seg['start'] and start_time <= speaker_seg['end']) or \
               (end_time >= speaker_seg['start'] and end_time <= speaker_seg['end']) or \
               (start_time <= speaker_seg['start'] and end_time >= speaker_seg['end']):
                assigned_speaker = speaker_mapping[speaker_seg['speaker']]
                break
        
        result.append({
            'speaker': assigned_speaker,
            'text': text,
            'start': start_time,
            'timestamp': format_timestamp(start_time)
        })
    
    return result

def process_youtube_video(video_id: str, num_speakers: int, output_file: str="dialogue_output.txt"):
    """
    Main function to process YouTube video and extract speaker-identified dialogue
    
    Args:
        video_id (str): YouTube video ID
        num_speakers (int): Number of speakers in the video
        output_file (str): Output text file name
    """
    print(f"Processing video: {video_id}")
    
    # Step 1: Download audio
    print("1. Downloading audio...")
    audio_path = download_youtube_audio(video_id)
    if not audio_path:
        print("Failed to download audio")
        return
    
    # Step 2: Transcribe audio
    print("2. Transcribing audio...")
    transcription = transcribe_audio(audio_path)
    if not transcription:
        print("Failed to transcribe audio")
        return
    
    # Step 3: Perform speaker diarization
    print("3. Performing speaker diarization...")
    speaker_segments = perform_speaker_diarization(audio_path=audio_path, num_speakers=num_speakers)
    if not speaker_segments:
        print("Failed to perform speaker diarization")
        return
    
    # Step 4: Combine transcription with speaker identification
    print("4. Assigning speakers to dialogue...")
    dialogue_with_speakers = assign_speakers_to_transcription(transcription, speaker_segments)
    
    # Step 5: Write to output file
    print("5. Writing output...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dialogue_with_speakers:
            f.write(f'{entry["speaker"]}: "{entry["text"]}", {entry["timestamp"]}\n')
    
    print(f"Dialogue saved to {output_file}")
    print(f"Total segments: {len(dialogue_with_speakers)}")
    
    # Clean up temporary audio file
    try:
        os.remove(audio_path)
        os.rmdir("temp_audio")
    except:
        pass

if __name__ == "__main__":
    # Replace with your video ID
    yt_df = pd.read_csv(os.path.join(module_dir, "../../../data/datasets/debates.csv"))
    
    for index, row in yt_df.iterrows():
        video_id = row['ID'].strip()
        num_speakers = row['numSpeakers']
        print(f"Processing video ID: {video_id} with {num_speakers} speakers")
        
        # Process the YouTube video to extract dialogue
        process_youtube_video(video_id, num_speakers, f"../../../data/raw/debate_transcripts/{video_id}_dialogue.txt")