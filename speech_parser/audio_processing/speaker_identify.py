import os
from pathlib import Path

# import numpy as np
from loguru import logger

from pyannote.audio import Pipeline

# from pyannote.core import Segment
from pyannote.audio.pipelines.utils.hook import ProgressHook

import csv
import json

from dotenv import load_dotenv

# Visit https://hf.co/settings/tokens to create your access token

# Load environment variables from .env
load_dotenv()
AUTH_TOKEN_HUGGINFACE = Path(os.getenv("AUTH_TOKEN_HUGGINFACE"))


def identify_speakers_pyannote(audio_file_path):
    """
    Use pyannote.audio to perform speaker diarization on the audio file and return speaker segments.
    Args:
        audio_file_path (str): Path to the audio file.
    Returns:
        list: List of dictionaries containing speaker, start_time, and end_time.
    """
    print(f"Performing speaker diarization on {audio_file_path} with pyannote.audio...")

    load_dotenv()
    token = os.getenv("AUTH_TOKEN_HUGGINFACE")  # Ensure your token is in .env and loaded

    # Load pretrained pyannote.audio pipeline for speaker diarization
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)

    # Apply the pipeline to the audio file for diarization
    with ProgressHook() as hook:
        diarization = pipeline(audio_file_path, hook=hook)
    # diarization = pipeline(audio_file_path) # No ProgressHook

    audio_file_path = str(audio_file_path)

    # Save RTTM file
    rttm_file_path = f"{str(audio_file_path)[:-4]}.rttm"  # Adjusting for .wav extension

    with open(rttm_file_path, "w") as rttm:
        diarization.write_rttm(rttm)
    print(f"RTTM file saved to: {rttm_file_path}")

    # Parse the diarization output to get speaker segments
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({"speaker": speaker, "start_time": turn.start, "end_time": turn.end})

    print(f"Speaker diarization completed. Segments: {speaker_segments}")
    return speaker_segments


def assign_speaker_names(rttm_segments):
    """
    Assign real names to the speakers by asking for user input.
    Args:
        rttm_segments (list): List of RTTM segment data.
    Returns:
        dict: A mapping of speaker IDs to real names.
    """
    speaker_set = set([segment["speaker"] for segment in rttm_segments])
    speaker_name_map = {}

    logger.info("Assign real names to speakers.")
    for speaker in speaker_set:
        real_name = input(f"Enter real name for {speaker}: ")
        speaker_name_map[speaker] = real_name

    logger.info(f"Assigned names: {speaker_name_map}")
    return speaker_name_map


def filter_rttm_segments(rttm_segments, min_duration=1.0):
    """
    Filters out RTTM segments shorter than the specified minimum duration.

    Args:
        rttm_segments (list): List of RTTM segments.
        min_duration (float): Minimum duration (in seconds) to keep a segment.

    Returns:
        list: Filtered list of RTTM segments.
    """
    filtered_segments = [
        segment for segment in rttm_segments if (segment["end_time"] - segment["start_time"]) >= min_duration
    ]

    logger.info(f"Filtered {len(rttm_segments) - len(filtered_segments)} short segments.")
    return filtered_segments


def check_rttm_file(audio_file_path):
    """
    Check if the RTTM file exists before performing diarization.
    Args:
        audio_file_path (str): Path to the audio file.
    Returns:
        bool: True if the RTTM file exists, False otherwise.
    """
    rttm_file_path = f"{str(audio_file_path)[:-4]}.rttm"
    if os.path.exists(rttm_file_path):
        logger.info(f"Using existing RTTM file: {rttm_file_path}")
        return rttm_file_path
    else:
        logger.info("RTTM file not found, proceeding with diarization.")
        return None


def load_rttm_file(rttm_file):
    """
    Parse the RTTM file and return the speaker segments.
    Args:
        rttm_file (str): Path to the RTTM file.
    Returns:
        list: List of speaker segments, each containing speaker label, start_time, and end_time.
    """
    speaker_segments = []

    with open(rttm_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue

            # Extract speaker information
            speaker = parts[7]
            start_time = float(parts[3])  # start time in seconds
            duration = float(parts[4])  # duration in seconds
            end_time = start_time + duration

            # Store the segment information
            speaker_segments.append({"speaker": speaker, "start_time": start_time, "end_time": end_time})

    return speaker_segments


def save_transcription_csv(speaker_transcription, file_path):
    if not speaker_transcription:
        print(f"No data to save in {file_path}")
        return

    with open(file_path, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Speaker", "Start Time", "End Time", "Transcription"])  # Column headers

        for entry in speaker_transcription:
            print(f"Writing to CSV: {entry}")
            writer.writerow(
                [entry["speaker"], entry["start_time"], entry["end_time"], entry["transcription"]]
            )
    print(f"Transcription saved to: {file_path}")


def save_transcription_json(speaker_transcription, file_path):
    if not speaker_transcription:
        print(f"No data to save in {file_path}")
        return

    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(speaker_transcription, json_file, ensure_ascii=False, indent=4)
    print(f"Transcription saved to: {file_path}")
