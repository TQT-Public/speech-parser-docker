# import json
import csv
import json
import os

import re
from pathlib import Path
from loguru import logger

import shutil

from dotenv import load_dotenv
import pandas as pd

from speech_parser.utils.env import env_as_float, env_as_int, env_as_path, env_as_str

# Load environment variables from .env
load_dotenv()

# Define paths for default and custom configurations
VOSK_MODEL_PATH_ENV = env_as_path("VOSK_MODEL_PATH", "./model/vosk")
MODEL_NAME = env_as_str("MODEL_NAME", "vosk-model-ru-0.42")
# VOSK_MODEL_FULL_PATH_ENV = Path(str(VOSK_MODEL_PATH_ENV))
VOSK_MODEL_FULL_PATH_ENV = env_as_path("VOSK_MODEL_PATH", "./model/vosk/vosk-model-ru-0.42")
CONFIG_PATH_ENV = env_as_path("CONFIG_PATH", "./configs")
CONFIG_FILE = Path(VOSK_MODEL_FULL_PATH_ENV, "conf", "model.conf")
# CONFIG_FILE = os.path.join(VOSK_MODEL_FULL_PATH_ENV, "conf", "model.conf")
# DEFAULT_CONFIG_FILE = os.path.join(VOSK_MODEL_FULL_PATH_ENV, "conf", "model_default.config")
DEFAULT_CONFIG_FILE = Path(CONFIG_PATH_ENV, "model_default.config")


def format_dialogue_for_summary(df):
    """
    Format the dialogue data for summarization by combining speaker, transcription, and timestamps.

    Args:
        df (pd.DataFrame): DataFrame containing dialogue data with 'speaker', 'transcription', 'start_time', and 'end_time'.

    Returns:
        str: A formatted dialogue string for use as a prompt for language models.
    """
    dialogue_text = "\n".join(
        [
            f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
            for _, row in df.iterrows()
            if pd.notna(row["transcription"])
        ]
    )
    return dialogue_text


def create_empty_csv_and_json_if_not_exists(csv_file_path: str, json_file_path: str):
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["speaker", "transcription", "start_time", "end_time"])
        print(f"Created empty CSV file: {csv_file_path}")
    if not os.path.exists(json_file_path):
        with open(json_file_path, mode="w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        print(f"Created empty JSON file: {json_file_path}")


def parse_config_file(config_file_path):
    config = {}
    with open(config_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue

            # Handle both --key=value and --key value formats
            if line.startswith("--"):
                if "=" in line:
                    key, value = line.split("=", maxsplit=1)  # Split on `=`
                else:
                    try:
                        key, value = re.split(r"\s+", line, maxsplit=1)  # Split on space
                    except ValueError:
                        logger.error(f"Could not parse line in config: {line}")
                        continue
                config[key] = value
            else:
                logger.error(f"Unexpected format in config: {line}")
    return config


def write_config_file(config, file_path):
    """Write the config dictionary back to the config file in the original format."""
    with open(file_path, "w") as file:
        for key, value in config.items():
            file.write(f"{key}={value}\n")


def set_custom_vosk_config():

    load_dotenv()
    CUSTOM_VOSK_BEAM = env_as_float("CUSTOM_VOSK_BEAM", 13.0)
    CUSTOM_VOSK_MAX_ACTIVE = env_as_int("CUSTOM_VOSK_MAX_ACTIVE", 7000)
    CUSTOM_VOSK_LATTICE_BEAM = env_as_float("CUSTOM_VOSK_LATTICE_BEAM", 6.0)

    custom_config = {
        "--beam": CUSTOM_VOSK_BEAM,
        "--max-active": CUSTOM_VOSK_MAX_ACTIVE,
        "--lattice-beam": CUSTOM_VOSK_LATTICE_BEAM,
        "--endpoint.silence-phones": "1:2:3:4:5:6:7:8:9:10",
    }

    # Back up the default config if not already backed up
    if not os.path.exists(DEFAULT_CONFIG_FILE):
        shutil.copy(CONFIG_FILE, DEFAULT_CONFIG_FILE)
        logger.info(f"Backed up default config to {DEFAULT_CONFIG_FILE}")

    # Parse the existing config and update with custom values
    config = parse_config_file(CONFIG_FILE)
    config.update(custom_config)

    # Write the custom settings back to model.conf with = format
    with open(CONFIG_FILE, "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    logger.info(f"Applied custom Vosk configuration to {CONFIG_FILE}")


def set_default_vosk_config():
    # Restore default config from backup
    if os.path.exists(DEFAULT_CONFIG_FILE):
        shutil.copy(DEFAULT_CONFIG_FILE, CONFIG_FILE)
        logger.info(f"Restored default config from {DEFAULT_CONFIG_FILE}")
    else:
        logger.error("Default config not found. Unable to restore.")


# [Deleting]


def delete_audio_segments(folder_path, filename_stem, speaker_name_map=None):
    """
    Delete all instances of {filename}_converted_SPEAKER_X_partX.wav or
    {filename}_converted_{real_speaker}_partX.wav in a specific folder.

    Args:
        folder_path (str or Path): Path to the folder containing the audio files.
        filename_stem (str): Stem of the original audio file (e.g., "ZOOM0068").
        speaker_name_map (dict, optional): A dictionary mapping speaker IDs to real speaker names.
    """
    folder_path = Path(folder_path)

    # Regular expression to match files in the pattern {filename}_converted_SPEAKER_X_partX.wav
    file_pattern_generic = re.compile(rf"^{filename_stem}_converted_SPEAKER_\d+_part\d+\.wav$")

    # If speaker_name_map exists, create patterns to match files with real speaker names
    if speaker_name_map:
        file_patterns_real = [
            re.compile(rf"^{filename_stem}_converted_{re.escape(real_name)}_part\d+\.wav$")
            for real_name in speaker_name_map.values()
        ]
    else:
        file_patterns_real = []

    # Loop through all files in the folder
    for file in folder_path.iterdir():
        if file.is_file() and (
            file_pattern_generic.match(file.name)
            or any(pattern.match(file.name) for pattern in file_patterns_real)
        ):
            logger.info(f"Deleting file: {file}")
            try:
                file.unlink()  # Delete the file
            except Exception as e:
                logger.error(f"Error deleting file {file}: {e}")

    logger.info(f"All matching files deleted in {folder_path}.")


def ensure_directory_exists(directory):
    """Ensure that a directory exists, and create it if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


if __name__ == "__main__":
    # $ python -m utils.helpers
    # Example: Delete all segments from "ZOOM0068" in the folder

    # delete_audio_segments(
    #     "./audio_files/parts", "ZOOM0068", {1: "MIRO", 2: "SERGO"}
    # )
    # Run Variants:
    delete_audio_segments(
        "./audio_files/parts",
        "ZOOM0067",
        {1: "SPEAKER_00", 2: "SPEAKER_01", 3: "SPEAKER_02"},
    )
    # -- Custom Vosk testing
    # set_custom_vosk_config()
    # set_default_vosk_config()
