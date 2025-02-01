# utils/cli.py
# import os
from pathlib import Path

# from dotenv import set_key
import sys

from loguru import logger


def list_audio_files(workspace: Path, extension=".wav"):
    """Return a sorted list of all audio files with the given extension in the workspace."""
    return sorted(workspace.glob(f"*{extension}"))


def select_audio_file(workspace: Path, extension=".wav"):
    """Prompts the user to select an audio file from the workspace."""
    files = list_audio_files(workspace, extension)
    if not files:
        print(f"No audio files with extension '{extension}' found in {workspace}")
        sys.exit(1)
    logger.debug("Select an audio file:")
    for i, file in enumerate(files, start=1):
        logger.info(f"{i}. {file.name}")
    try:
        choice = int(input("Enter the number of the audio file to use: "))
        if choice < 1 or choice > len(files):
            raise ValueError("Invalid choice")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    return files[choice - 1]


def update_env_file(key: str, value: str, env_file: str = ".env"):
    """Updates the .env file with the provided key-value pair."""
    from dotenv import set_key

    env_path = Path(env_file)
    if not env_path.exists():
        env_path.write_text("")  # create empty .env if missing
    set_key(str(env_path), key, value)
    logger.debug(f"Updated {key} in {env_file} to '{value}'")
