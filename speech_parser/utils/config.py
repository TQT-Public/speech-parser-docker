# utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

from speech_parser.utils.env import env_as_float, env_as_int

# Load .env file (from project root)
load_dotenv()


def env_as_bool(key, default=False):
    return os.getenv(key, str(default)).lower() == "true"


def env_as_str(key, default=""):
    return os.getenv(key, default)


def env_as_path(key, default="."):
    return Path(os.getenv(key, default))


# General settings
DRY_RUN = env_as_bool("DRY_RUN", False)
ENABLE_AI = env_as_bool("ENABLE_AI", True)
USE_BATCHES = env_as_bool("USE_BATCHES", False)
BATCH_SIZE = env_as_float("BATCH_SIZE", "10.0")
TOKEN_LIMIT = env_as_int("TOKEN_LIMIT", "4096")

# Audio and output paths
AUDIOWORKSPACE = env_as_path("AUDIOWORKSPACE", "./audio_parts")
WORKSPACE = env_as_path("WORKSPACE", "./sources")
AUDIO_FILE_NAME = env_as_str("AUDIO_FILE_NAME", "ZOOM0067.wav")
OUTPUT_DIR = env_as_path("OUTPUT_DIR", "./output")
OUTPUT_DIR_PARTS = env_as_path("OUTPUT_DIR_PARTS", "./audio_files/parts")

# Vosk model configuration
VOSK_MODEL_PATH = env_as_path("VOSK_MODEL_PATH", "./models")
MODEL_NAME = env_as_str("MODEL_NAME", "vosk-model-ru-0.42")
VOSK_MODEL_FULL_PATH = VOSK_MODEL_PATH / MODEL_NAME

# AI settings
AI_MODEL_NAME = env_as_str("AI_MODEL_NAME", "gpt-4")

# # Use helper functions to convert environment variables to proper types.
# # (env_as_bool, env_as_path, etc. are assumed to be defined in your config module)
# from speech_parser.utils.config import (
#     env_as_bool,
#     env_as_path,
#     env_as_str,
# )
# DRY_RUN = env_as_bool("DRY_RUN", False)
# AUDIOWORKSPACE = env_as_path("AUDIOWORKSPACE", "./audio_parts")
# AUDIO_FILE_NAME = env_as_str("AUDIO_FILE_NAME", "ZOOM0067.wav")
# OUTPUT_DIR = env_as_path("OUTPUT_DIR", "./output")
# ENABLE_AI = env_as_bool("ENABLE_AI", True)
# MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", "3"))
# USE_BATCHES = env_as_bool("USE_BATCHES", False)
# BATCH_SIZE = float(os.getenv("BATCH_SIZE", "10.0"))
# AI_MODEL_NAME = env_as_str("AI_MODEL_NAME", "gpt-3.5-turbo")
# TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT", "4096"))  # For example, 4096 tokens

# # CLI: Let user choose an audio file from WORKSPACE
# workspace_path = env_as_path("WORKSPACE", "./sources")
# selected_audio = choose_audio_file(workspace_path)
# update_env_file("AUDIO_FILE_NAME", selected_audio.name)  # Update .env file
# logger.info(f"Selected audio file: {selected_audio}")

# # Construct full paths
# audio_file = workspace_path / selected_audio.name
# output_dir = Path(OUTPUT_DIR)
# output_dir_parts = Path(OUTPUT_DIR_PARTS)
# output_dir.mkdir(parents=True, exist_ok=True)
# output_dir_parts.mkdir(parents=True, exist_ok=True)
# # AUDIOWORKSPACE is used for storing CSV/JSON results
# AUDIOWORKSPACE = env_as_path("AUDIOWORKSPACE", "./audio_parts")
# converted_file_path = AUDIOWORKSPACE / f"{audio_file.stem}_converted.wav"
