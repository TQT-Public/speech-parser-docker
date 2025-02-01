# utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

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
BATCH_SIZE = float(os.getenv("BATCH_SIZE", "10.0"))

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
