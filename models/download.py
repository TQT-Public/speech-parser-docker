import os
from pathlib import Path
import requests
import zipfile
from loguru import logger

# Define constants
# VOSK_MODEL_DIR = Path("C:/TQT-DEV/ext/speech-parser/models")
VOSK_MODEL_DIR = Path(".")
RUSSIAN_GPU_MODEL = "vosk-model-ru-gpu"  # Change with actual model pairs
ENGLISH_GPU_MODEL = "vosk-model-en-gpu"  # Use list_vosk() function provided


def download_vosk_model(model_name: str, model_path: Path) -> Path:
    """
    Downloads and extracts the specified Vosk model if it doesn't already exist.

    Args:
        model_name (str): Name of the Vosk model to download.
        model_path (Path): Path where the model should be saved.

    Returns:
        Path: Path to the downloaded/extracted model.
    """
    # Check if the model already exists
    model_dir = model_path / model_name
    if model_dir.exists():
        logger.info(f"Model {model_name} already exists at {model_dir}.")
        return model_dir

    # Create the directory for models if it doesn't exist
    model_path.mkdir(parents=True, exist_ok=True)

    # Define the download URL
    model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    model_zip = model_path / f"{model_name}.zip"

    try:
        # Download the model
        logger.info(f"Downloading model {model_name} from {model_url}...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # Raise an error if the request failed

        with open(model_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        logger.info(f"Model {model_name} downloaded successfully to {model_zip}.")

        # Extract the model
        logger.info(f"Extracting model {model_name}...")
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall(model_path)
        logger.info(f"Model {model_name} extracted to {model_dir}.")

        # Clean up the zip file
        model_zip.unlink()
        logger.info(f"Cleaned up zip file: {model_zip}.")

    except Exception as e:
        logger.error(f"Failed to download or extract model {model_name}: {e}")
        raise

    return model_dir


def download_russian_model():
    """Download the GPU-enabled Russian Vosk model."""
    logger.info("Downloading Russian Vosk GPU model...")
    return download_vosk_model(RUSSIAN_GPU_MODEL, VOSK_MODEL_DIR)


def download_english_model():
    """Download the GPU-enabled English Vosk model (not used in the project)."""
    logger.info("Downloading English Vosk GPU model...")
    return download_vosk_model(ENGLISH_GPU_MODEL, VOSK_MODEL_DIR)


if __name__ == "__main__":
    # Main program execution
    try:
        russian_model_path = download_russian_model()
        logger.success(f"Russian GPU-enabled model is ready at {russian_model_path}.")
    except Exception as e:
        logger.error(f"Failed to prepare Russian Vosk GPU model: {e}")
