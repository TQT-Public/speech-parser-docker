import os
from pathlib import Path
import subprocess
import sys
from loguru import logger
import requests
import zipfile
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

VOSK_MODEL_PATH = Path(os.getenv("VOSK_MODEL_PATH"))
VOSK_TRANSCRIBER = "vosk-transcriber"  # Ensure vosk-transcriber is installed and in PATH


def check_and_download_model(model_name, model_path):
    """Check if the Vosk model exists locally, if not, download and extract it."""
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found locally. Downloading...")
        download_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
        model_zip = os.path.join(model_path.parent, f"{model_name}.zip")

        response = requests.get(download_url, stream=True)
        with open(model_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Model {model_name} downloaded.")

        # Extract the downloaded model
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall(model_path.parent)
        print(f"Model {model_name} extracted.")
        print(f"Model Path: {model_path}")

        # Clean up the zip file after extraction
        os.remove(model_zip)
    else:
        print(f"Model {model_name} already exists.")
        print(f"Model Path: {model_path}")

    return model_path


def list_vosk_models():
    """Execute the command to list available Vosk models."""
    print("Fetching available Vosk models...")
    result = subprocess.run(
        [VOSK_TRANSCRIBER, "--list-models"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(f"Error listing models: {result.stderr.decode()}")
        sys.exit(1)

    models = result.stdout.decode().splitlines()
    print("Available Vosk models:")
    for idx, model in enumerate(models):
        print(f"{idx + 1}. {model}")
    return models


def download_vosk_model(model_name, download_path):
    """Download and extract a selected Vosk model."""
    model_url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    print(f"Downloading model {model_name}...")
    response = requests.get(model_url, stream=True)

    if response.status_code == 200:
        model_zip = download_path / f"{model_name}.zip"
        with open(model_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Model {model_name} downloaded successfully.")

        # Extract the downloaded model
        print(f"Extracting {model_name}...")
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall(download_path)
        print(f"Model {model_name} extracted.")

        # Clean up the zip file after extraction
        model_zip.unlink()
        return download_path / model_name
    else:
        print(f"Failed to download model {model_name}.")
        sys.exit(1)


def main():
    models = list_vosk_models()
    choice = int(input(f"Choose a model (1-{len(models)}): "))
    if not (1 <= choice <= len(models)):
        sys.exit(1)

    selected_model = models[choice - 1]
    model_path = VOSK_MODEL_PATH / selected_model

    if not model_path.exists():
        model_path = download_vosk_model(selected_model, VOSK_MODEL_PATH)
    else:
        logger.info(f"Model {selected_model} already exists.")

    logger.debug(
        f"New model is downloaded or existing one used \n \
                 Make sure to set VOSK_MODEL_PATH to {model_path} \n \
                     model_name to {selected_model}"
    )


if __name__ == "__main__":
    #  $ python -m models.download_model
    main()
