import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from speech_analyzer.dialogue_analyzer import analyze_dialogue
from speech_analyzer.model_loader import (
    # load_all_models,
    load_stable_diffusion_model,
    load_vosk_model,
    unsloth_model_loader,
)
from speech_parser import speech_parser, env_as_bool, env_as_str, env_as_path, filename_as_csv
from speech_parser.audio_processing.save_results import save_summary


def load_model_by_key(model_name_key):
    """
    Loads the corresponding model and tokenizer based on the model_name_key.

    Parameters:
    - model_name_key (str): The key for the model (e.g., "mistral", "llama", "deepseek").
    - models (dict): A dictionary of all loaded models.

    Returns:
    - tuple: (model, tokenizer) corresponding to the model_name_key.
    """
    available_models = ["vosk", "llama", "mistral", "falcon", "deepseek", "stable_diffusion"]
    if model_name_key not in available_models:
        raise ValueError(f"Invalid model key: {model_name_key}. Available keys: {available_models}")

    if model_name_key in ["llama", "mistral", "falcon", "deepseek"]:
        model, tokenizer = unsloth_model_loader(model_name_key)
    elif model_name_key == "stable_diffusion":
        model = load_stable_diffusion_model()
        tokenizer = None  # Stable Diffusion doesn’t use a tokenizer
    elif model_name_key == "vosk":
        model = load_vosk_model()
        tokenizer = None

    return model, tokenizer


def main():
    load_dotenv()
    DRY_RUN = env_as_bool("DRY_RUN", False)
    AUDIOWORKSPACE_ENV = env_as_path("AUDIOWORKSPACE", "./audio_parts")
    AUDIO_FILE_NAME = env_as_str("AUDIO_FILE_NAME", "ZOOM0067.wav")
    OUTPUT_DIR = env_as_path("OUTPUT_DIR", "./output")
    ENABLE_AI = env_as_bool("ENABLE_AI", "True")

    # Construct the full path to the CSV file
    csv_file = Path(AUDIOWORKSPACE_ENV) / filename_as_csv(AUDIO_FILE_NAME)
    csv_file = csv_file.resolve()  # Convert to absolute path

    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"CSV file path: {csv_file}")

    # Debug: Check if the file exists
    if csv_file.exists():
        logger.debug(f"File exists: {csv_file}")
    else:
        logger.error(f"File does not exist: {csv_file}")
        return

    if not DRY_RUN:
        csv_file = speech_parser.speech_parser()
    elif not csv_file.exists():
        logger.error(f"CSV file {csv_file} not found.")
        return
    else:
        logger.info(f"Files for {csv_file} already exist. Loading file...")

    if ENABLE_AI:
        ai_model_key = env_as_str("AI_MODEL_NAME", "mistral")  # Default to "mistral" if not specified

        try:
            model, tokenizer = load_model_by_key(ai_model_key)
            logger.info(f"Loaded model: {ai_model_key}")
            # models = load_all_models()
            # model, tokenizer = load_model_by_key(ai_model_key, models)
        except ValueError as e:
            logger.error(f"Model loading failed: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return

        summary = analyze_dialogue(csv_file, model, tokenizer)

        # Ensure summary is a list of dictionaries
        if isinstance(summary, str):
            summary = [{"transcription": summary}]  # Wrap the summary in a list of dictionaries

        save_summary(summary, AUDIO_FILE_NAME, OUTPUT_DIR)
        logger.info(f"Summary saved at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
