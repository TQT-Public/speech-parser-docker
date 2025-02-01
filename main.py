import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

from speech_analyzer.csv_loader import load_csv_data_for_model
from speech_analyzer.dialogue_analyzer import analyze_dialogue
from speech_analyzer.gpt_loader import load_gpt_model
from speech_analyzer.model_loader import (
    # load_all_models,
    load_stable_diffusion_model,
    load_vosk_model,
    unsloth_model_loader,
)
from speech_parser import speech_parser, env_as_bool, env_as_str, env_as_path
from speech_parser.audio_processing.save_results import save_summary

import csv
import json
import pandas as pd


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


def create_empty_csv_and_json_if_not_exists(csv_file_path, json_file_path):
    """
    Create empty CSV and JSON files if they do not exist.

    Args:
        csv_file_path (str): Path to the CSV file.
        json_file_path (str): Path to the JSON file.
    """
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["speaker", "transcription", "start_time", "end_time"])
        logger.info(f"Created empty CSV file: {csv_file_path}")

    if not os.path.exists(json_file_path):
        with open(json_file_path, mode="w", encoding="utf-8") as json_file:
            json.dump([], json_file, ensure_ascii=False, indent=4)
        logger.info(f"Created empty JSON file: {json_file_path}")


def create_empty_csv_if_not_exists(csv_file_path):
    """
    Creates an empty CSV with the appropriate headers if the file doesn't exist.
    Args:
        csv_file_path (str): Path to the CSV file to check or create.
    """
    if not os.path.exists(csv_file_path):
        # Define headers matching your data schema
        headers = ["speaker", "transcription", "start_time", "end_time"]
        empty_df = pd.DataFrame(columns=headers)
        empty_df.to_csv(csv_file_path, index=False)
        print(f"Created empty CSV file: {csv_file_path}")
    else:
        print(f"CSV file already exists: {csv_file_path}")


def load_model_by_key(model_name_key):
    """
    Loads the corresponding model and tokenizer based on the model_name_key.

    Parameters:
    - model_name_key (str): The key for the model (e.g., "mistral", "llama", "deepseek").
    - models (dict): A dictionary of all loaded models.

    Returns:
    - tuple: (model, tokenizer) corresponding to the model_name_key.
    """
    # TODO: have list of available models keys as a config - pydantic, для простых конфигов что-то более подходящее типа гидры или pydantic
    available_models = [
        "gpt-4",
        "gpt-3.5",
        "vosk",
        "llama",
        "mistral",
        "falcon",
        "deepseek",
        "stable_diffusion",
    ]
    if model_name_key not in available_models:
        raise ValueError(f"Invalid model key: {model_name_key}. Available keys: {available_models}")

    if model_name_key in ["llama", "mistral", "falcon", "deepseek"]:
        model, tokenizer = unsloth_model_loader(model_name_key)
    elif model_name_key in ["gpt-4", "gpt-3.5"]:
        model, tokenizer = load_gpt_model(model_name_key)
    elif model_name_key == "stable_diffusion":
        model = load_stable_diffusion_model()
        tokenizer = None  # Stable Diffusion doesn’t use a tokenizer
    elif model_name_key == "vosk":
        model = load_vosk_model()
        tokenizer = None

    return model, tokenizer


def main():
    load_dotenv(".env")

    DRY_RUN = env_as_bool("DRY_RUN", False)
    # AUDIOWORKSPACE = env_as_path("AUDIOWORKSPACE", "./audio_parts")
    AUDIO_FILE_NAME = env_as_path("AUDIO_FILE_NAME", "ZOOM0067.wav")
    OUTPUT_DIR = env_as_path("OUTPUT_DIR", "./output")
    ENABLE_AI = env_as_bool("ENABLE_AI", "True")

    # csv_file = Path(AUDIOWORKSPACE) / f"{AUDIO_FILE_NAME.stem}.csv"
    # json_file = Path(AUDIOWORKSPACE) / f"{AUDIO_FILE_NAME.stem}.json"
    csv_file = Path(OUTPUT_DIR) / f"{AUDIO_FILE_NAME.stem}.csv"
    json_file = Path(OUTPUT_DIR) / f"{AUDIO_FILE_NAME.stem}.json"

    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"CSV file path: {csv_file}")

    if csv_file.exists():
        logger.debug(f"File exists: {csv_file}")
    else:
        logger.error(f"File does not exist: {csv_file}")
        create_empty_csv_and_json_if_not_exists(csv_file, json_file)

    if not DRY_RUN:
        csv_file = speech_parser.speech_parser()
    elif not csv_file.exists():
        logger.error(f"CSV file {csv_file} not found.")
        return
    else:
        logger.info(f"Files for {csv_file} already exist. Loading file...")

        # Load CSV data
        df = load_csv_data_for_model(csv_file, show_stats=True)
        logger.debug(df.describe())

    if ENABLE_AI:
        start_time = datetime.datetime.now()
        logger.debug(f"AI analysis - Start Time: {start_time}")
        ai_model_key = env_as_str("AI_MODEL_NAME", "gpt3.5")

        try:
            model, tokenizer = load_model_by_key(ai_model_key)
            logger.info(f"Loaded model: {ai_model_key}")
        except ValueError as e:
            logger.error(f"Model loading failed: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return

        # Format dialogue and summarize
        # dialogue_text = format_dialogue_for_summary(df)
        # summary = generate_summary(dialogue_text, model, tokenizer)
        summary = analyze_dialogue(csv_file, model, tokenizer)

        if isinstance(summary, str):
            summary = [{"transcription": summary}]

        save_summary(summary, AUDIO_FILE_NAME, OUTPUT_DIR, ai_model_key)
        logger.debug(f"Summary saved at {OUTPUT_DIR}\nModel AI used: {ai_model_key}")

        end_time = datetime.datetime.now()
        logger.debug(f"AI - End Time: {end_time}")
        total_runtime = end_time - start_time
        logger.debug(f"AI - Total run time: {total_runtime}")


if __name__ == "__main__":
    main()
