from loguru import logger

# from silero import apply_punctuation
from speech_analyzer.csv_loader import format_dialogue_for_summary, load_csv_data_for_model
from speech_parser.utils.cli import select_audio_file, update_env_file
from speech_parser.utils.env import env_as_path

import os
import yaml
import torch
from torch import package


def example_silero():
    """
    Downloads the Silero model and applies punctuation to the text.
    """
    torch.hub.download_url_to_file(
        "https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml",
        "latest_silero_models.yml",
        progress=False,
    )

    with open("latest_silero_models.yml", "r", encoding="utf-8") as yaml_file:
        models = yaml.load(yaml_file, Loader=yaml.SafeLoader)

    model_conf = models.get("te_models").get("latest")
    available_languages = list(model_conf.get("languages"))
    logger.debug(f"Available languages: {available_languages}")

    available_punct = list(model_conf.get("punct"))
    logger.info(f"Available punctuation marks: {available_punct}")

    model_url = model_conf.get("package")
    model_dir = "./models/silero"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, os.path.basename(model_url))

    if not os.path.isfile(model_path):
        torch.hub.download_url_to_file(model_url, model_path, progress=True)

    imp = package.PackageImporter(model_path)
    model = imp.load_pickle("te_model", "model")
    logger.info(model.examples)

    return model


def add_punctuation_to_transcription(transcription_text, model, lan="ru"):
    """
    Apply punctuation restoration model to the transcription.

    Args:
        transcription_text (str): The transcription text to apply punctuation.
        model (object): The Silero model loaded.
        lan (str): Language to apply the punctuation model (default is 'ru').

    Returns:
        str: Transcription text with restored punctuation, cleaned of any `[UNK]` tokens.
    """
    output_text = model.enhance_text(transcription_text, lan)

    # Remove unknown tokens if any appear - silero add [UNK]
    output_text = output_text.replace("[UNK]", ".").strip()
    output_text = output_text.replace(". .", ". \n").strip()

    return output_text
    return model.enhance_text(transcription_text, lan)


if __name__ == "__main__":
    # Python script entry point
    model = example_silero()  # Load the Silero model

    # CLI: Let user choose an audio file from WORKSPACE
    workspace_path = env_as_path("WORKSPACE", "./sources")
    selected_audio = select_audio_file(workspace_path)
    update_env_file("AUDIO_FILE_NAME", selected_audio.name)  # Update .env file
    logger.info(f"Selected audio file: {selected_audio}")

    # Construct full paths
    audio_file = workspace_path / selected_audio.name
    AUDIOWORKSPACE = env_as_path("AUDIOWORKSPACE", "./audio_parts")
    csv_file = AUDIOWORKSPACE / f"{audio_file.stem}.csv"

    # Usage in transcription flow
    df = load_csv_data_for_model(str(csv_file))
    dialogue_text = format_dialogue_for_summary(df)

    # Apply punctuation to the transcription using the model
    transcription = add_punctuation_to_transcription(dialogue_text, model, lan="ru")

    # Log the result
    logger.info(f"\nInput text without: \n{dialogue_text}\n")
    logger.debug(f"\nText with punctuation: {transcription}")
    # TODO: add saving of punctuation CSV
