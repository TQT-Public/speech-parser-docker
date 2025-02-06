# punctuation.py
from loguru import logger
from speech_analyzer.csv_loader import format_dialogue_for_summary, load_csv_data_for_model
from speech_parser.utils.env import env_as_bool
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
    model_url = model_conf.get("package")
    model_dir = "./models/silero"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, os.path.basename(model_url))

    if not os.path.isfile(model_path):
        torch.hub.download_url_to_file(model_url, model_path, progress=True)

    imp = package.PackageImporter(model_path)
    model = imp.load_pickle("te_model", "model")
    logger.info("Silero model loaded successfully")

    return model


# def add_punctuation_to_transcription(transcription_text, model, lan="ru"):
#     """
#     Apply punctuation restoration model to the transcription.

#     Args:
#         transcription_text (str): The transcription text to apply punctuation.
#         model (object): The Silero model loaded.
#         lan (str): Language to apply the punctuation model (default is 'ru').

#     Returns:
#         str: Transcription text with restored punctuation, cleaned of any `[UNK]` tokens.
#     """
#     output_text = model.enhance_text(transcription_text, lan)
#     output_text = output_text.replace("\[UNK\]", ".").replace(". .", ". \n").strip()

#     return output_text


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

    # Replace unknown tokens with a period or empty space
    output_text = output_text.replace("[UNK]", ".").strip()
    output_text = output_text.replace(". .", ". \n").strip()

    return output_text


def add_punctuation(csv_file, save_as_new=True):
    """
    Process a CSV file to add punctuation to the transcription text.

    Args:
        csv_file (Path): The CSV file containing transcription data.
        save_as_new (bool): If True, save the output as a new file.
    """
    if not env_as_bool("ADD_PUNCTUATION", True):
        logger.info("ADD_PUNCTUATION is disabled, skipping punctuation process.")
        return

    model = example_silero()  # Load Silero model

    # Load transcription data from CSV
    df = load_csv_data_for_model(csv_file)
    dialogue_text = format_dialogue_for_summary(df)

    # Apply punctuation restoration
    transcription_with_punctuation = add_punctuation_to_transcription(dialogue_text, model)
    punctuated_lines = transcription_with_punctuation.split(". \n")

    # Handle the case where lengths don't match
    if len(punctuated_lines) != len(df):
        logger.warning(
            f"Length mismatch: {len(punctuated_lines)} punctuated lines, but {len(df)} rows in the CSV."
        )
        if len(punctuated_lines) > len(df):
            punctuated_lines = punctuated_lines[: len(df)]  # Truncate excess lines
        else:
            punctuated_lines.extend([""] * (len(df) - len(punctuated_lines)))  # Pad with empty strings

    # Save the CSV with punctuation
    df["transcription"] = punctuated_lines
    if save_as_new:
        punctuated_csv_file = csv_file.with_name(f"{csv_file.stem}_punctuated.csv")
        df.to_csv(punctuated_csv_file, index=False)
        logger.info(f"Punctuated transcription saved to {punctuated_csv_file}")
        return punctuated_csv_file
    else:
        df.to_csv(csv_file, index=False)
        logger.info(f"Punctuated transcription saved to {csv_file}")
        return csv_file


if __name__ == "__main__":
    # python -m speech_analyzer.csv_loader
    csv_file_path = "./audio_files/ZOOM0067.csv"
    # Test with a sample CSV
    # add_punctuation("path/to/your/csv/file.csv")
    punctuated_csv_file = add_punctuation(csv_file_path, save_as_new=True)
    logger.debug(f"CSV: {punctuated_csv_file}")
