import os
import csv
import json
from pathlib import Path
from loguru import logger

from speech_parser.utils.env_manager import EnvManager

# Initialize the environment using EnvManager
env_manager = EnvManager()


def save_filtered_dialogue(speaker_transcription, audio_name, output_dir):
    filtered_dir = os.path.join(output_dir, str(audio_name))
    os.makedirs(filtered_dir, exist_ok=True)  # Ensure the directory exists

    filtered_file = os.path.join(filtered_dir, "filtered_transcription.txt")
    with open(filtered_file, "w", encoding="utf-8") as f:
        for entry in speaker_transcription:
            f.write(f"{entry['speaker']}: {entry['transcription']}\n")
    logger.debug(f"Filtered dialogue saved to: {filtered_file}")


def save_summary(summary, audio_name, output_dir, ai_model_key="google-gemini", translated=False):
    """
    Save the summary of the dialogue to a file.

    Args:
        summary (str, list, or list of dicts): The generated summary.
        audio_name (str): Name of the audio file (without extension).
        output_dir (str): Directory to save the summary.
    """
    # Remove file extension from audio_name
    audio_name = os.path.splitext(audio_name)[0]

    # Create the output directory if it doesn't exist
    addition = "-batches" if env_manager.get_bool("USE_BATCHES") is True else "-split"
    summary_dir = Path(output_dir, f"{Path(audio_name).stem}{addition}")
    # summary_dir = Path(output_dir, audio_name, addition)
    os.makedirs(summary_dir, exist_ok=True)

    # Define the summary file path
    if not translated:
        summary_file = Path(summary_dir.resolve(), f"{ai_model_key}_summary.md")
    else:
        summary_file = Path(summary_dir.resolve(), f"{ai_model_key}_summary_russian.md")

    # Handle different input types
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# Resume\n\n")  # Start line for .md
        if isinstance(summary, str):
            f.write(summary)  # Write the string directly
        elif isinstance(summary, list):
            if all(isinstance(item, str) for item in summary):
                f.write("\n".join(summary))  # Join list of strings with newlines
            elif all(isinstance(item, dict) for item in summary):
                # Convert list of dictionaries to a formatted string
                formatted_summary = []
                for item in summary:
                    # Adjusting this section to properly handle summary entries
                    if "transcription" in item:
                        formatted_summary.append(item["transcription"])
                    else:
                        logger.warning(f"Skipping invalid dictionary item: {item}")
                f.write("\n".join(formatted_summary))  # Write formatted summary
            else:
                raise TypeError(f"Unsupported list item type in summary: {type(summary[0])}")
        else:
            raise TypeError(
                f"Unsupported summary type: {type(summary)}. Expected str, list, or list of dicts."
            )

    logger.debug(f"Discussion summary made by {ai_model_key} saved to: {summary_file}")


def save_transcription_results(speaker_transcription, audio_name, output_dir, csv_file, json_file, rttm_file):
    """
    Save transcription results to CSV, JSON, and RTTM files.
    Args:
        transcription_data (list): List of transcription results.
        audio_name (str): Name of the audio file (without extension).
        output_dir (Path): Directory to save the results.
    """
    # Sort segments by start_time to ensure linear timeline
    speaker_transcription.sort(key=lambda x: x["start_time"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log the transcription data
    logger.debug(f"Transcription Data: {speaker_transcription}")

    # Ensure no empty transcriptions are saved
    speaker_transcription = [segment for segment in speaker_transcription if segment.get("transcription")]

    # Check if transcription is empty before saving
    if not speaker_transcription:
        logger.warning("No valid transcriptions found. Not saving empty results.")
        return

    # Save CSV
    csv_file = output_dir / f"{audio_name}.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["speaker", "transcription", "start_time", "end_time"])
        writer.writeheader()
        for row in speaker_transcription:
            writer.writerow(row)

    # Save JSON
    json_file = output_dir / f"{audio_name}.json"
    with open(json_file, mode="w", encoding="utf-8") as json_file:
        json.dump(speaker_transcription, json_file, ensure_ascii=False, indent=4)

    logger.info(f"Files saved: {csv_file}, {json_file}, {rttm_file}")
