import os
import csv
import json
from loguru import logger


def save_filtered_dialogue(speaker_transcription, audio_name, output_dir):
    filtered_dir = os.path.join(output_dir, str(audio_name))
    os.makedirs(filtered_dir, exist_ok=True)  # Ensure the directory exists

    filtered_file = os.path.join(filtered_dir, "filtered_transcription.txt")
    with open(filtered_file, "w", encoding="utf-8") as f:
        for entry in speaker_transcription:
            f.write(f"{entry['speaker']}: {entry['transcription']}\n")
    logger.debug(f"Filtered dialogue saved to: {filtered_file}")


def save_summary(summary, audio_name, output_dir):
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
    summary_dir = os.path.join(output_dir, audio_name)
    os.makedirs(summary_dir, exist_ok=True)

    # Define the summary file path
    summary_file = os.path.join(summary_dir, "summary.txt")

    # Handle different input types
    with open(summary_file, "w", encoding="utf-8") as f:
        if isinstance(summary, str):
            f.write(summary)  # Write the string directly
        elif isinstance(summary, list):
            if all(isinstance(item, str) for item in summary):
                f.write("\n".join(summary))  # Join list of strings with newlines
            elif all(isinstance(item, dict) for item in summary):
                # Convert list of dictionaries to a formatted string
                formatted_summary = []
                for item in summary:
                    if "speaker" in item and "transcription" in item:
                        formatted_summary.append(f"{item['speaker']}: {item['transcription']}")
                    else:
                        logger.warning(f"Skipping invalid dictionary item: {item}")
                f.write("\n".join(formatted_summary))  # Write formatted summary
            else:
                raise TypeError(f"Unsupported list item type in summary: {type(summary[0])}")
        else:
            raise TypeError(
                f"Unsupported summary type: {type(summary)}. Expected str, list, or list of dicts."
            )

    logger.debug(f"Discussion summary saved to: {summary_file}")


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

    # Save CSV
    csv_file = output_dir / f"{audio_name}.csv"
    # Save the results to a CSV file
    with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["speaker", "transcription", "start_time", "end_time"])
        writer.writeheader()
        for row in speaker_transcription:
            writer.writerow(row)

    # Save JSON
    json_file = output_dir / f"{audio_name}.json"
    # Save the results to a JSON file
    with open(json_file, mode="w", encoding="utf-8") as json_file:
        json.dump(speaker_transcription, json_file, ensure_ascii=False, indent=4)

    logger.info(f"Files saved: {csv_file}, {json_file}, {rttm_file}")


# def save_transcription_results(speaker_transcription, csv_file, json_file, rttm_file):
#     """
#     Saves the speaker transcription results to CSV and JSON files.
#     Args:
#         speaker_transcription (list): List of transcriptions for each speaker segment.
#     """
#     # Sort segments by start_time to ensure linear timeline
#     speaker_transcription.sort(key=lambda x: x["start_time"])

#     # Save the results to a CSV file
#     with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=["speaker", "transcription", "start_time", "end_time"])
#         writer.writeheader()
#         for row in speaker_transcription:
#             writer.writerow(row)

#     # Save the results to a JSON file
#     with open(json_file, mode="w", encoding="utf-8") as json_file:
#         json.dump(speaker_transcription, json_file, ensure_ascii=False, indent=4)

#     logger.info(f"Files saved: {csv_file}, {json_file}, {rttm_file}")
