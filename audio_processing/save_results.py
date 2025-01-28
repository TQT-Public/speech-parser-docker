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


def save_summary(speaker_transcription, audio_name, output_dir):
    summary_dir = os.path.join(output_dir, str(audio_name))
    os.makedirs(summary_dir, exist_ok=True)  # Ensure the directory exists

    summary_file = os.path.join(summary_dir, "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join([entry["transcription"] for entry in speaker_transcription]))
    logger.debug(f"Discussion summary saved to: {summary_file}")


def save_transcription_results(speaker_transcription, csv_file, json_file, rttm_file):
    """
    Saves the speaker transcription results to CSV and JSON files.
    Args:
        speaker_transcription (list): List of transcriptions for each speaker segment.
    """
    # Define file paths for saving the transcription results
    # csv_path = "./audio_files/ZOOM0067.csv"
    # json_path = "./audio_files/ZOOM0067.json"
    # rttm_file_path = "./audio_files/ZOOM0067.rttm"
    csv_path = csv_file
    json_path = json_file
    rttm_file_path = rttm_file

    # Save the results to a CSV file
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["speaker", "transcription", "start_time", "end_time"])
        writer.writeheader()
        for row in speaker_transcription:
            writer.writerow(row)

    # Save the results to a JSON file
    with open(json_path, mode="w", encoding="utf-8") as json_file:
        json.dump(speaker_transcription, json_file, ensure_ascii=False, indent=4)

    logger.info(f"Files saved: {csv_path}, {json_path}, {rttm_file_path}")
