import os
import subprocess
import sys
import requests
import zipfile
import time
from pathlib import Path
from multiprocessing import Pool
from pydub.utils import mediainfo


# Set workspace and other constants
WORKSPACE = Path("D:/A-Samples/recorder-H1N1/2024-witcher-miro")
VOSK_MODEL_PATH = Path("C:/TQT-DEV/ext/speech-parser")
VOSK_TRANSCRIBER = "vosk-transcriber"  # Ensure vosk-transcriber is installed and in PATH
THRESHOLD_WAV_DUR = 30  # Minutes
RUN_AFTER = True
OUTPUT_DIR = VOSK_MODEL_PATH / "output1"  # Directory to save output files

# Dynamically set the audio file name
audio_file_name = "ZOOM0067_converted.WAV"


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


def convert_wav_to_pcm(audio_file):
    """Convert WAV file to 16kHz 16-bit mono PCM."""
    output_file = audio_file.stem + "_converted.wav"
    output_path = audio_file.parent / output_file
    command = [
        "ffmpeg",
        "-i",
        str(audio_file),
        "-ac",
        "1",  # Mono
        "-ar",
        "16000",  # 16 kHz
        "-sample_fmt",
        "s16",  # 16-bit
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def process_audio_file(audio_file, model_path, part_num):
    """Process the audio file with Vosk transcriber."""
    print(f"Processing audio file: {audio_file}")
    command = ["python", "speerecon1.py", str(audio_file), str(model_path)]
    subprocess.run(command, check=True)

    # After processing, handle the output files (summary, filtered)
    summary_file = OUTPUT_DIR / f"{audio_file.stem}_part{part_num}_summary.txt"
    filtered_file = OUTPUT_DIR / f"{audio_file.stem}_part{part_num}_filtered.txt"

    print(f"Filtered dialogue saved to: {filtered_file}")
    print(f"Discussion summary saved to: {summary_file}")

    return summary_file, filtered_file


def combine_output_files(audio_file_stem):
    """Combine part summary and filtered files into final output using the correct file stem."""
    final_summary = os.path.join(OUTPUT_DIR, "final_summary.txt")
    final_filtered = os.path.join(OUTPUT_DIR, "final_filtered.txt")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    try:
        with open(final_summary, "w", encoding="utf-8") as summary_out:
            for part in range(1, 13):  # Assuming 12 parts
                part_file = os.path.join(OUTPUT_DIR, f"{audio_file_stem}_part{part}_summary.txt")
                if os.path.exists(part_file):
                    with open(part_file, "r", encoding="utf-8") as part_file_content:
                        summary_out.write(part_file_content.read())
                    print(f"Added {part_file} to final summary.")

        with open(final_filtered, "w", encoding="utf-8") as filtered_out:
            for part in range(1, 13):  # Assuming 12 parts
                part_file = os.path.join(OUTPUT_DIR, f"{audio_file_stem}_part{part}_filtered.txt")
                if os.path.exists(part_file):
                    with open(part_file, "r", encoding="utf-8") as part_file_content:
                        filtered_out.write(part_file_content.read())
                    print(f"Added {part_file} to final filtered dialogue.")
    except Exception as e:
        print(f"An error occurred while combining files: {e}")


def split_audio_file(audio_file):
    """Split the audio file if it's longer than the threshold."""
    print(f"Checking duration of {audio_file}...")
    info = mediainfo(str(audio_file))
    duration = float(info["duration"]) / 60  # Duration in minutes

    if duration >= THRESHOLD_WAV_DUR:
        print(f"Audio file duration is {duration:.2f} minutes, splitting...")
        # Split the audio file in half iteratively
        parts = []
        part_num = 1
        while duration >= THRESHOLD_WAV_DUR:
            part1 = audio_file.parent / f"{audio_file.stem}_part{part_num}.wav"
            part2 = audio_file.parent / f"{audio_file.stem}_part{part_num+1}.wav"
            split_command = ["ffmpeg", "-i", str(audio_file), "-t", str(duration / 2), str(part1)]
            subprocess.run(split_command, check=True)
            split_command = ["ffmpeg", "-i", str(audio_file), "-ss", str(duration / 2), str(part2)]
            subprocess.run(split_command, check=True)
            parts = [part1, part2]
            audio_file = part1
            part_num += 2
            duration = float(mediainfo(str(audio_file))["duration"]) / 60
        return parts
    else:
        return [audio_file]


def run_parallel_processing(audio_files, model_path):
    """Run the Vosk transcription in parallel for multiple files."""
    with Pool(4) as pool:
        results = pool.starmap(
            process_audio_file, [(file, model_path, idx + 1) for idx, file in enumerate(audio_files)]
        )
    return results


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
        print(f"Model {selected_model} already exists.")

    # Use the correct audio file from the dynamic name
    audio_file = WORKSPACE / audio_file_name
    converted_audio_file = convert_wav_to_pcm(audio_file)

    if RUN_AFTER:
        audio_files_to_process = split_audio_file(converted_audio_file)
        results = run_parallel_processing(audio_files_to_process, model_path)

        # Combine results into final summary and filtered files
        combine_output_files(converted_audio_file.stem)


if __name__ == "__main__":
    main()
