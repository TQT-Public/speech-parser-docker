# from pathlib import Path
import subprocess

# import os
from loguru import logger
from dotenv import load_dotenv

from speech_parser.utils.env import env_as_int

# Load environment variables from .env
load_dotenv()

VOSK_SAMPLE_RATE = env_as_int("VOSK_SAMPLE_RATE", 16000)


def convert_wav_to_mp3(audio_file):
    """
    Convert a WAV file to MP3 using ffmpeg.
    Args:
        wav_file_path (str): Path to the WAV file to convert.
    """
    output_file = str(audio_file)
    output_path = output_file.replace(".wav", ".mp3")
    print(f"End-point: {output_path}")

    command = ["ffmpeg", "-i", str(audio_file), "-codec:a", "libmp3lame", "-b:a", "192k", output_path]
    subprocess.run(command, check=True)
    logger.info(f"Converted {str(audio_file)} to {output_path}")


def convert_wav_to_pcm(audio_file):
    """Convert WAV file to 16kHz 16-bit mono PCM."""
    output_file = audio_file.stem + "_converted.wav"
    output_path = audio_file.parent / output_file
    command = [
        "ffmpeg",
        "-y",  # Add this flag to automatically overwrite existing files #NOTE: remove
        "-i",
        str(audio_file),
        "-ac",
        "1",  # Mono
        "-ar",
        str(VOSK_SAMPLE_RATE),  # Use sample rate from .env - default 16kHz
        "-sample_fmt",
        "s16",  # 16-bit
        str(output_path),
    ]
    subprocess.run(command, check=True)
    print(f"Conversion complete: {output_path}")
    return output_path


if __name__ == "__main__":
    # $ python -m audio_processing.convert_audio
    audio_file = "ZOOM0068.wav"
    audio_path = f"{audio_file[:-4]}_converted.wav"
    end_path = f"./audio_files/{str(audio_path)}"
    print(f"Target: {end_path}")
    convert_wav_to_mp3(end_path)
