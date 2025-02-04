# import multiprocessing
import os
import json
from pathlib import Path
import wave

import torch
from tqdm import tqdm
import vosk

from pydub import AudioSegment

from loguru import logger
from dotenv import load_dotenv

from vosk import SetLogLevel

from speech_parser.utils.env import env_as_bool, env_as_int, env_as_path, env_as_str

torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat32
# Load environment variables from .env
load_dotenv()

# # Load variables
AUDIOWORKSPACE_ENV = env_as_path("AUDIOWORKSPACE", "./audio_parts")
OUTPUT_DIR_PARTS_ENV = env_as_path("OUTPUT_DIR_PARTS" "./audio_parts/parts")

VOSK_MODEL_PATH_ENV = env_as_path("VOSK_MODEL_PATH", "./models/vosk")
MODEL_NAME = env_as_str("MODEL_NAME")
VOSK_MODEL_FULL_PATH_ENV = Path(str(VOSK_MODEL_PATH_ENV))
logger.debug(f"Using model: {str(VOSK_MODEL_FULL_PATH_ENV)}")

SetLogLevel(-1)  # Disable logs

ENABLE_VOSK_LOGS = env_as_bool("ENABLE_VOSK_LOGS", "True")
ENABLE_VOSK_GPU = env_as_bool("ENABLE_VOSK_GPU", "True")
FASTER_DECODING = env_as_bool("FASTER_DECODING", "True")

VOSK_MAX_CHUNK_SIZE = env_as_int("VOSK_MAX_CHUNK_SIZE", 10000)
VOSK_SAMPLE_RATE = env_as_int("VOSK_SAMPLE_RATE", 16000)

VOSK_MAX_CHUNK_SIZE = env_as_int("VOSK_MAX_CHUNK_SIZE", 10000)
VOSK_SAMPLE_RATE = env_as_int("VOSK_SAMPLE_RATE", 16000)


# Function to set up Vosk logging
def setup_vosk_logging():
    if ENABLE_VOSK_LOGS:
        vosk.SetLogLevel(0)  # Enable logs
        logger.debug("Vosk logs enabled.")
    else:
        vosk.SetLogLevel(-1)  # Disable logs
        logger.debug("Vosk logs disabled.")


# Function to set up GPU (if available and enabled)
def setup_vosk_gpu(model_path):
    if torch.cuda.is_available():
        logger.info("CUDA is available: GPU detected.")
    else:
        logger.debug(f"No GPU found! Running on CPU. at {model_path}")

    if ENABLE_VOSK_GPU:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            logger.debug("CUDA not detected. Please ensure you're running with GPU support.")
        else:
            logger.info("CUDA detected. GPU mode is enabled for Vosk.")
        rec = vosk.KaldiRecognizer(vosk.Model(str(model_path)), VOSK_SAMPLE_RATE)  # With default params
    else:
        logger.debug("GPU disabled. Running in CPU mode.")
        rec = vosk.KaldiRecognizer(vosk.Model(str(model_path)), VOSK_SAMPLE_RATE)  # CPU mode

    if FASTER_DECODING:
        rec.SetBeam(8)  # Lower the beam for faster decoding
        rec.SetMaxActive(5000)

    # Vosk processes chunks of audio data, so increasing this size allows the GPU to handle larger chunks in a single pass
    # rec.SetMaxChunkSize(int(VOSK_MAX_CHUNK_SIZE))  # Vosk Chunk Size (in bytes) #TODO: fix

    return rec


# Transcribe audio using Vosk
def transcribe_audio_vosk(audio_file, model):
    setup_vosk_logging()  # Configure Vosk logs

    with wave.open(audio_file, "rb") as wf:
        # model = vosk.Model(str(model))  # Re-initialization of Vosk #NOTE: remove
        print(model)

        # Use the function where you're processing the audio
        rec = setup_vosk_gpu(VOSK_MODEL_FULL_PATH_ENV)  # GPU or CPU mode based on flag

        transcription = []

        while True:
            data = wf.readframes(VOSK_SAMPLE_RATE)  # Read audio data in chunks
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                transcription.append(result.get("text", ""))

        # Get the final result
        final_result = json.loads(rec.FinalResult())
        transcription.append(final_result.get("text", ""))
        # wf.close()

        return " ".join(transcription)


def process_audio_file(audio_file, model_path):
    """
    Process the audio file using Vosk for transcription and speaker diarization.
    """
    logger.debug(f"Processing audio file: {audio_file}")

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file {audio_file} not found.")

    with wave.open(audio_file, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            return

        # Load Vosk model
        print(f"Model Path: {model_path}")
        path = Path(VOSK_MODEL_FULL_PATH_ENV)

        try:
            transcription = transcribe_audio_vosk(audio_file, path)
            return transcription
        except Exception as e:
            logger.error(f"Error during transcription for file {audio_file}: {e}")
            return ""


def prepare_segment_data_batching(batched_segments, model_path, total_audio_length, audio_name):
    """
    Prepares the segment data for multiprocessing.
    Args:
        batched_segments (list): List of batched segments containing start/end times and speaker labels.
        model_path (Path): Path to the Vosk model.
        total_audio_length (float): Total length of the audio file in seconds.
    Returns:
        list: List of dictionaries containing the segment data for each batch.
    """
    load_dotenv()
    output_dir = Path(os.getcwd(), os.getenv("OUTPUT_DIR_PARTS"))
    output_dir.resolve()
    # output_dir = env_as_path("OUTPUT_DIR_PARTS" "./audio_parts/parts")
    logger.debug(f"Output Dir Path: {str(output_dir)}")
    segment_data_list = []
    total_batches = len(batched_segments)

    for batch_num, batch in enumerate(batched_segments):
        batch_start_time = batch[0]["start_time"]
        batch_end_time = batch[-1]["end_time"]
        batch_speakers = set([seg["speaker"] for seg in batch])

        # Construct the full path for the segment file
        audio_segment_file_path = Path(output_dir) / f"{audio_name}_part{batch_num + 1}.wav"
        # audio_segment_file_path = Path(str(output_dir), f"{audio_name}_part{batch_num + 1}.wav")
        audio_segment_file_path.resolve()
        logger.info(f"Batch Path: {str(audio_segment_file_path)}")

        segment_data_list.append(
            {
                "audio_segment_file": str(audio_segment_file_path),
                "model_path": model_path,
                "speakers": batch_speakers,
                "segment_num": batch_num,
                "total_segments": total_batches,
                "start_time": batch_start_time,
                "end_time": batch_end_time,
                "total_audio_length": total_audio_length,
            }
        )

    return segment_data_list


def prepare_segment_data(rttm_segments, model_path, total_audio_length, audio_name):
    """
    Prepares the segment data for multiprocessing.
    Args:
        rttm_segments (list): List of segments containing start/end times and speaker labels.
        model_path (Path): Path to the Vosk model.
        total_audio_length (float): Total length of the audio file in seconds.
    Returns:
        list: List of dictionaries containing the segment data for each speaker.
    """
    load_dotenv()
    # output_dir = Path(os.getenv("OUTPUT_DIR_PARTS"))
    output_dir = Path(os.getcwd(), os.getenv("OUTPUT_DIR_PARTS"))
    output_dir.resolve()
    # output_dir = env_as_path("OUTPUT_DIR_PARTS" "./audio_parts/parts")
    logger.debug(f"Output Dir Path: {str(output_dir)}")
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_data_list = []
    total_segments = len(rttm_segments)
    with tqdm(total=total_segments, desc="Processing Segments") as pbar:
        for segment_num, segment in enumerate(rttm_segments):
            speaker = segment["speaker"]
            start_time = segment["start_time"]
            end_time = segment["end_time"]

            # Construct the full path for the segment file
            audio_segment_file_path = Path(output_dir) / f"{audio_name}_{speaker}_part{segment_num + 1}.wav"
            audio_segment_file_path.resolve()

            logger.info(f"Segment Path: {str(audio_segment_file_path)}")
            audio_segment_file = f"{str(audio_segment_file_path)}"
            segment_data_list.append(
                {
                    "audio_segment_file": str(audio_segment_file),
                    "model_path": model_path,
                    "speaker": speaker,
                    "segment_num": segment_num,
                    "total_segments": total_segments,
                    "start_time": start_time,
                    "end_time": end_time,
                    "total_audio_length": total_audio_length,
                }
            )
            pbar.update(1)

    return segment_data_list


def split_audio_by_segments(audio_file, rttm_segments, output_dir, speaker_name_map=None):
    """
    Split audio into segments based on RTTM and assign real speaker names if provided.

    Args:
        audio_file (Path): Path to the original audio file.
        rttm_segments (list): List of RTTM segments with start/end times and speaker labels.
        output_dir (str or Path): Directory to save the audio segments.
        speaker_name_map (dict, optional): Mapping of generic speaker names to real names.

    Returns:
        list: A list of dictionaries containing segment info (speaker, start_time, end_time, and audio_segment_file).
    """
    # output_dir = env_as_path("OUTPUT_DIR_PARTS" "./audio_parts/parts")
    output_dir = Path(os.getcwd(), os.getenv("OUTPUT_DIR_PARTS"))
    output_dir.resolve()
    audio_segment_files = []
    audio = AudioSegment.from_wav(audio_file)

    total_segments = len(rttm_segments)
    with tqdm(total=total_segments, desc="Processing Segments") as pbar:

        for segment_num, segment in enumerate(rttm_segments):
            speaker = segment["speaker"]

            # Use real speaker names if available
            if speaker_name_map and speaker in speaker_name_map:
                speaker = speaker_name_map[speaker]

            # Construct the full path for the segment file
            audio_segment_file_path = (
                Path(output_dir) / f"{audio_file.stem}_{speaker}_part{segment_num + 1}.wav"
            )
            audio_segment_file_path.resolve()

            logger.info(f"Segment Path: {str(audio_segment_file_path)}")

            # Extract and save the audio segment
            start_time_ms = segment["start_time"] * 1000  # Convert to milliseconds
            end_time_ms = segment["end_time"] * 1000  # Convert to milliseconds
            audio_segment = audio[start_time_ms:end_time_ms]
            audio_segment.export(str(audio_segment_file_path), format="wav")

            # Append the segment info, including the file path
            audio_segment_files.append(
                {
                    "speaker": speaker,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "audio_segment_file": str(audio_segment_file_path),
                }
            )

            pbar.update(1)

    return audio_segment_files


def split_audio_by_time(audio_file, start_time, end_time, output_file):
    """
    Split the audio file based on start and end times and save the segment.
    Args:
        audio_file (str): Path to the original audio file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        output_file (str): Path to save the split audio segment.
    """
    audio = AudioSegment.from_wav(audio_file)
    start_ms = start_time * 1000  # Convert to milliseconds
    end_ms = end_time * 1000  # Convert to milliseconds
    segment = audio[start_ms:end_ms]
    segment.export(str(output_file), format="wav")


def process_speaker_segments(audio_file, model_path, rttm_segments, output_dir):
    """
    Process each speaker segment by running Vosk transcription on it.
    Args:
        audio_file (str): Path to the original audio file.
        model_path (str): Path to the Vosk model.
        rttm_segments (list): List of speaker segments from RTTM file.
        output_dir (str): Directory to save the output transcriptions.
    """
    speaker_segments = split_audio_by_speaker(audio_file, rttm_segments)
    full_transcription = []

    total_segments = len(speaker_segments)
    with tqdm(total=total_segments, desc="Processing Segments") as pbar:

        for speaker, segment_file in speaker_segments:
            # Run Vosk transcription on each segment
            print(f"Transcribing segment {segment_file} for speaker {speaker}...")
            transcription = transcribe_audio_vosk(segment_file, model_path)
            transcription_text = " ".join([json.loads(t).get("text", "") for t in transcription])

            full_transcription.append(
                {"speaker": speaker, "transcription": transcription_text, "segment_file": segment_file}
            )
            pbar.update(1)

    # Save the results
    json_output_file = Path(output_dir, f"{os.path.basename(audio_file)}_transcription.json")
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(full_transcription, f, ensure_ascii=False, indent=4)

    print(f"Transcriptions saved to {json_output_file}")
    return full_transcription


def split_audio_by_speaker(audio_file, rttm_segments):
    """
    Split the original audio into segments based on speaker timestamps from the RTTM file.
    Args:
        audio_file (str): Path to the original audio file.
        rttm_segments (list): List of speaker segments with start_time, end_time, and speaker label.
    Returns:
        list: List of tuples containing speaker label and path to the corresponding audio segment file.
    """
    audio = AudioSegment.from_wav(audio_file)
    output_segments = []

    total_segments = len(rttm_segments)
    with tqdm(total=total_segments, desc="Processing Segments") as pbar:
        for i, segment in enumerate(rttm_segments):
            start_time_ms = segment["start_time"] * 1000  # Convert to milliseconds
            end_time_ms = segment["end_time"] * 1000  # Convert to milliseconds
            speaker = segment["speaker"]

            # Slice the audio segment
            audio_segment = audio[start_time_ms:end_time_ms]
            segment_filename = f"{audio_file[:-4]}_{speaker}_segment_{i}.wav"
            audio_segment.export(str(segment_filename), format="wav")
            output_segments.append((speaker, segment_filename))

            pbar.update(1)

    return output_segments
