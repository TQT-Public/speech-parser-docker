import multiprocessing
import os
from pathlib import Path
import shutil

from tqdm import tqdm
from speech_parser.audio_processing.convert_audio import convert_wav_to_mp3, convert_wav_to_pcm
from speech_parser.audio_processing.process_audio import (
    prepare_segment_data,
    process_audio_file,
    split_audio_by_segments,
)
from models.download_model import check_and_download_model
from speech_parser.audio_processing.speaker_identify import (
    assign_speaker_names,
    check_rttm_file,
    filter_rttm_segments,
    identify_speakers_pyannote,
    load_rttm_file,
)
from speech_parser.utils.env import env_as_bool, env_as_float, env_as_int, env_as_path, env_as_str
from speech_parser.utils.helpers import delete_audio_segments, set_custom_vosk_config, set_default_vosk_config
from speech_parser.audio_processing.save_results import (
    save_filtered_dialogue,
    save_summary,
    save_transcription_results,
)

# from multiprocessing import Pool
from loguru import logger
import datetime
from dotenv import load_dotenv
from pydub import AudioSegment


# Load environment variables from .env
load_dotenv()


# Load variables
MAX_PROCESSES = env_as_int("MAX_PROCESSES", 3)
DRY_RUN = env_as_bool("DRY_RUN", "False")
CLEAR_AUDIO_PARTS_RUN = env_as_bool("CLEAR_AUDIO_PARTS_RUN", "False")
ASSIGNSPEAKERS = env_as_bool("ASSIGNSPEAKERS", "False")
FILTER_UNNECESSARY_RTTM = env_as_bool("FILTER_UNNECESSARY_RTTM", "True")
MIN_RTTM_DURATION = env_as_float("MIN_RTTM_DURATION", 2.0)
USE_CUSTOM_VOSK = env_as_bool("USE_CUSTOM_VOSK", "False")
USE_BATCHES = env_as_bool("USE_BATCHES", "False")
BATCH_SIZE = env_as_float("BATCH_SIZE", 10.0)  # Default batch size in seconds

# File paths from .env
AUDIO_FILE_NAME = env_as_path("AUDIO_FILE_NAME", "ZOOM067.wav")
WORKSPACE_ENV = env_as_path("WORKSPACE", "./sources")
VOSK_MODEL_PATH_ENV = env_as_path("VOSK_MODEL_PATH", "./models/vosk")
MODEL_NAME = env_as_str("MODEL_NAME")
VOSK_MODEL_FULL_PATH_ENV = Path(str(VOSK_MODEL_PATH_ENV))
# VOSK_MODEL_FULL_PATH_ENV = Path(str(VOSK_MODEL_PATH_ENV), str(MODEL_NAME))
OUTPUT_DIR = env_as_path("OUTPUT_DIR", "./output")
AUDIOWORKSPACE_ENV = env_as_path("AUDIOWORKSPACE", "./audio_parts")
OUTPUT_DIR_PARTS = env_as_path("OUTPUT_DIR_PARTS" "./audio_parts/parts")


def batch_segments(rttm_segments, batch_size):
    """
    Batch RTTM segments into larger chunks based on batch_size.
    Args:
        rttm_segments (list): List of segments with 'start_time', 'end_time', and 'speaker'.
        batch_size (float): Size of each batch in seconds.
    Returns:
        list: List of batched segments, where each batch has 'start_time', 'end_time', and 'speaker'.
    """
    batched_segments = []
    current_batch = None
    current_start_time = None
    current_end_time = None
    current_speaker = None

    for segment in rttm_segments:
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        speaker = segment["speaker"]

        # If there's no current batch or the batch would exceed the batch size, finalize the current batch
        if current_batch is None:
            current_batch = []
            current_start_time = start_time
            current_speaker = speaker
        elif end_time - current_start_time > batch_size:
            batched_segments.append(
                {
                    "speaker": current_speaker,
                    "start_time": current_start_time,
                    "end_time": current_end_time,
                }
            )
            current_batch = []
            current_start_time = start_time
            current_speaker = speaker

        current_batch.append(segment)
        current_end_time = end_time

    # Append the last batch
    if current_batch:
        batched_segments.append(
            {
                "speaker": current_speaker,
                "start_time": current_start_time,
                "end_time": current_end_time,
            }
        )

    # Split the audio for each batch and save as a WAV file
    audio_file = AUDIOWORKSPACE_ENV / f"{AUDIO_FILE_NAME.stem}_converted.wav"
    # output_dir = Path(OUTPUT_DIR_PARTS)
    output_dir = Path(os.getcwd(), os.getenv("OUTPUT_DIR_PARTS"))

    audio = AudioSegment.from_wav(audio_file)
    for batch_num, batch in enumerate(batched_segments):
        start_time_ms = batch["start_time"] * 1000
        end_time_ms = batch["end_time"] * 1000

        # Extract and save the audio batch segment
        audio_segment = audio[start_time_ms:end_time_ms]
        audio_segment_file_path = (
            Path(output_dir) / f"{AUDIO_FILE_NAME.stem}_converted_{batch['speaker']}_part{batch_num + 1}.wav"
        )
        audio_segment_file_path.resolve()
        audio_segment.export(str(audio_segment_file_path), format="wav")
        batch["audio_segment_file"] = str(audio_segment_file_path)

    return batched_segments


def check_and_process_audio(audio_file, model_path, output_dir):
    """
    Check if the transcription or diarization has already been performed before proceeding.
    Args:
        audio_file (str): Path to the audio file.
        model_path (str): Path to the model.
        output_dir (str): Path to save output files.
    """
    # Generate file names based on the audio file name
    rttm_file = Path(output_dir, f"{os.path.basename(audio_file)}.rttm")
    json_file = Path(output_dir, f"{os.path.basename(audio_file)}_transcription.json")

    # Check if the output files already exist
    if os.path.exists(rttm_file) and os.path.exists(json_file):
        print(f"Files for {audio_file} already exist. Skipping processing.")
        return json_file  # Return existing JSON file path

    # If files do not exist, proceed with processing
    return process_audio_file(audio_file, model_path, output_dir)


# Transcription using Vosk model
def process_audio_segments(segment_data):
    """
    Function to process individual audio segments in parallel.
    Args:
        segment_data (dict): Dictionary containing data for each segment to process.
    """
    audio_segment_file = segment_data["audio_segment_file"]
    model_path = segment_data["model_path"]  # Pass model path as string
    speaker = segment_data["speaker"]
    segment_num = segment_data["segment_num"]
    total_segments = segment_data["total_segments"]
    start_time = segment_data["start_time"]
    end_time = segment_data["end_time"]
    total_audio_length = segment_data["total_audio_length"]

    logger.info(
        f"Processing segment {segment_num + 1} out of {total_segments} for {speaker} "
        f"({start_time:.2f}-{end_time:.2f}s of {total_audio_length:.2f}s total)"
    )

    try:
        print(model_path)  # Initialize Vosk model with correct path
        model = Path(VOSK_MODEL_PATH_ENV)
        transcription = process_audio_file(str(audio_segment_file), model)
        logger.info(f"Transcription for {speaker} segment {segment_num + 1}: {transcription}")
    except Exception as e:
        logger.error(f"Error processing segment {segment_num + 1}: {e}")
        return {"speaker": speaker, "transcription": "", "start_time": start_time, "end_time": end_time}

    return {
        "speaker": speaker,
        "transcription": transcription if transcription else "",
        "start_time": start_time,
        "end_time": end_time,
    }


def speech_parser():

    load_dotenv()
    # Environmental Variables Defines
    WORKSPACE = Path(WORKSPACE_ENV)
    VOSK_MODEL_PATH = Path(VOSK_MODEL_PATH_ENV)  # Make sure this points to the model folder

    audio_file = WORKSPACE / AUDIO_FILE_NAME
    model_name = MODEL_NAME

    output_dir = OUTPUT_DIR
    output_dir_parts = OUTPUT_DIR_PARTS

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_parts, exist_ok=True)

    AUDIOWORKSPACE = Path(AUDIOWORKSPACE_ENV)
    audio_name = f"{audio_file.stem}"

    converted_file_path = AUDIOWORKSPACE / f"{audio_file.stem}_converted.wav"

    start_time = datetime.datetime.now()  # Record the start time
    logger.debug(f"Start Time: {start_time}")

    # Clear /parts/ folder from previous RUN
    if CLEAR_AUDIO_PARTS_RUN:
        delete_audio_segments(
            OUTPUT_DIR_PARTS,
            f"{audio_name}",
            {1: "SPEAKER_00", 2: "SPEAKER_01", 3: "SPEAKER_02"},  # TODO: add all speaker variations
        )

    # Download or verify model path
    model_path = check_and_download_model(model_name, VOSK_MODEL_PATH)

    print(f"Model path - check at {model_path}")

    if USE_CUSTOM_VOSK:
        set_custom_vosk_config()
    else:
        set_default_vosk_config()

    # -- Start Audio Processing
    if not converted_file_path.exists():
        converted_file = convert_wav_to_pcm(audio_file)
        shutil.move(converted_file, converted_file_path)
        logger.info(f"Converted file saved to: {converted_file_path}")
        # -- Convert to mp3
        convert_wav_to_mp3(converted_file_path)
        logger.info(f"Mp3 file saved to: {str(converted_file_path)[:-3]}mp3")
    else:
        logger.info(f"Using existing converted file: {converted_file_path}")

    # Perform speaker diarization via API: identify_speakers_pyannote(converted_file_path)
    rttm_file = check_rttm_file(converted_file_path)
    if not rttm_file:
        rttm_segments = identify_speakers_pyannote(converted_file_path)
    else:
        rttm_segments = load_rttm_file(rttm_file)  # Use load_rttm_file to parse the existing RTTM

    # After processing RTTM segments - assign found speaker names in console prompt
    if ASSIGNSPEAKERS:
        speaker_name_map = assign_speaker_names(rttm_segments)

        # Update the RTTM segments with real names before saving the transcription
        for segment in rttm_segments:
            segment["speaker"] = speaker_name_map[segment["speaker"]]

    # Filter RTTM segments shorter than MIN_RTTM_DURATION seconds
    if FILTER_UNNECESSARY_RTTM:
        rttm_segments = filter_rttm_segments(rttm_segments, min_duration=MIN_RTTM_DURATION)

    # Use batching if enabled
    if USE_BATCHES:
        logger.info(f"Batch processing enabled. Batching segments with batch size: {BATCH_SIZE} seconds.")
        rttm_segments = batch_segments(rttm_segments, BATCH_SIZE)
        logger.debug(f"Segments now: {rttm_segments}")
    else:
        # Create speaker WAV segments, split audio into segments for each speaker
        logger.info(f"Splitting segments with RTTM file: {rttm_file} by {len(rttm_segments)} parts.")
        rttm_segments = split_audio_by_segments(
            converted_file_path, rttm_segments, output_dir_parts
        )  # No batching
        logger.debug(f"Audio segments created: {rttm_segments}")

    # Calculate total length based on the last segment's end_time
    total_audio_length = max([segment["end_time"] for segment in rttm_segments])
    logger.info(f"Total audio length: {total_audio_length}s")

    # Save transcription to CSV and JSON
    csv_file = AUDIOWORKSPACE / f"{audio_file.stem}.csv"
    json_file = AUDIOWORKSPACE / f"{audio_file.stem}.json"
    rttm_file = AUDIOWORKSPACE / f"{audio_file.stem}.rttm"

    # Prepare segment data
    segment_data_list = prepare_segment_data(
        rttm_segments, str(VOSK_MODEL_PATH), total_audio_length, f"{audio_name}_converted"
    )

    if not DRY_RUN:
        # Multiprocessing pool to process segments
        logger.info(f"Using max number of CPU processes: {MAX_PROCESSES}")
        pool = multiprocessing.Pool(processes=MAX_PROCESSES)  # Adjust based on your system
        # Process segments with progress bar
        speaker_transcription = []
        try:
            # speaker_transcription = pool.map(process_audio_segments, segment_data_list) # Pure CPU
            with tqdm(total=len(segment_data_list), desc="Processing Segments") as pbar:
                for result in pool.imap_unordered(process_audio_segments, segment_data_list):
                    speaker_transcription.append(result)
                    pbar.update(1)
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            pool.close()
            pool.join()

        # Save results to files after processing all segments
        save_transcription_results(
            speaker_transcription, audio_name, output_dir, csv_file, json_file, rttm_file
        )
        save_filtered_dialogue(speaker_transcription, audio_name, output_dir)
        save_summary(speaker_transcription, audio_name, output_dir)

        logger.info(f"Files saved: {csv_file}, {json_file}, {rttm_file}")

    end_time = datetime.datetime.now()  # Record the end time
    logger.debug(f"End Time: {end_time}")

    # Calculate and log the total runtime
    total_runtime = end_time - start_time
    logger.debug(f"Total run time: {total_runtime}")

    return csv_file
    # logger.info(f"Total run time: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")


if __name__ == "__main__":
    speech_parser()
