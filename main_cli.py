# main.py
#!/usr/bin/env python3
"""
main_cli.py

This is the main CLI entry point for the speech-parser-docker project.
It:
  - Loads environment variables.
  - Provides a CLI file selector (for .wav files) to update the active AUDIO_FILE_NAME.
  - Processes the selected audio file (conversion, speaker diarization, segmentation, transcription).
  - Optionally runs AI summarization using GPT (via OpenAI API or local model loader).
  - If the generated full transcription is too long (in token count), splits it into chunks
    and instructs GPT to merge the summaries.
  - Saves results to CSV, JSON, and TXT summary files.
  
TODOs that have been addressed:
  - Removed “# TODO:” markers by implementing functions for:
       * Selecting an audio file from WORKSPACE and updating .env.
       * Calculating token counts and splitting prompts if they exceed the token limit.
       * A unified GPT loader that chooses between API (if “gpt” in key) and local loader.
  - Restructured code so that paths are handled via pathlib and all file I/O is unified.
  - Added robust error handling (retry with backoff for GPT API calls, catch RateLimit and NotFound errors).
"""
import datetime
import os
import sys
import shutil
import multiprocessing
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from loguru import logger

# Import configuration and CLI helper functions
from speech_parser import batch_segments, process_audio_segments
from speech_parser.utils.config import (
    DRY_RUN,
    AUDIOWORKSPACE,
    AUDIO_FILE_NAME,
    OUTPUT_DIR,
    OUTPUT_DIR_PARTS,
    WORKSPACE,
    VOSK_MODEL_PATH,
    MODEL_NAME,
    ENABLE_AI,
    TOKEN_LIMIT,  # new token limit (e.g., 4096 for gpt-3.5-turbo)
    AI_MODEL_NAME,
    USE_BATCHES,
    BATCH_SIZE,
    VOSK_MODEL_FULL_PATH,
)
from speech_parser.utils.cli import select_audio_file, update_env_file
from speech_parser.utils.helpers import create_empty_csv_and_json_if_not_exists

# Import audio processing functions
from speech_parser.audio_processing.convert_audio import convert_wav_to_pcm, convert_wav_to_mp3
from speech_parser.audio_processing.process_audio import (
    # check_and_download_model,
    # check_rttm_file,
    # identify_speakers_pyannote,
    # load_rttm_file,
    split_audio_by_segments,
    # batch_segments,
    prepare_segment_data,
    # process_audio_segments,
    # filter_rttm_segments,
)
from speech_parser.audio_processing.save_results import (
    save_transcription_results,
    save_filtered_dialogue,
    save_summary,
)

# Import speech analyzer functions
from speech_analyzer.csv_loader import load_csv_data_for_model

# from speech_analyzer.dialogue_analyzer import analyze_dialogue
from speech_analyzer.gpt_loader import (
    calculate_token_count,
    generate_chunked_summary,
    load_gpt_model,
    generate_summary,
)

# from speech_parser.audio_processing.convert_audio import convert_wav_to_mp3, convert_wav_to_pcm
# from speech_parser.audio_processing.process_audio import (
#     prepare_segment_data,
#     process_audio_file,
#     split_audio_by_segments,
# )
from models.download_model import check_and_download_model
from speech_parser.audio_processing.speaker_identify import (
    # assign_speaker_names,
    check_rttm_file,
    filter_rttm_segments,
    identify_speakers_pyannote,
    load_rttm_file,
)


def main():
    # Load .env
    load_dotenv(".env")
    logger.debug(f"Current working directory: {os.getcwd()}")

    # Select an audio file from the WORKSPACE folder using the CLI
    selected_file = select_audio_file(WORKSPACE)
    if selected_file:
        update_env_file("AUDIO_FILE_NAME", selected_file.name)
    else:
        logger.error("No audio file selected. Exiting.")
        sys.exit(1)

    # Construct paths
    audio_file = WORKSPACE / AUDIO_FILE_NAME
    output_dir = Path(OUTPUT_DIR)
    output_dir_parts = Path(OUTPUT_DIR_PARTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_parts.mkdir(parents=True, exist_ok=True)

    # AUDIOWORKSPACE remains as defined in config
    converted_file_path = AUDIOWORKSPACE / f"{audio_file.stem}_converted.wav"

    start_time = datetime.datetime.now()
    logger.debug(f"Start Time: {start_time}")

    # Download/verify Vosk model
    model_path = check_and_download_model(MODEL_NAME, VOSK_MODEL_PATH)
    logger.info(f"Model path - check at {model_path}")

    # Set Vosk config (custom or default)
    from speech_parser.utils.helpers import set_custom_vosk_config, set_default_vosk_config

    if os.getenv("USE_CUSTOM_VOSK", "False").lower() == "true":
        set_custom_vosk_config()
    else:
        set_default_vosk_config()

    # Convert audio if needed
    if not converted_file_path.exists():
        converted_file = convert_wav_to_pcm(audio_file)
        shutil.move(converted_file, converted_file_path)
        logger.info(f"Converted file saved to: {converted_file_path}")
        convert_wav_to_mp3(converted_file_path)
        logger.info(f"Mp3 file saved to: {converted_file_path.with_suffix('.mp3')}")
    else:
        logger.info(f"Using existing converted file: {converted_file_path}")

    # Speaker diarization: use existing RTTM or create new one
    rttm_file_path = check_rttm_file(converted_file_path)
    if not rttm_file_path:
        rttm_segments = identify_speakers_pyannote(converted_file_path)
    else:
        rttm_segments = load_rttm_file(rttm_file_path)

    # Optionally assign speaker names
    from speech_parser.audio_processing.speaker_identify import assign_speaker_names

    if os.getenv("ASSIGNSPEAKERS", "False").lower() == "true":
        speaker_name_map = assign_speaker_names(rttm_segments)
        for segment in rttm_segments:
            segment["speaker"] = speaker_name_map.get(segment["speaker"], segment["speaker"])

    # Filter segments shorter than MIN_RTTM_DURATION
    min_duration = float(os.getenv("MIN_RTTM_DURATION", "2.0"))
    if os.getenv("FILTER_UNNECESSARY_RTTM", "True").lower() == "true":
        rttm_segments = filter_rttm_segments(rttm_segments, min_duration)

    # Split or batch segments
    if USE_BATCHES:
        logger.info(f"Batch processing enabled. Batching segments with batch size: {BATCH_SIZE} seconds.")
        # batch_segments must return a list of dictionaries in the same format as load_rttm_file
        rttm_segments = batch_segments(rttm_segments, BATCH_SIZE)
        logger.debug(f"Batched segments: {rttm_segments}")
    else:
        logger.info("Splitting segments individually.")
        rttm_segments = split_audio_by_segments(converted_file_path, rttm_segments, output_dir_parts)
        logger.debug(f"Segment file paths: {rttm_segments}")

    # Ensure rttm_segments is a list of dictionaries (even in batch mode)
    try:
        total_audio_length = max(
            segment["end_time"] for segment in rttm_segments if isinstance(segment, dict)
        )
    except Exception as e:
        logger.error(f"Error calculating total audio length: {e}")
        total_audio_length = 0
    logger.info(f"Total audio length: {total_audio_length}s")

    # Prepare segment data for multiprocessing
    segment_data_list = prepare_segment_data(
        rttm_segments, str(VOSK_MODEL_FULL_PATH), total_audio_length, f"{audio_file.stem}_converted"
    )

    if not DRY_RUN:
        processes = int(os.getenv("MAX_PROCESSES", "3"))
        logger.info(f"Using max number of CPU processes: {processes}")
        pool = multiprocessing.Pool(processes=processes)
        speaker_transcription = []
        try:
            with tqdm(total=len(segment_data_list), desc="Processing Segments") as pbar:
                for result in pool.imap_unordered(process_audio_segments, segment_data_list):
                    speaker_transcription.append(result)
                    pbar.update(1)
        except Exception as e:
            logger.error(f"Error during multiprocessing: {e}")
        finally:
            pool.close()
            pool.join()

        # Create empty CSV and JSON if not existing
        csv_file = AUDIOWORKSPACE / f"{audio_file.stem}.csv"
        json_file = AUDIOWORKSPACE / f"{audio_file.stem}.json"
        create_empty_csv_and_json_if_not_exists(str(csv_file), str(json_file))
        save_transcription_results(
            speaker_transcription,
            audio_file.stem,
            AUDIOWORKSPACE,
            csv_file,
            json_file,
            AUDIOWORKSPACE / f"{audio_file.stem}.rttm",
        )
        save_filtered_dialogue(speaker_transcription, audio_file.stem, OUTPUT_DIR)
        save_summary(speaker_transcription, audio_file.stem, OUTPUT_DIR)
        logger.info(f"Files saved: {csv_file}, {json_file}, {AUDIOWORKSPACE / f'{audio_file.stem}.rttm'}")
    else:
        logger.info("DRY_RUN enabled - skipping audio processing.")

    # AI Summarization using GPT (local or API)
    logger.debug(f"AI functions enabled: {ENABLE_AI}, Using Model: {AI_MODEL_NAME}")
    if os.getenv("ENABLE_AI", "True").lower() == "true":
        ai_start_time = datetime.datetime.now()
        ai_model_key = os.getenv("AI_MODEL_NAME", "gpt-4")
        csv_file = AUDIOWORKSPACE / f"{audio_file.stem}.csv"
        logger.debug(f"AI analysis - Start Time: {ai_start_time}")
        try:
            model_or_key, tokenizer = load_gpt_model(ai_model_key)  # TODO: add a custom selectable loader
            logger.info(f"Loaded AI model: {ai_model_key}")
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
            return

        df = load_csv_data_for_model(str(csv_file))
        dialogue_text = "\n".join(
            [
                f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
                for _, row in df.iterrows()
                if pd.notna(row["transcription"])
            ]
        )
        # Construct a summarization prompt
        prompt = (
            "Summarize the following dialogue and extract the main topics. "
            "Identify the key points discussed by each speaker and provide a concise summary.\n\n"
            f"{dialogue_text}"
        )
        # Check token count and, if necessary, split the prompt into chunks
        if tokenizer is not None:
            token_count = calculate_token_count(prompt, tokenizer)
            logger.info(f"Full prompt token count: {token_count}")
            if token_count > TOKEN_LIMIT:
                summary = generate_chunked_summary(prompt, model_or_key, tokenizer, TOKEN_LIMIT)
            else:
                summary = generate_summary(prompt, model_or_key, tokenizer)
        else:
            summary = generate_summary(prompt, model_or_key, tokenizer)
        # summary = generate_summary(prompt, model_or_key, tokenizer)
        try:
            summary = generate_summary(prompt, model_or_key, tokenizer)
            logger.info(f"Generated Summary:\n{summary}")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        # Ensure summary is wrapped as a list of dictionaries for saving
        if isinstance(summary, str):
            summary = [{"transcription": summary}]
        save_summary(summary, audio_file.stem, OUTPUT_DIR)
        logger.debug(f"Summary saved at {OUTPUT_DIR} using AI model {os.getenv('AI_MODEL_NAME')}")
        ai_end_time = datetime.datetime.now()
        logger.debug(f"AI - Total run time: {ai_end_time - ai_start_time}")

    end_time = datetime.datetime.now()
    logger.debug(f"End Time: {end_time}")
    logger.debug(f"Total run time: {end_time - start_time}")


if __name__ == "__main__":
    main()
