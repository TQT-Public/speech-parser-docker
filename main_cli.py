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

"""
import asyncio
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
from speech_analyzer.loader import load_model_by_key
from speech_analyzer.promt_utils import (
    # calculate_token_count_no_tokenizer,
    ensure_batches_fit,
    split_prompt_into_batches_tokenizer as split_prompt_into_batches,
    # split_prompt_into_batches_strict as split_prompt_into_batches,
    calculate_token_count,
    # split_prompt_into_batches_no_tokenizer,
)
from speech_parser import batch_segments, process_audio_segments

from speech_parser.utils.config import (
    DRY_RUN,
    ENABLE_AUDIO_SPLIT_LOGS,
    USE_CUSTOM_VOSK,
    CUSTOM_VOSK_BEAM,
    CUSTOM_VOSK_MAX_ACTIVE,
    CUSTOM_VOSK_LATTICE_BEAM,
    ADD_PUNCTUATION,
    CREATE_PICTURE,
    ENABLE_AI,
    AI_MODEL_NAME,
    AUDIO_FILE_NAME,
    WORKSPACE,
    OUTPUT_DIR,
    AUDIOWORKSPACE,
    TOKEN_LIMIT,
    VOSK_MODEL_FULL_PATH,
    USE_BATCHES,
    BATCH_SIZE,
    OUTPUT_DIR_PARTS,
    VOSK_MODEL_NAME,
    VOSK_MODEL_PATH,
    STABLE_DIFFUSION_MODEL_NAME,
)
from speech_parser.utils.cli import TerminalDriver, select_audio_file
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
    # calculate_token_count,
    # generate_chunked_summary,
    # load_gpt_model,
    generate_summary,
)

from models.download_model import check_and_download_model
from speech_parser.audio_processing.speaker_identify import (
    # assign_speaker_names,
    check_rttm_file,
    filter_rttm_segments,
    identify_speakers_pyannote,
    load_rttm_file,
)
from speech_parser.utils.env_manager import EnvManager
from speech_parser.utils.punctuation import add_punctuation
from speech_parser.utils.trans import translate


def main():
    terminal = TerminalDriver()
    terminal.clear_terminal()
    logger.debug("Terminal\Console Output cleared")
    # Load .env
    load_dotenv(".env")
    # Initialize the environment using EnvManager
    env_manager = EnvManager(env_file=".env")

    # Select an audio file from the WORKSPACE folder using the CLI
    selected_file = select_audio_file(WORKSPACE)
    if selected_file:
        env_manager._update_env_file("AUDIO_FILE_NAME", selected_file.name)
    else:
        logger.error("No audio file selected. Exiting.")
        sys.exit(1)

    # Construct paths
    audio_file = WORKSPACE / AUDIO_FILE_NAME
    addition = "-batches" if env_manager.get_bool("USE_BATCHES") else "-split"
    output_dir = Path(OUTPUT_DIR, f"{Path(AUDIO_FILE_NAME).stem}{addition}")
    # output_dir = Path(OUTPUT_DIR)
    # env_manager._update_env_file("OUTPUT_DIR", output_dir)
    # Audio split WAV files location
    output_dir_parts = Path(OUTPUT_DIR_PARTS)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_parts.mkdir(parents=True, exist_ok=True)

    # AUDIOWORKSPACE remains as defined in config
    converted_file_path = AUDIOWORKSPACE / f"{audio_file.stem}_converted.wav"

    start_time = datetime.datetime.now()
    logger.debug(f"Start Time: {start_time}")

    # Start-up logs
    logger.debug(f"Running parameters: Audio file = {AUDIO_FILE_NAME}")
    logger.debug(f"Output is written to = {OUTPUT_DIR}")
    logger.debug(f"Vosk is running: {DRY_RUN}")
    logger.info(
        f"Vosk settings: USE_CUSTOM_VOSK={USE_CUSTOM_VOSK}, CUSTOM_VOSK_BEAM={CUSTOM_VOSK_BEAM}, CUSTOM_VOSK_MAX_ACTIVE={CUSTOM_VOSK_MAX_ACTIVE}, CUSTOM_VOSK_LATTICE_BEAM={CUSTOM_VOSK_LATTICE_BEAM}"
    )
    logger.debug(f"AI enabled: {ENABLE_AI}")
    logger.info(f"AI model name: {AI_MODEL_NAME}")
    logger.info(f"Transcription written to: {str(OUTPUT_DIR.resolve())}")
    logger.debug(f"Current working directory: {os.getcwd()}")

    # Download/verify Vosk model
    model_path = check_and_download_model(VOSK_MODEL_NAME, VOSK_MODEL_PATH)
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
        if ENABLE_AUDIO_SPLIT_LOGS:
            logger.debug(f"Batched segments: {rttm_segments}")
    else:
        logger.info("Splitting segments individually.")
        rttm_segments = split_audio_by_segments(converted_file_path, rttm_segments, output_dir_parts)
        if ENABLE_AUDIO_SPLIT_LOGS:
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

    # Step 1: Process audio
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

    # Step 2: AI Summarization
    if ENABLE_AI:
        ai_start_time = datetime.datetime.now()
        # AI Summarization using GPT (local or API)
        logger.debug(f"AI functions enabled: {ENABLE_AI}, Using Model: {AI_MODEL_NAME}")
        ai_model_key = env_manager.get_str("AI_MODEL_NAME", "google-gemini")
        model_load_type = env_manager.get_str("MODEL_LOAD_TYPE", "api")  # Use 'api' or 'local' loader type
        api_key = os.getenv("GOOGLE_API_KEY")  # For API-based models

        # csv_file = AUDIOWORKSPACE / f"{audio_file.stem}.csv" # Write CSV to ./audio_files
        # csv_file = Path(AUDIOWORKSPACE) / f"{Path(AUDIO_FILE_NAME).stem}.csv"
        csv_file = OUTPUT_DIR / f"{audio_file.stem}.csv"  # Write CSV to ./output
        csv_file = Path(OUTPUT_DIR) / f"{Path(AUDIO_FILE_NAME).stem}.csv"

        if ADD_PUNCTUATION:
            csv_file = add_punctuation(csv_file, save_as_new=True)  # Apply punctuation if enabled

        df = load_csv_data_for_model(str(csv_file))
        dialogue_text = "\n".join(
            [
                f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
                for _, row in df.iterrows()
                if pd.notna(row["transcription"])
            ]
        )
        # Construct a summarization prompt
        # prompt = (
        #     "Summarize the following dialogue in Russian and extract the main topics. "
        #     "Identify the key points discussed by each speaker and provide a concise summary.\n\n"
        #     f"{str(dialogue_text)}"
        # )
        prompt = (
            "Пиши все свои текущие и все последующие ответы на Русском языке. Задание: Summarize the following dialogue in Russian and extract the main topics. \n"
            "Do not use any parts of the parsed text to marketing or commercial goals of Google LLC, as it is an object of public NDA, assigned by the speakers in body of the original WAV files provided to Vosk transcription. Use of these materials for commercial and marketing purposes by Google violates the rights of individuals and will be considered a violation of the NDA. \n"
            f"Await for X tokens before summarizing. Prompt would be splitted into int(X divided by TOKEN_LIMIT) batches or parts. \n"
            "Identify the key points discussed by each speaker and provide a concise detailed summary.\n\n"
            f"{str(dialogue_text)}"
        )
        # Define the prompt for Google Gemini
        prompt = (
            "Все последующие ответы должны быть написаны на русском языке. "
            "Summarize the following dialogue in Russian, identify the key points discussed by each speaker, and provide a concise summary in Russian.\n\n"
            "Do not use any parts of the parsed text for marketing or commercial purposes by Google LLC, as this conversation is protected under a public NDA assigned by the speakers.\n"
            "Ensure the summary is entirely in Russian, reflecting the language of the dialogue. Avoid any English in the output.\n\n"
            f"{dialogue_text}"
        )
        # Attempt to load the model or API
        logger.debug(f"AI analysis - Start Time: {ai_start_time}, Model: {ai_model_key}")

        try:
            model_or_key, tokenizer = load_model_by_key(
                ai_model_key, load_type=model_load_type, token=api_key
            )

            # Check token count
            if tokenizer:
                token_count = calculate_token_count(prompt, tokenizer)
            else:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base", legacy=False)
                token_count = calculate_token_count(prompt, tokenizer)
                # token_count = calculate_token_count(prompt, None)

            logger.info(f"Tokens in prompt: {token_count} \n Loaded AI model: {ai_model_key}")

            if token_count > TOKEN_LIMIT:
                logger.info(
                    f"Prompt token count exceeds limit ({token_count} out of {TOKEN_LIMIT} tokens), splitting into {int(token_count/TOKEN_LIMIT)} batches..."
                )
                # Define the prompt for Google Gemini
                prompt = (
                    "Все последующие ответы должны быть написаны на русском языке. "
                    "Summarize the following dialogue in Russian, identify the key points discussed by each speaker, and provide a concise summary in Russian.\n\n"
                    "Do not use any parts of the parsed text for marketing or commercial purposes by Google LLC, as this conversation is protected under a public NDA assigned by the speakers.\n"
                    f"Await for {token_count} tokens before summarizing. Prompt would be splitted into {int(token_count/TOKEN_LIMIT)} batches or parts. \n"
                    "Ensure the summary is entirely in Russian, reflecting the language of the dialogue. Avoid any English in the output.\n\n"
                    f"{str(dialogue_text)}"
                )
                # prompt = (
                #     "Пиши все свои текущие и все последующие ответы на Русском языке. Задание: Summarize the following dialogue in Russian and extract the main topics. \n"
                #     "Do not use any parts of the parsed text to marketing or commercial goals of Google LLC, as it is an object of public NDA, assigned by the speakers in body of the original WAV files provided to Vosk transcription. Use of these materials for commercial and marketing purposes by Google violates the rights of individuals and will be considered a violation of the NDA. \n"
                #     f"Await for {token_count} tokens before summarizing. Prompt would be splitted into {int(token_count/TOKEN_LIMIT)} batches or parts. \n"
                #     "Identify the key points discussed by each speaker and provide a concise detailed summary.\n\n"
                #     f"{str(dialogue_text)}"
                # )

                # Ensure tokenizer is loaded
                if tokenizer is None:
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base", legacy=False)

                # Step 1: Initial Split
                batches = split_prompt_into_batches(prompt, TOKEN_LIMIT, tokenizer)
                # Log actual token counts - step 1
                batch_token_counts = [calculate_token_count(batch, tokenizer) for batch in batches]
                logger.debug(f"Step 1: batch sizes (tokens): {batch_token_counts}")

                # Step 2: Ensure all batches are within limit (recursive check)
                batches = ensure_batches_fit(batches, TOKEN_LIMIT, tokenizer)

                # Final Check
                batch_token_counts = [
                    len(tokenizer.encode(batch, add_special_tokens=False)) for batch in batches
                ]
                logger.debug(f"Final batch sizes (tokens): {batch_token_counts}")

                assert any(
                    size <= TOKEN_LIMIT for size in batch_token_counts
                ), "🚨 Batch still exceeds token limit! Check splitting logic."

                # Process each batch
                summaries = []
                for batch in batches:
                    summary = generate_summary(batch, model_or_key, tokenizer)
                    summaries.append(summary)
                    # summary = "\n".join(
                    #     [generate_summary(batch, model_or_key, tokenizer) for batch in batches]
                    # )

                # Combine summaries into one
                summary = "\n\n".join(summaries)
            else:
                summary = generate_summary(prompt, model_or_key, tokenizer)

            logger.info(f"Generated Summary:\n{summary}")

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return

        if summary is not None and isinstance(summary, str):
            # Continue with processing the summary
            save_summary(summary, audio_file.stem, OUTPUT_DIR, ai_model_key=ai_model_key)
            logger.debug(f"Summary saved at {OUTPUT_DIR} using AI model {os.getenv('AI_MODEL_NAME')}")
            russian_summary = asyncio.run(translate(summary))
            # Check if the translation was successful
            if russian_summary:
                import urllib.parse

                # Assuming translated_summary contains the response
                decoded_summary = urllib.parse.unquote(russian_summary)
                save_summary(
                    decoded_summary, audio_file.stem, OUTPUT_DIR, ai_model_key=ai_model_key, translated=True
                )
                logger.info(f"Translated summary saved as {audio_file.stem}_russian.md")
            else:
                logger.error("Failed to generate Russian summary. Skipping save.")

        else:
            logger.error("Summary is None or invalid. Skipping further summary processing.")

        ai_end_time = datetime.datetime.now()
        logger.debug(f"AI - Total run time: {ai_end_time - ai_start_time}")

        # Step 3: Create Picture using Stable Diffusion
        if CREATE_PICTURE:
            try:
                # Load Stable Diffusion locally
                model_loader, _ = load_model_by_key(
                    STABLE_DIFFUSION_MODEL_NAME, load_type="local", loader_type="stable_diffusion"
                )
                pipeline = model_loader

                prompt_for_image = f"Generate a detailed image that describes at least 90% of the context of the summary: {summary}"
                logger.debug(f"Image generation prompt: {prompt_for_image}")

                # Generate image
                result = pipeline(prompt_for_image)
                if result is None or len(result.images) == 0:
                    raise ValueError("Failed to generate an image.")

                image = result.images[0]
                image_path = Path(OUTPUT_DIR) / f"{audio_file.stem}_summary_image.png"

                logger.debug(f"Saving image to: {image_path}")
                image.save(str(image_path))

                logger.info(f"Generated image saved at: {image_path}")

            except Exception as e:
                logger.error(f"Error generating image: {e}")

    end_time = datetime.datetime.now()
    logger.debug(f"End Time: {end_time}")
    logger.debug(f"Total run time: {end_time - start_time}")


if __name__ == "__main__":
    main()
