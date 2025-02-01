import os
from pathlib import Path
from loguru import logger
import openai
import pandas as pd
import json

# import torch
from speech_analyzer.csv_loader import (
    format_dialogue_for_summary,
    load_csv_data_for_model,
    # load_csv_for_neural_network,
    tokenize_text_data,
    # load_csv_for_neural_network,
)
from speech_analyzer.gpt_loader import call_gpt_api, load_gpt_model
from speech_analyzer.model_loader import load_all_models
from speech_parser.speech_parser import env_as_path
from speech_parser.utils.env import env_as_str

# from speech_analyzer.custom_llama import LlamaModel_fast_forward_inference


def load_dialogue_data(file_path):
    """
    Load dialogue data from a CSV or JSON file.

    Args:
        file_path (str or Path): Path to the input file.

    Returns:
        pd.DataFrame: A DataFrame with columns ['speaker', 'transcription', 'start_time', 'end_time'].
    """
    file_path = Path(file_path)  # Ensure file_path is a Path object
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix == ".csv":
        # Load CSV file
        df = pd.read_csv(file_path)
        logger.debug(f"CSV file loaded. Columns: {df.columns}")
        logger.debug(f"First row: {df.iloc[0].to_dict()}")

        # Ensure the required columns are present
        required_columns = {"speaker", "transcription", "start_time", "end_time"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain the following columns: {required_columns}")

        # Rename 'transcription' to 'text' for consistency
        # df = df.rename(columns={"transcription": "text"})

    elif file_path.suffix == ".json":
        # Load JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.debug(f"JSON file loaded. First item: {data[0]}")

        # Convert JSON to DataFrame
        df = pd.DataFrame(data)

        # Ensure the required columns are present
        required_columns = {"speaker", "transcription", "start_time", "end_time"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"JSON file must contain the following keys: {required_columns}")

        # Rename 'transcription' to 'text' for consistency
        # df = df.rename(columns={"transcription": "text"})

    else:
        raise ValueError("Unsupported file format. Expected .csv or .json")

    # Ensure the DataFrame has the correct columns
    df = df[["speaker", "transcription", "start_time", "end_time"]]
    return df


def load_prompt(model_name):
    # Example usage:
    # llama_prompt = load_prompt("llama")
    # mistral_prompt = load_prompt("mistral")
    # stable_diffusion_prompt = load_prompt("stable_diffusion")
    prompt_dir = Path("./models/prompts")
    prompt_file = prompt_dir / f"prompt_{model_name}.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found for model: {model_name}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def generate_summary_gpt(prompt, model, tokenizer=None):
    """
    Generate a summary using the specified model.

    Args:
        prompt (str): The input text to summarize.
        model: The language model to use for summarization.
        tokenizer: Tokenizer for the model (if local).

    Returns:
        str: The generated summary.
    """
    if isinstance(model, str):  # If using GPT API
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=1500,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    else:  # Using local model
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=1500)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_summary(prompt, model, tokenizer):
    """
    Generate a summary using a local model or API, based on configuration.

    Args:
        prompt (str): The input prompt to summarize.
        model: The language model (can be None for API calls).
        tokenizer: The tokenizer (can be None for API calls).

    Returns:
        str: The generated summary.
    """
    use_local_model = os.getenv("USE_LOCAL_MODEL", "False").lower() == "true"

    if use_local_model:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate summary using the model
        outputs = model.generate(**inputs, max_length=1500)

        # Decode the generated tokens into text
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    else:
        # Call GPT API
        model_version = os.getenv("GPT_API_VERSION_1", "gpt-3.5")
        return call_gpt_api(prompt, model_version)


def summarize_dialogue(dialogue_data, model, tokenizer):
    # Format dialogue for input
    dialogue_text = "\n".join([f"{row['speaker']}: {row['transcription']}" for row in dialogue_data])
    input_text = f"Summarize the following dialogue:\n{dialogue_text}"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def analyze_dialogue(file_path, model, tokenizer):
    """
    Analyze dialogue data and generate a summary.

    Args:
        file_path (str or Path): Path to the input file (CSV or JSON).
        model: The language model for summarization.
        tokenizer: The tokenizer for the language model.

    Returns:
        str: A summary of the dialogue.
    """
    # Load dialogue data (replace this with actual loading function) # TODO: fix
    dialogue_data = load_dialogue_data(file_path)
    # Load the CSV and display statistics
    df = load_csv_data_for_model(file_path, show_stats=True)

    dialogue_text = format_dialogue_for_summary(df)
    # Format dialogue data from CSV for summarization
    # dialogue_text = "\n".join(
    #     [
    #         f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
    #         for _, row in df.iterrows()
    #     ]
    # )

    # Show the first few rows of the data
    logger.debug("\nSample Data:")
    logger.info(df.head())
    logger.info(df.describe())
    # dialogue_data = load_csv_for_neural_network(file_path, target_column="transcription")

    # Format dialogue for input
    dialogue_text = "\n".join(
        [
            f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
            for _, row in dialogue_data.iterrows()
        ]
    )

    # Refine the prompt for better summarization
    prompt = (
        "Summarize the following dialogue and extract the main topics. "
        "Identify the key points discussed by each speaker and provide a concise summary.\n\n"
        f"{dialogue_text}"
    )

    # Generate summary (ensuring dtype is set correctly)
    summary = generate_summary(prompt, model, tokenizer)

    return summary


# def analyze_dialogue(file_path, model, tokenizer):
#     """
#     Analyze dialogue data and generate a summary.

#     Args:
#         file_path (str or Path): Path to the input file (CSV or JSON).
#         model: The language model for summarization.
#         tokenizer: The tokenizer for the language model.

#     Returns:
#         str: A summary of the dialogue.
#     """
#     # Load dialogue data
#     dialogue_data = load_dialogue_data(file_path)

#     dialogue_text = format_dialogue_for_summary(df)
#     # Format dialogue for input
#     dialogue_text = "\n".join(
#         [
#             f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
#             for _, row in dialogue_data.iterrows()
#         ]
#     )

#     # Refine the prompt for better summarization
#     prompt = (
#         "Summarize the following dialogue and extract the main topics. "
#         "Identify the key points discussed by each speaker and provide a concise summary.\n\n"
#         f"{dialogue_text}"
#     )

#     # # Tokenize the prompt
#     # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     # # Generate summary using the model
#     # outputs = model.generate(**inputs, max_length=1500)

#     # # Decode the generated tokens into text
#     # summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Generate summary (ensuring dtype is set correctly)
#     with torch.no_grad():
#         summary = generate_summary(prompt, model, tokenizer)

#     return summary


if __name__ == "__main__":
    # python -m speech_analyzer.dialogue_analyzer
    AUDIO_FILE_NAME = env_as_path("AUDIO_FILE_NAME", "ZOOM067.wav")
    csv_file = Path(f"{os.path.basename(AUDIO_FILE_NAME)}.csv")
    models = load_all_models()
    model, tokenizer = models[env_as_str("AI_MODEL_NAME", "deepseek")]
    summary = analyze_dialogue(csv_file, model, tokenizer)
    logger.debug(f"Summary oof the text: {summary}")

    # Load the CSV and display statistics
    df = load_csv_data_for_model(csv_file, show_stats=True)
    # Format dialogue for summary
    # Format dialogue data from CSV for summarization
    dialogue_text = format_dialogue_for_summary(df)

    # Show the first few rows of the data
    logger.debug("\nSample Data:")
    logger.info(df.head())
    logger.info(df.describe())

    # Example of tokenizing the 'transcription' column from the CSV
    tokenized_data = tokenize_text_data(df, column_name="transcription")
    logger.debug(tokenized_data)

    # X, Y = load_csv_for_neural_network(csv_file_path, target_column="transcription")
    # Load data for neural networks
    # X, Y = load_csv_for_neural_network(csv_file, target_column="transcription")
    # logger.debug(f"\nTensor Data: X={X}, Y={Y}")
    # # logger.debug("\nTensor transcription Data:")
    # # logger.info(f"\nTensor transcription Data: X: {X}")
    # # logger.debug(f"\nTensor transcription Data: Y: {X}")
    # logger.debug("Data Statistics X:")
    # logger.info(pd.DataFrame(X).describe())
    # logger.debug("Data Statistics Y:")
    # logger.info(pd.DataFrame(Y).describe())

    # Generate a summary for the dialogue
    model_version = os.getenv("GPT_API_VERSION_1", "gpt-3.5")
    model, tokenizer = load_gpt_model(model_version)
    summary = generate_summary(dialogue_text, model, tokenizer)

    logger.debug("Generated Summary:\n")
    logger.info(f"Summary generated by {model_version}:\n", summary)
