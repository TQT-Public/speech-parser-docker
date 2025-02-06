# speech_analyzer/csv_loader.py
import os
import pandas as pd
import torch
from transformers import AutoTokenizer
from loguru import logger


def load_csv_data_for_model(csv_file_path: str, show_stats: bool = True) -> pd.DataFrame:
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    if show_stats:
        logger.info("Data Types:")
        logger.info(df.dtypes)
        logger.info("Missing Values:")
        logger.info(df.isnull().sum())
        logger.info("Statistics Summary:")
        logger.info(df.describe(include="all"))
    return df


def tokenize_text_data(df: pd.DataFrame, column_name: str = "transcription"):
    # Filter out empty rows
    df = df[df[column_name].notna() & (df[column_name].str.strip() != "")]
    if df.empty:
        raise ValueError(f"No valid data found in column {column_name}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenized_data = tokenizer(df[column_name].tolist(), padding=True, truncation=True, return_tensors="pt")
    return tokenized_data


def load_csv_for_neural_network(
    csv_file_path: str, target_column: str = None, normalize: bool = False, show_stats: bool = True
):
    df = pd.read_csv(csv_file_path)
    if show_stats:
        logger.info(df.describe(include="all"))
    if target_column:
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
    else:
        X = df.values
        y = None
    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    try:
        X_tensor = torch.tensor(X, dtype=torch.float32)
    except Exception as e:
        logger.error(f"Error converting features to tensor: {e}")
        raise
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor
    return X_tensor


def format_dialogue_for_summary(df, delimiter="\n"):
    """
    Format the dialogue data for summarization by combining speaker, transcription, and timestamps.

    Args:
        df (pd.DataFrame): DataFrame containing dialogue data with 'speaker', 'transcription', 'start_time', and 'end_time'.
        delimiter (str): The delimiter to use between dialogue lines (default is newline).

    Returns:
        str: A formatted dialogue string for use as a prompt for language models.
    """
    dialogue_text = delimiter.join(
        [
            f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
            for _, row in df.iterrows()
            if pd.notna(row["transcription"])
        ]
    )
    return dialogue_text


if __name__ == "__main__":
    # python -m speech_analyzer.csv_loader
    csv_file_path = "./audio_files/ZOOM0067.csv"

    # Load the CSV and display statistics
    df = load_csv_data_for_model(csv_file_path, show_stats=True)
    # Format dialogue for summary
    # Format dialogue data from CSV for summarization
    dialogue_text = format_dialogue_for_summary(df)

    # Show the first few rows of the data
    logger.debug("\nSample Data:")
    logger.info(df.head())
    logger.info(df.describe())

    # Example of tokenizing the 'transcription' column from the CSV
    tokenized_data = tokenize_text_data(df, column_name="transcription")
    logger.debug(f"{str(tokenized_data)}")

    # X, Y = load_csv_for_neural_network(csv_file_path, target_column="transcription")
    # Load data for neural networks
    # X, Y = load_csv_for_neural_network(csv_file_path, target_column="transcription")
    # logger.debug(f"\nTensor Data: X={X}, Y={Y}")
    # # logger.debug("\nTensor transcription Data:")
    # # logger.info(f"\nTensor transcription Data: X: {X}")
    # # logger.debug(f"\nTensor transcription Data: Y: {X}")
    # logger.debug("Data Statistics X:")
    # logger.info(pd.DataFrame(X).describe())
    # logger.debug("Data Statistics Y:")
    # logger.info(pd.DataFrame(Y).describe())
