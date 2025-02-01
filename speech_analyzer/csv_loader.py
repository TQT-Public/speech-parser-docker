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


def format_dialogue_for_summary(df):
    """
    Format the dialogue data for summarization by combining speaker, transcription, and timestamps.

    Args:
        df (pd.DataFrame): DataFrame containing dialogue data with 'speaker', 'transcription', 'start_time', and 'end_time'.

    Returns:
        str: A formatted dialogue string for use as a prompt for language models.
    """
    dialogue_text = "\n".join(
        [
            f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
            for _, row in df.iterrows()
            if pd.notna(row["transcription"])
        ]
    )
    return dialogue_text


# def tokenize_text_data(df, column_name="transcription"):
#     """
#     Tokenizes the text data in the given column for use in neural networks.

#     Args:
#         df (pd.DataFrame): DataFrame containing text data.
#         column_name (str): Name of the column containing the text to be tokenized.

#     Returns:
#         dict: Dictionary containing tokenized text data ready for neural network input.
#     """
#     # Filter out rows where the 'transcription' column is empty or NaN
#     df = df[df[column_name].notna() & df[column_name].str.strip().astype(bool)]

#     # Log a warning if any empty rows were filtered out
#     if len(df) == 0:
#         raise ValueError(f"No valid data found in the '{column_name}' column for tokenization.")

#     tokenizer = AutoTokenizer.from_pretrained("gpt2")

#     # Add padding token if not present
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # Define PAD token if missing

#     # Tokenize the text
#     tokenized_data = tokenizer(df[column_name].tolist(), padding=True, truncation=True, return_tensors="pt")

#     return tokenized_data


# def load_csv_data_for_model(csv_file_path, show_stats=True):
#     """
#     Load CSV data, show basic statistics, and prepare it for use with language models.

#     Args:
#         csv_file_path (str): Path to the CSV file.
#         show_stats (bool): If True, show basic statistics like missing values and data types.

#     Returns:
#         pd.DataFrame: Data loaded into a Pandas DataFrame.
#     """
#     if not os.path.exists(csv_file_path):
#         raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

#     # Load CSV file into a DataFrame
#     df = pd.read_csv(csv_file_path)

#     if show_stats:
#         # Display data types
#         print("\nData Types:")
#         print(df.dtypes)

#         # Display missing values
#         print("\nMissing Values:")
#         print(df.isnull().sum())

#         # Display basic statistics (for numeric columns)
#         print("\nStatistics Summary:")
#         print(df.describe(include="all"))  # include='all' shows both numeric and categorical summary

#     return df


# def load_csv_for_neural_network(csv_file_path, target_column=None, normalize=False, show_stats=True):
#     """
#     Load CSV data and prepare it for neural network processing.

#     Args:
#         csv_file_path (str): Path to the CSV file.
#         target_column (str): Column name of the target labels (optional).
#         normalize (bool): Whether to normalize the data or not.
#         show_stats (bool): Whether to show dataset statistics (mean, std, etc.).

#     Returns:
#         X (Tensor): Feature data for neural network (PyTorch Tensor).
#         y (Tensor): Target labels (PyTorch Tensor), if provided.
#     """
#     # Load the CSV data into a pandas DataFrame
#     df = pd.read_csv(csv_file_path)

#     # Show statistics if requested
#     if show_stats:
#         logger.debug("Data Statistics:")
#         logger.info(df.describe())  # Shows basic statistics: count, mean, std, min, max, etc.

#     # Drop non-numeric columns like transcription for feature extraction
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

#     # Separate features (X) and target (y)
#     if target_column:
#         X = df[numeric_cols].drop(columns=[target_column]).values  # Drop the target column, keep features
#         y = df[target_column].values  # Extract the target column
#     else:
#         X = df[numeric_cols].values  # Use only numeric columns as features if no target column is specified
#         y = None  # No target labels

#     # Optional: Normalize the data (feature scaling)
#     if normalize:
#         X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

#     # Convert data to PyTorch tensors
#     X_tensor = torch.tensor(X, dtype=torch.float32)

#     if y is not None:
#         y_tensor = torch.tensor(y, dtype=torch.float32)  # Convert target labels to tensor
#         return X_tensor, y_tensor
#     else:
#         return X_tensor


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
    # TODO: fix KeyError: "['transcription'] not found in axis"
    # X, Y = load_csv_for_neural_network(csv_file_path, target_column="transcription")
    # logger.debug(f"\nTensor Data: X={X}, Y={Y}")
    # # logger.debug("\nTensor transcription Data:")
    # # logger.info(f"\nTensor transcription Data: X: {X}")
    # # logger.debug(f"\nTensor transcription Data: Y: {X}")
    # logger.debug("Data Statistics X:")
    # logger.info(pd.DataFrame(X).describe())
    # logger.debug("Data Statistics Y:")
    # logger.info(pd.DataFrame(Y).describe())
