# speech_analyzer/dialogue_analyzer.py
import pandas as pd
from loguru import logger
from .gpt_loader import generate_summary


def load_dialogue_data(csv_file: str) -> pd.DataFrame:
    """Load the dialogue CSV file into a DataFrame."""
    df = pd.read_csv(csv_file)
    logger.debug(f"CSV loaded. Columns: {df.columns.tolist()}")
    return df


def format_dialogue_for_summary(df: pd.DataFrame) -> str:
    """Format dialogue rows for summarization prompt."""
    dialogue_text = "\n".join(
        [
            f"{row['speaker']} ({row['start_time']}-{row['end_time']}): {row['transcription']}"
            for _, row in df.iterrows()
            if pd.notna(row["transcription"])
        ]
    )
    return dialogue_text


def analyze_dialogue(csv_file: str, model_version: str, tokenizer) -> str:
    """
    Analyze dialogue data from a CSV file and generate a summary.
    """
    df = load_dialogue_data(csv_file)
    dialogue_text = format_dialogue_for_summary(df)
    prompt = (
        "Summarize the following dialogue and extract the main topics. "
        "Identify the key points discussed by each speaker and provide a concise summary.\n\n"
        f"{dialogue_text}"
    )
    summary = generate_summary(prompt, model_version, tokenizer)
    return summary
