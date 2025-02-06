import os
from pathlib import Path
import re
from dotenv import load_dotenv
from loguru import logger


def env_as_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() == "true"


def env_as_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default))


def env_as_int(name: str, default: int = 1) -> int:
    return int(os.getenv(name, default))


def env_as_float(name: str, default: float = 1.0) -> float:
    return float(os.getenv(name, default))


def env_as_path(name: str, default: Path = ".") -> Path:
    return Path(os.getenv(name, default))


def update_env_file(key: str, value: str, env_file: str = ".env"):
    """Updates the .env file with the provided key-value pair."""
    from dotenv import set_key

    env_path = Path(env_file)
    if not env_path.exists():
        env_path.write_text("")  # create empty .env if missing
    set_key(str(env_path), key, value)
    logger.debug(f"Updated {key} in {env_file} to '{value}'")


def filename_as_csv(audio_file_name):
    """Convert audio file name to CSV file name."""
    return f"{os.path.splitext(audio_file_name)[0]}.csv"


def convert_model_name(model_name, to_short=True):
    """
    Converts a Hugging Face model name to a shorter form or vice versa.

    Parameters:
    - model_name (str): The model name to convert.
    - to_short (bool): If True, converts to the short form. If False, converts back to the full form.

    Returns:
    - str: The converted model name.
    """
    if to_short:
        # Convert from "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit" to "Mistral-7B-Instruct-v0.3-bnb-4bit"
        match = re.search(r"[^/]+$", model_name)
        if match:
            return match.group(0)
        else:
            raise ValueError("Invalid model name format for shortening.")
    else:
        # Convert from "Mistral-7B-Instruct-v0.3-bnb-4bit" to "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit"
        # Assuming the prefix is "unsloth/" (you can modify this as needed)
        return f"unsloth/{model_name}"


def convert_model_name_lowercase(model_name, to_short=True):
    # Example usage:
    # model_name = "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit"
    # short_name = convert_model_name(model_name, to_short=True)
    # print(f"Short form: {short_name}")  # Output: mistral-7b-instruct-v0.3-bnb-4bit
    if to_short:
        match = re.search(r"[^/]+$", model_name)
        if match:
            return match.group(0).lower()  # Convert to lowercase
        else:
            raise ValueError("Invalid model name format for shortening.")
    else:
        return f"unsloth/{model_name}"


if __name__ == "__main__":
    # python -m speech_parser.utils.env
    load_dotenv()
    # Example usage:
    model_name = "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit"

    # Convert to short form
    short_name = convert_model_name(model_name, to_short=True)
    print(f"Short form: {short_name}")  # Output: Mistral-7B-Instruct-v0.3-bnb-4bit

    # Convert back to full form
    full_name = convert_model_name(short_name, to_short=False)
    print(f"Full form: {full_name}")  # Output: unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit


def get_model_env_vars(model_name_key):
    model_real_name = env_as_str(f"{model_name_key.upper()}_MODEL_REAL_NAME")
    model_path = env_as_path(f"{model_name_key.upper()}_MODEL_PATH")
    return model_real_name, model_path
