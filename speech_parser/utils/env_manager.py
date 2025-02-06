import os
from pathlib import Path
from dotenv import load_dotenv, set_key
import re
from loguru import logger


class EnvManager:
    def __init__(self, env_file=".env"):
        self.env_file = env_file
        self._load_env()

    def _load_env(self):
        """Load environment variables from the .env file."""
        load_dotenv(self.env_file)

    def _update_env_file(self, key: str, value: str):
        """Updates the .env file with the provided key-value pair."""
        env_path = Path(self.env_file)
        if not env_path.exists():
            env_path.write_text("")  # create empty .env if missing
        set_key(str(env_path), key, value)
        logger.debug(f"Updated {key} in {self.env_file} to '{value}'")

    # Get methods for various types
    def get_bool(self, key: str, default: bool = False) -> bool:
        return os.getenv(key, str(default)).lower() == "true"

    def get_str(self, key: str, default: str = "") -> str:
        return os.getenv(key, default)

    def get_int(self, key: str, default: int = 1) -> int:
        return int(os.getenv(key, default))

    def get_float(self, key: str, default: float = 1.0) -> float:
        return float(os.getenv(key, default))

    def get_path(self, key: str, default: Path = ".") -> Path:
        return Path(os.getenv(key, default))

    # Set methods for various types
    def set_bool(self, key: str, value: bool):
        self._update_env_file(key, str(value).lower())

    def set_str(self, key: str, value: str):
        self._update_env_file(key, value)

    def set_int(self, key: str, value: int):
        self._update_env_file(key, str(value))

    def set_float(self, key: str, value: float):
        self._update_env_file(key, str(value))

    def set_path(self, key: str, value: Path):
        self._update_env_file(key, str(value))

    # Reset to default
    def reset_to_default(self, key: str, default):
        """Resets the given environment variable to its default value."""
        if isinstance(default, bool):
            self.set_bool(key, default)
        elif isinstance(default, int):
            self.set_int(key, default)
        elif isinstance(default, float):
            self.set_float(key, default)
        elif isinstance(default, Path):
            self.set_path(key, default)
        else:
            self.set_str(key, default)

    # Utility functions for specific needs
    def filename_as_csv(self, audio_file_name: str) -> str:
        """Convert audio file name to CSV file name."""
        return f"{os.path.splitext(audio_file_name)[0]}.csv"

    def convert_model_name(self, model_name: str, to_short=True) -> str:
        """
        Converts a Hugging Face model name to a shorter form or vice versa.
        """
        if to_short:
            match = re.search(r"[^/]+$", model_name)
            if match:
                return match.group(0)
            else:
                raise ValueError("Invalid model name format for shortening.")
        else:
            return f"unsloth/{model_name}"

    def convert_model_name_lowercase(self, model_name: str, to_short=True) -> str:
        """
        Converts model name to lowercase.
        """
        if to_short:
            match = re.search(r"[^/]+$", model_name)
            if match:
                return match.group(0).lower()
            else:
                raise ValueError("Invalid model name format for shortening.")
        else:
            return f"unsloth/{model_name}"

    def get_model_env_vars(self, model_name_key: str):
        """
        Retrieves the model's real name and path based on the environment variables.
        """
        model_real_name = self.get_str(f"{model_name_key.upper()}_MODEL_REAL_NAME")
        model_path = self.get_path(f"{model_name_key.upper()}_MODEL_PATH")
        return model_real_name, model_path


# Usage example:

if __name__ == "__main__":
    env_manager = EnvManager()

    # Get environment variables
    logger.debug("Initial env values: ")
    logger.info(env_manager.get_str("AI_MODEL_NAME"))
    logger.debug(env_manager.get_bool("ENABLE_AI"))
    logger.info(env_manager.get_path("OUTPUT_DIR"))

    # Set environment variables
    env_manager.set_str("AI_MODEL_NAME", "gpt-4")
    env_manager.set_bool("ENABLE_AI", True)
    logger.debug("Setting env values pair: ( AI_MODEL_NAME, ENABLE_AI )")
    logger.info(env_manager.get_str("AI_MODEL_NAME"))
    logger.debug(env_manager.get_bool("ENABLE_AI"))

    # Reset environment variables to defaults
    env_manager.reset_to_default("AI_MODEL_NAME", "google-gemini")
    env_manager.reset_to_default("ENABLE_AI", False)
    logger.debug("Resetting env values...")
    logger.info(env_manager.get_str("AI_MODEL_NAME"))
    logger.debug(env_manager.get_bool("ENABLE_AI"))

    # Model name handling example
    short_name = env_manager.convert_model_name("unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit", to_short=True)
    logger.debug(f"Short form: {short_name}")
    full_name = env_manager.convert_model_name(short_name, to_short=False)
    logger.info(f"Full form: {full_name}")

    # Getting model environment variables
    real_name, model_path = env_manager.get_model_env_vars("LLAMA")
    logger.debug(f"Model real name: {real_name}, Path: {model_path}")
