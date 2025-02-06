# utils/cli.py
# import os
from pathlib import Path

# from dotenv import set_key
import subprocess
import sys
import os

from dotenv import load_dotenv
from loguru import logger


class TerminalDriver:
    def __init__(self, env_file=".env"):
        self.env_file = env_file
        self._load_env()
        if sys.platform in ("linux", "darwin"):
            self.clear = "clear"
        elif sys.platform == "win32":
            self.clear = "cls"
        else:
            self.clear = ""
            logger.debug("Platfrom not supported", file=sys.stderr)
            exit(1)

    def _load_env(self):
        """Load environment variables from the .env file."""
        load_dotenv(self.env_file)

    def clear_terminal(self) -> None:
        os.system(self.clear)

    def clear_terminal_subprocess(self):
        subprocess.run(self.clear, shell=True)


# ########################################################
# -- Audio management from CLI
# ########################################################


def list_audio_files(workspace: Path, extension=".wav"):
    """Return a sorted list of all audio files with the given extension in the workspace."""
    return sorted(workspace.glob(f"*{extension}"))


def select_audio_file(workspace: Path, extension=".wav"):
    """Prompts the user to select an audio file from the workspace."""
    files = list_audio_files(workspace, extension)
    if not files:
        print(f"No audio files with extension '{extension}' found in {workspace}")
        sys.exit(1)
    logger.debug("Select an audio file:")
    for i, file in enumerate(files, start=1):
        logger.info(f"{i}. {file.name}")
    try:
        choice = int(input("Enter the number of the audio file to use: "))
        if choice < 1 or choice > len(files):
            raise ValueError("Invalid choice")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    return files[choice - 1]


if __name__ == "__main__":
    terminal = TerminalDriver()
    terminal.clear_terminal()
    logger.debug("Terminal\Console Output cleared")
