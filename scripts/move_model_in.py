import shutil
from pathlib import Path

from loguru import logger

from speech_parser.speech_parser import env_as_path
from speech_parser.utils.env import env_as_str

# This script moves a model from I: drive to the local C: drive.


def move_model_in(model_name):
    # Define paths
    # source_base = Path("I:/DEV/models")
    # dest_base = Path("./models")

    source_base = env_as_path("MODEL_OUTER_DIR", "I:/DEV/models")
    dest_base = env_as_path("MODEL_DIR", "./models")

    LLAMA_MODEL_NAME = env_as_str("LLAMA_MODEL_NAME", "llama-2-7b")
    MISTRAL_MODEL_NAME = env_as_str("MISTRAL_MODEL_NAME", "mistral-7b-v0.3")
    FALCON_MODEL_NAME = env_as_str("FALCON_MODEL_NAME", "falcon-7b")
    DEEPSEEK_MODEL_NAME = env_as_str("DEEPSEEK_MODEL_NAME", "DeepSeek-R1-Distill-Llama-8B")
    STABLE_DIFFUSION_MODEL_NAME = env_as_str("STABLE_DIFFUSION_MODEL_NAME", "stable-diffusion-v1-5")

    # Map model names to their respective folders
    model_mapping = {
        "llama": f"ai/llama/{LLAMA_MODEL_NAME}",
        "mistral": f"ai/mistral/{MISTRAL_MODEL_NAME}",
        "falcon": f"ai/falcon/{FALCON_MODEL_NAME}",
        "deepseek": f"ai/deepseek/{DEEPSEEK_MODEL_NAME}",
        "stable_diffusion": f"stable_diffusion/{STABLE_DIFFUSION_MODEL_NAME}",
    }

    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}")

    source_path = Path(source_base, model_mapping[model_name])
    dest_path = Path(dest_base, model_mapping[model_name])

    # Ensure destination directory exists
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy files
    logger.debug(f"Moving {model_name} from {source_path} to {dest_path}...")
    for item in source_path.iterdir():
        if item.is_file():
            shutil.copy2(item, dest_path / item.name)
        elif item.is_dir():
            shutil.copytree(item, dest_path / item.name, dirs_exist_ok=True)

    logger.debug(f"Model {model_name} moved successfully.")


if __name__ == "__main__":
    # python -m scripts.move_model_in llama
    import sys

    if len(sys.argv) != 2:
        logger.debug("Usage: python move_model_in.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    move_model_in(model_name)
