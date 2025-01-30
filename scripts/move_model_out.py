import os
import shutil
from pathlib import Path

from loguru import logger

from speech_parser.utils.env import env_as_bool, env_as_path, env_as_str


def move_model_out(model_name):

    # Define paths
    # source_base = Path("./models")
    # dest_base = Path("I:/DEV/models")

    MODEL_ENABLE_COPY = env_as_bool("MODEL_ENABLE_COPY", "True")  # TODO: copy=0, delete=1 simply deletes
    MOVE_DELETE = env_as_bool("MODEL_MOVE_AND_DELETE", "False")

    source_base = env_as_path("MODEL_DIR", "./models")
    dest_base = env_as_path("MODEL_OUTER_DIR", "I:/DEV/models")

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
    # model_mapping = {
    #     "llama": "ai/llama/llama-2-7b",
    #     "mistral": "ai/mistral/mistral-7b-v0.1",
    #     "falcon": "ai/falcon/falcon-7b",
    #     "deepseek": "ai/deepseek/DeepSeek-R1-Distill-Llama-8B",
    #     "stable_diffusion": "stable_diffusion/stable-diffusion-v1-5",
    # }

    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}")

    # source_path = source_base / model_mapping[model_name]
    # dest_path = dest_base / model_mapping[model_name]

    source_path = Path(source_base, model_mapping[model_name])
    dest_path = Path(dest_base, model_mapping[model_name])

    # Ensure destination directory exists
    dest_path.mkdir(parents=True, exist_ok=True)

    if MODEL_ENABLE_COPY:
        # Copy files
        logger.debug(f"Moving {model_name} from {source_path} to {dest_path}...")
        # Do only copy now
        for item in source_path.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, dest_path / item.name, dirs_exist_ok=True)

        logger.debug(f"Model {model_name} moved successfully.")

    if MOVE_DELETE:
        for item in source_path.iterdir():
            if item.is_file():
                os.remove(item)
            elif item.is_dir():
                shutil.rmtree(item)

        logger.debug(f"Model {model_name} deleted successfully.")


if __name__ == "__main__":
    # python -m scripts.move_model_out "llama"
    # python -m scripts.move_model_out llama
    import sys

    if len(sys.argv) != 2:
        print("Usage: python move_model_out.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    move_model_out(model_name)


# LLAMA_MODEL_PATH_ENV = env_as_path("LLAMA_MODEL_PATH", "./models/ai/llama/llama-2-7b")
# FALCON_MODEL_PATH_ENV = env_as_path("FALCON_MODEL_PATH", "./models/ai/falcon/falcon-7b")
# DEEPSEEK_MODEL_PATH_ENV = env_as_path(
#     "DEEPSEEK_MODEL_PATH", "./models/ai/deepseek/DeepSeek-R1-Distill-Llama-8B"
# )
# STABLE_DIFFUSION_MODEL_PATH_ENV = env_as_path(
#     "STABLE_DIFFUSION_MODEL_PATH", "./models/stable_diffusion/stable-diffusion-v1-5"
# )
