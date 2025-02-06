# utils/config.py
from speech_parser.utils.env_manager import EnvManager
from pathlib import Path

# Initialize the EnvManager
env_manager = EnvManager(env_file=".env")

# General settings
DRY_RUN = env_manager.get_bool("DRY_RUN", False)
ASSIGNSPEAKERS = env_manager.get_bool("ASSIGNSPEAKERS", False)
FILTER_UNNECESSARY_RTTM = env_manager.get_bool("FILTER_UNNECESSARY_RTTM", True)
CLEAR_AUDIO_PARTS_RUN = env_manager.get_bool("CLEAR_AUDIO_PARTS_RUN", True)
ENABLE_VOSK_LOGS = env_manager.get_bool("ENABLE_VOSK_LOGS", False)
ENABLE_VOSK_GPU = env_manager.get_bool("ENABLE_VOSK_GPU", True)
ENABLE_AUDIO_SPLIT_LOGS = env_manager.get_bool("ENABLE_AUDIO_SPLIT_LOGS", True)
ADD_PUNCTUATION = env_manager.get_bool("ADD_PUNCTUATION", False)
CREATE_PICTURE = env_manager.get_bool("CREATE_PICTURE", True)
FASTER_DECODING = env_manager.get_bool("FASTER_DECODING", False)
USE_CUSTOM_VOSK = env_manager.get_bool("USE_CUSTOM_VOSK", True)
USE_BATCHES = env_manager.get_bool("USE_BATCHES", False)
ENABLE_AI = env_manager.get_bool("ENABLE_AI", True)
USE_LOCAL_MODEL = env_manager.get_bool("USE_LOCAL_MODEL", False)
MODEL_LOAD_TYPE = env_manager.get_str("MODEL_LOAD_TYPE", "api")
AI_MODEL_NAME = env_manager.get_str("AI_MODEL_NAME", "google-gemini")

# Numeric settings
MAX_PROCESSES = env_manager.get_int("MAX_PROCESSES", 3)
BATCH_SIZE = env_manager.get_float("BATCH_SIZE", 10.0)
MIN_RTTM_DURATION = env_manager.get_float("MIN_RTTM_DURATION", 3.0)
CUSTOM_VOSK_BEAM = env_manager.get_float("CUSTOM_VOSK_BEAM", 15.0)
CUSTOM_VOSK_MAX_ACTIVE = env_manager.get_float("CUSTOM_VOSK_MAX_ACTIVE", 10000)
CUSTOM_VOSK_LATTICE_BEAM = env_manager.get_float("CUSTOM_VOSK_LATTICE_BEAM", 8.0)
VOSK_SAMPLE_RATE = env_manager.get_int("VOSK_SAMPLE_RATE", 16000)
VOSK_MAX_CHUNK_SIZE = env_manager.get_int("VOSK_MAX_CHUNK_SIZE", 20000)
TOKEN_LIMIT = env_manager.get_int("TOKEN_LIMIT", 4096)

# Audio and output paths
AUDIOWORKSPACE = env_manager.get_path("AUDIOWORKSPACE", "./audio_files")
WORKSPACE = env_manager.get_path("WORKSPACE", "./sources")
AUDIO_FILE_NAME = env_manager.get_str("AUDIO_FILE_NAME", "ZOOM0067.wav")
OUTPUT_DIR = env_manager.get_path("OUTPUT_DIR", "./output")
OUTPUT_DIR_PARTS = env_manager.get_path("OUTPUT_DIR_PARTS", "./audio_files/parts")

# Vosk model configuration
VOSK_MODEL_NAME = env_manager.get_str("VOSK_MODEL_NAME", "vosk-model-ru-0.42")
VOSK_MODEL_PATH = env_manager.get_path("VOSK_MODEL_PATH", f"./models/vosk/{VOSK_MODEL_NAME}")
VOSK_MODEL_FULL_PATH = VOSK_MODEL_PATH

# AI models configuration (paths, names)
MODELS_DIR = env_manager.get_path("MODELS_DIR", "./models")
MODEL_OUTER_DIR = env_manager.get_path("MODEL_OUTER_DIR", "I:/DEV/models")
MODEL_ENABLE_COPY = env_manager.get_bool("MODEL_ENABLE_COPY", True)
MODEL_MOVE_AND_DELETE = env_manager.get_bool("MODEL_MOVE_AND_DELETE", True)

# LLama, Mistral, Falcon, DeepSeek, Stable Diffusion models configuration
LLAMA_MODEL_NAME = env_manager.get_str("LLAMA_MODEL_NAME", "llama-2-7b")
LLAMA_MODEL_REAL_NAME = env_manager.get_str("LLAMA_MODEL_REAL_NAME", "unsloth/Meta-Llama-3.1-8B-bnb-4bit")
LLAMA_MODEL_PATH = env_manager.get_path("LLAMA_MODEL_PATH", f"{MODELS_DIR}/ai/llama/{LLAMA_MODEL_NAME}")

MISTRAL_MODEL_NAME = env_manager.get_str("MISTRAL_MODEL_NAME", "mistral-7b-v0.3")
MISTRAL_MODEL_REAL_NAME = env_manager.get_str(
    "MISTRAL_MODEL_REAL_NAME", "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit"
)
MISTRAL_MODEL_PATH = env_manager.get_path(
    "MISTRAL_MODEL_PATH", f"{MODELS_DIR}/ai/mistral/{MISTRAL_MODEL_NAME}"
)

FALCON_MODEL_NAME = env_manager.get_str("FALCON_MODEL_NAME", "falcon-7b")
FALCON_MODEL_REAL_NAME = env_manager.get_str("FALCON_MODEL_REAL_NAME", "tiiuae/falcon-7b")
FALCON_MODEL_PATH = env_manager.get_path("FALCON_MODEL_PATH", f"{MODELS_DIR}/ai/falcon/{FALCON_MODEL_NAME}")

DEEPSEEK_MODEL_NAME = env_manager.get_str("DEEPSEEK_MODEL_NAME", "DeepSeek-R1-Distill-Llama-8B")
DEEPSEEK_MODEL_REAL_NAME = env_manager.get_str(
    "DEEPSEEK_MODEL_REAL_NAME", "unsloth/DeepSeek-R1-Distill-Llama-8B"
)
DEEPSEEK_MODEL_PATH = env_manager.get_path(
    "DEEPSEEK_MODEL_PATH", f"{MODELS_DIR}/ai/deepseek/{DEEPSEEK_MODEL_NAME}"
)

STABLE_DIFFUSION_MODEL_NAME = env_manager.get_str("STABLE_DIFFUSION_MODEL_NAME", "stable-diffusion-v1-5")
STABLE_DIFFUSION_MODEL_REAL_NAME = env_manager.get_str(
    "STABLE_DIFFUSION_MODEL_REAL_NAME", "runwayml/stable-diffusion-v1-5"
)
STABLE_DIFFUSION_MODEL_PATH = env_manager.get_path(
    "STABLE_DIFFUSION_MODEL_PATH", f"{MODELS_DIR}/stable_diffusion/{STABLE_DIFFUSION_MODEL_NAME}"
)

# HuggingFace settings
USE_SAFETENSORS = env_manager.get_bool("USE_SAFETENSORS", True)
USE_4BIT = env_manager.get_bool("USE_4BIT", False)
USE_CPU_OFFLOAD = env_manager.get_bool("USE_CPU_OFFLOAD", True)

# API Keys and settings
GPT_API_VERSION_1 = env_manager.get_str("GPT_API_VERSION_1", "gpt-3.5-turbo")
GPT_API_VERSION_2 = env_manager.get_str("GPT_API_VERSION_2", "gpt-4o")
CUDA_VISIBLE_DEVICES = env_manager.get_str("CUDA_VISIBLE_DEVICES", "0")

# Path settings
CONFIG_PATH = env_manager.get_path("CONFIG_PATH", "./configs")
SCRIPT_PATH = env_manager.get_path("SCRIPT_PATH", "./scripts")
