import os
from loguru import logger
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from vosk import Model as VoskModel
from diffusers import StableDiffusionPipeline  # For Stable Diffusion
from dotenv import load_dotenv
import subprocess
from models.download_model import check_and_download_model
from speech_analyzer.gpt_loader import load_gpt_model
from speech_parser.utils.env import env_as_bool
from unsloth import FastLanguageModel

# from speech_analyzer.custom_llama import LlamaModel_fast_forward_inference

USE_4BIT = env_as_bool("USE_4BIT", "True")
USE_SAFETENSORS = env_as_bool("USE_SAFETENSORS", "True")
USE_CPU_OFFLOAD = env_as_bool("USE_SAFETENSORS", "False")


def load_only_one_model():
    # TODO: to not load RAM with unused models cache in memory, also makes sure that in the end of all runs clear {USERS}/.cache
    # return function generator - appropriate loader func
    pass


def load_all_models():
    """
    List available models for dynamic loading.

    Returns:
        dict: A dictionary of available models and their corresponding loading functions.
    """
    return {
        # "vosk": lambda: load_vosk_model(), # idk
        "gpt-3.5": lambda: load_gpt_model(api_version="gpt-3.5"),
        "gpt-4": lambda: load_gpt_model(api_version="gpt-4"),
        # # NOTE: Unsloth loader - compiled to C (weights lesser - but need specific setup)
        "mistral": lambda: unsloth_model_loader("mistral"),
        # "llama": lambda: unsloth_model_loader("llama"),
        # "falcon": unsloth_model_loader(os.getenv("FALCON_MODEL_NAME"), os.getenv("FALCON_MODEL_PATH")),
        # "deepseek": unsloth_model_loader(os.getenv("DEEPSEEK_MODEL_NAME"), os.getenv("DEEPSEEK_MODEL_PATH")),
        # Classic loader
        # "llama": load_language_model(os.getenv("LLAMA_MODEL_NAME"), os.getenv("LLAMA_MODEL_PATH")),
        "mistral-hf": hugginface_model_loader(
            os.getenv("MISTRAL_MODEL_NAME"), os.getenv("MISTRAL_MODEL_PATH")
        ),
        # "falcon": load_language_model(os.getenv("FALCON_MODEL_NAME"), os.getenv("FALCON_MODEL_PATH")),
        # "deepseek": load_language_model(os.getenv("DEEPSEEK_MODEL_NAME"), os.getenv("DEEPSEEK_MODEL_PATH")),
        # Add more models as needed
        "stable_diffusion": load_stable_diffusion_model(),
    }


def load_model_object(model_name_key: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    load_dotenv()
    models = {
        "vosk": load_vosk_model(),
        "llama": hugginface_model_loader(os.getenv("LLAMA_MODEL_NAME"), os.getenv("LLAMA_MODEL_PATH")),
        "mistral": hugginface_model_loader(os.getenv("MISTRAL_MODEL_NAME"), os.getenv("MISTRAL_MODEL_PATH")),
        "falcon": hugginface_model_loader(os.getenv("FALCON_MODEL_NAME"), os.getenv("FALCON_MODEL_PATH")),
        "deepseek": hugginface_model_loader(
            os.getenv("DEEPSEEK_MODEL_NAME"), os.getenv("DEEPSEEK_MODEL_PATH")
        ),
        "stable_diffusion": load_stable_diffusion_model(),
    }
    # if model_name_key == 'vosk':
    #     return models["vosk"].value
    if model_name_key == models[model_name_key]:
        return models[model_name_key][0], models[model_name_key][1]
    else:
        return logger.debug("Not a valid model key")


def load_model_by_key(model_key):
    """
    Load the appropriate model by key.

    Args:
        model_key (str): The key to specify the model to be loaded.

    Returns:
        tuple: Loaded model and tokenizer (if applicable).
    """
    models = load_all_models()
    if model_key in models:
        return models[model_key]()
    else:
        raise ValueError(f"Model '{model_key}' is not available. Available options: {list(models.keys())}")


def unsloth_model_loader(model_name_key, max_seq_length=2048, dtype=None, load_in_4bit=True):
    """
    Load a model using the unsloth library for faster and more efficient inference.

    Args:
        model_name_key (str): Key to identify the model from environment variables (e.g., "mistral").
        max_seq_length (int): Maximum sequence length.
        dtype: Data type for computation (e.g., torch.float16 or torch.float32).
        load_in_4bit (bool): Whether to use 4-bit quantization.

    Returns:
        model: The loaded model.
        tokenizer: The tokenizer.
    """
    model_name = os.getenv(f"{model_name_key.upper()}_MODEL_REAL_NAME")
    model_path = os.getenv(f"{model_name_key.upper()}_MODEL_PATH")
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    # Ensure dtype is a valid PyTorch dtype
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_bf16_supported() else torch.float32

    # Ensure dtype is a valid PyTorch dtype
    # dtype = torch.float16 if dtype == "float16" else torch.float32
    # dtype = torch.bfloat16 if dtype == "float16" else torch.bfloat16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=os.getenv("AUTH_TOKEN_HUGGINFACE"),
    )

    # Enable inference mode
    model = FastLanguageModel.for_inference(model)

    logger.debug(f"Using: Loader: Unsloth  Model: {model_name}")
    logger.info(f"Located at Path: {model_path}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model, tokenizer


def load_vosk_model() -> VoskModel:
    model_path = os.getenv("VOSK_MODEL_PATH")
    model_name = os.getenv("VOSK_MODEL_NAME")
    model_path = check_and_download_model(model_name, model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at {model_path}")
    return VoskModel(model_path)


def hugginface_model_loader(model_name, model_path) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Disable 4-bit quantization
    quantization_config = None  # No quantization

    if USE_4BIT:
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Enable nested quantization for better memory efficiency
            bnb_4bit_quant_type="nf4",  # Use 4-bit NormalFloat quantization
            bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation
        )
    # Enable CPU offloading
    if USE_CPU_OFFLOAD:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=False,  # Disable 4-bit quantization
            llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        token=os.getenv("AUTH_TOKEN_HUGGINFACE"),
        use_safetensors=env_as_bool("USE_SAFETENSORS", "True"),  # Force PyTorch format
        # load_in_4bit=env_as_bool("USE_4BIT", "True"),  # Enable 4-bit quantization
        quantization_config=quantization_config,  # Disable quantization
    )
    return model, tokenizer


def load_stable_diffusion_model():
    model_path = os.getenv("STABLE_DIFFUSION_MODEL_PATH")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Stable Diffusion model not found at {model_path}")
    pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline.to("cuda")
    return pipeline


# def generate_summary(prompt, model, tokenizer):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # Move inputs to model's device
#     outputs = model.generate(**inputs, max_length=1500)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_code(prompt):
    # prompt = "Write a Python function to calculate the Fibonacci sequence."
    # print(generate_code(prompt))
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_command_generator(model_name, model_name_key, token):
    if model_name_key == "llama":
        command = f"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token='{token}').save_pretrained('./models/ai/llama/{model_name}')"
    if model_name_key == "mistral":
        command = f"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token='{token}').save_pretrained('./models/ai/llama/{model_name}')"

    logger.debug(command)
    return command


def cmd_loader():
    load_dotenv()
    model_name = os.getenv("LLAMA_MODEL_NAME")  # "FALCON_MODEL_NAME"
    token = os.getenv("AUTH_TOKEN_HUGGINFACE")
    command = load_command_generator(model_name, "llama", token)
    full_command = f'python -c "{command}"'
    logger.debug(full_command)
    subprocess.run(full_command, shell=True)


if __name__ == "__main__":
    # python -m speech_analyzer.model_loader
    # models = load_all_models()
    cmd_loader()


# !pip install unsloth
# Also get the latest nightly Unsloth!
# !pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
# from unsloth import FastLanguageModel
# import torch

# max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
# dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Llama-3.1 2x faster
#     "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
#     "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
#     "unsloth/Meta-Llama-3.1-405B-bnb-4bit",  # 4bit for 405b!
#     "unsloth/Mistral-Small-Instruct-2409",  # Mistral 22b 2x faster!
#     "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
#     "unsloth/Phi-3.5-mini-instruct",  # Phi-3.5 2x faster!
#     "unsloth/Phi-3-medium-4k-instruct",
#     "unsloth/gemma-2-9b-bnb-4bit",
#     "unsloth/gemma-2-27b-bnb-4bit",  # Gemma 2x faster!
#     "unsloth/Llama-3.2-1B-bnb-4bit",  # NEW! Llama 3.2 models
#     "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
#     "unsloth/Llama-3.2-3B-bnb-4bit",
#     "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
#     "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",  # NEW! Llama 3.3 70B!
# ]  # More models at https://huggingface.co/unsloth

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/Llama-3.2-3B-Instruct",  # or choose "unsloth/Llama-3.2-1B-Instruct"
#     max_seq_length=max_seq_length,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )
