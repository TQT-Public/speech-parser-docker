import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from speech_parser.utils.env import env_as_bool, env_as_path

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",  # device_map="cpu",  # Force CPU
    token=os.getenv("AUTH_TOKEN_HUGGINFACE"),
    use_safetensors=env_as_bool("USE_SAFETENSORS", "True"),  # False = Force PyTorch format
    load_in_4bit=env_as_bool("USE_4BIT", "True"),  # Enable 4-bit quantization
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", token=os.getenv("AUTH_TOKEN_HUGGINFACE")
)
# model.save_pretrained("./models/ai/llama/llama-2-7b")
# tokenizer.save_pretrained("./models/ai/llama/llama-2-7b")

model.save_pretrained(env_as_path("LLAMA_MODEL_PATH", "./models/ai/llama/llama-2-7b"))
tokenizer.save_pretrained(env_as_path("LLAMA_MODEL_PATH", "./models/ai/llama/llama-2-7b"))
