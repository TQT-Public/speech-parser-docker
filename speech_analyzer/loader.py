# speech_analyzer/loader.py
import os
import time
import torch
import requests
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# BitsAndBytesConfig
# from vosk import Model as VoskModel
# from unsloth import FastLanguageModel
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv

from speech_analyzer.promt_utils import calculate_token_count
from speech_parser.utils.env_manager import EnvManager

# Load environment variables
load_dotenv()
load_dotenv(".env")
# Initialize the environment using EnvManager
env_manager = EnvManager(env_file=".env")

# Define basic token and rate limits (adjust as needed)
TOKEN_LIMITS = {
    "gpt-3.5-turbo": 4096,
    "gpt-4": 8192,
    "claude-3": 20000,
    "claude-3.5-haiku": 25000,
    "google-gemini": 8192,
    "deepseek": 8192,
}
RATE_LIMITS = {
    "claude-3": (
        5,
        10000,
        2000,
    ),  # (requests per minute, input token limit per minute, output tokens per minute)
    "claude-3.5-haiku": (5, 25000, 5000),
    "gpt-4": (60, 8192),
    "google-gemini": (15, 8192),
}


##############################################
# New: Google Gemini Client Implementation
# ##############################################


class GoogleGeminiClient:
    """
    A client for the Google Gemini API based on the quickstart guide.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        # Construct the URL with the Gemini model and your API key as a query parameter.
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}

    def completions_create(self, prompt: str):
        """
        Make a POST request to the Gemini API with the given prompt.

        Args:
            prompt (str): The input text prompt.

        Returns:
            dict: The JSON response from the API.
        """
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        # Make the API request.
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Google Gemini API error: {response.status_code} - {response.text}")
        return response.json()


# =============================================================================
# API-based Loader – for remote models such as GPT, Claude, Gemini, DeepSeek.
# =============================================================================


class APIModuleLoader:
    """
    Unified interface for handling API-based language models (GPT, Anthropic Claude, Google Gemini, DeepSeek).
    """

    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.rate_limits = RATE_LIMITS.get(model_name, None)
        self.token_limit = env_manager.get_int("TOKEN_LIMIT", 4096)
        # self.token_limit = TOKEN_LIMITS.get(model_name, None)
        self.request_counter = 0
        self.last_request_time = time.time()

        # Initialize the appropriate API client
        if "claude" in model_name.lower():
            from anthropic import Client as ClaudeClient

            # self.api_client = ClaudeClient().api_key
            self.api_client = ClaudeClient(api_key=api_key)
        elif "gpt" in model_name.lower():
            import openai

            openai.api_key = api_key
            self.api_client = openai
        elif "google-gemini" in model_name.lower():
            # Replace with your actual Gemini client initialization.
            self.api_client = self.initialize_google_gemini(api_key)
        elif "deepseek" in model_name.lower():
            # Replace with your actual DeepSeek client
            from deepseek import DeepSeekAPI

            self.api_client = DeepSeekAPI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported API model: {model_name}")

    def call_api(self, prompt: str) -> str:
        """
        Call the API for the given prompt.
        """
        logger.info(f"Calling API for model: {self.model_name}")
        logger.debug(f"API token limit set to: {self.token_limit}")
        token_count = calculate_token_count(
            prompt, tokenizer=AutoTokenizer.from_pretrained("google/t5-v1_1-base", legacy=False)
        )
        logger.debug(f"Prompt length: {token_count}")
        logger.info(f"Current prompt: {prompt}")
        logger.debug(f"Words count: {len(prompt.split())}")

        if token_count > self.token_limit:
            raise ValueError(f"Prompt exceeds token limit for {self.model_name} - Try batching data")

        if self.is_rate_limited():
            logger.warning("Rate limit reached. Waiting to retry...")
            self.wait_for_rate_limit()

        model_lower = self.model_name.lower()
        if "claude" in model_lower:
            return self.call_claude(prompt)
        elif "gpt" in model_lower:
            return self.call_gpt(prompt)
        elif "google-gemini" in model_lower:
            return self.call_gemini_api(prompt)
        elif "deepseek" in model_lower:
            return self.call_deepseek_api(prompt)
        else:
            raise ValueError(f"Unsupported API model: {self.model_name}")

    def call_gpt(self, prompt: str) -> str:
        """
        Call OpenAI's ChatCompletion API using the new interface.
        """
        import openai

        openai.api_key = self.api_key
        client = openai.OpenAI(api_key=self.api_key)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(prompt)},
        ]
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
            )
            # Return the generated content (strip extra whitespace)
            return response.choices[0].message["content"].strip()

            response = self.api_client.ChatCompletion.create(
                model=self.model_name,  # e.g. "gpt-3.5-turbo" or "gpt-4"
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )
            return response["choices"][0]["message"]["content"]
        except self.api_client._exceptions.RateLimitError:
            logger.error("Rate limit error from GPT API.")
            raise
        except Exception as e:
            logger.error(f"Error calling GPT API: {e}")
            raise

    def call_claude(self, prompt: str) -> str:
        """
        Call the Anthropic Claude API.
        """

        try:
            headers = {"x-api-key": self.api_key}
            print(headers)  # TODO: add real Claude API handling
            response = self.api_client.completions.create(
                model=self.model_name, prompt=prompt, max_tokens_to_sample=self.token_limit
            )
            return response["completion"]
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            raise

    def call_gemini_api(self, prompt: str) -> str:
        """Call the Google Gemini API using the updated client interface."""
        try:
            logger.debug("Sending request to Gemini API with prompt:")
            logger.debug(prompt)
            # Send the request via our API client; note that no extra model parameter is passed
            response = self.api_client.completions_create(prompt=prompt)
            logger.debug("Received response from Gemini API:")
            logger.debug(response)

            # Gemini returns a 'candidates' list, so check that:
            candidates = response.get("candidates", [])
            if not candidates:
                logger.warning("No candidates found in the Gemini response.")
                return ""
            # Check if the first candidate's content has a 'parts' list with a nonempty 'text'
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                text = parts[0]["text"]
                if not text.strip():
                    logger.warning("Candidate returned empty text.")
                return text
            else:
                logger.warning("Response structure unexpected; cannot extract text.")
                return ""
        except Exception as e:
            logger.error(f"Error calling Google Gemini API: {e}")
            raise

    # Debug
    # def call_gemini_api(self, prompt: str) -> str:
    #     """Call the Google Gemini API using the updated client."""
    #     try:
    #         logger.debug("Sending request to Gemini API with prompt:")
    #         logger.debug(prompt)
    #         response = self.api_client.completions_create(prompt=prompt)
    #         logger.debug("Received response from Gemini API:")
    #         logger.debug(response)
    #         # According to the Gemini quickstart, the response may have a structure like:
    #         # { "choices": [ { "text": "generated text" } ] }
    #         choices = response.get("choices", [])
    #         if not choices:
    #             logger.warning("No choices found in the response.")
    #             return ""
    #         text = choices[0].get("text", "")
    #         if not text:
    #             logger.warning("Empty text returned in the response.")
    #         return text
    #     except Exception as e:
    #         logger.error(f"Error calling Google Gemini API: {e}")
    #         raise

    def call_deepseek_api(self, prompt: str) -> str:
        """
        Call the DeepSeek API.
        """
        try:
            response = self.api_client.generate(prompt=prompt, max_tokens=self.token_limit)
            return response["text"]
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            raise

    def is_rate_limited(self) -> bool:
        """
        Check if the current rate limits are exceeded.
        """
        if not self.rate_limits:
            return False
        now = time.time()
        elapsed_time = now - self.last_request_time
        # Check requests per minute limit:
        if elapsed_time < 60 / self.rate_limits[0]:
            return True
        if self.request_counter >= self.rate_limits[0]:
            return True
        self.request_counter += 1
        self.last_request_time = now
        return False

    def wait_for_rate_limit(self):
        now = time.time()
        elapsed_time = now - self.last_request_time
        wait_time = max(0, (60 / self.rate_limits[0]) - elapsed_time)
        time.sleep(wait_time)
        self.request_counter = 0

    def initialize_google_gemini(self, api_key: str):
        """
        Initialize and return a Google Gemini API client.
        Replace the endpoint and any request parameters as per actual documentation.
        """
        try:
            client = GoogleGeminiClient(api_key)
            logger.info("Initialized Google Gemini API client.")
            return client
        except Exception as e:
            logger.error(f"Error initializing Google Gemini API client: {e}")
            raise


# =============================================================================
# Local Loader – for loading models locally via Unsloth or Huggingface.
# =============================================================================


class LLMLocalLoader:
    """
    Handles loading of local language models. It can use the Unsloth loader (if installed)
    or fall back to Huggingface’s Transformers.
    """

    def __init__(self, token: str = None):
        self.token = token
        self.unsloth_installed = self.check_unsloth_installation()

    def check_unsloth_installation(self) -> bool:
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.from_pretrained()
            return True
        except ImportError:
            logger.warning("Unsloth is not installed. Falling back to Huggingface loading.")
            return False

    def load_local_model(self, model_name: str, loader_type: str = "huggingface"):
        """
        Load a local model using either Unsloth or Huggingface.
        """
        if loader_type == "unsloth" and self.unsloth_installed:
            return self.unsloth_model_loader(model_name)
        elif loader_type == "huggingface":
            return self.huggingface_model_loader(model_name)
        elif loader_type == "stable_diffusion":
            return self.stable_diffusion_model_loader(model_name)

    def unsloth_model_loader(
        self, model_name: str, max_seq_length: int = 2048, dtype=torch.float16, load_in_4bit: bool = True
    ):
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name, max_seq_length=max_seq_length, dtype=dtype, load_in_4bit=load_in_4bit
        )
        logger.debug(f"Loaded Unsloth model: {model_name}")
        # Optionally, put the model into inference mode:
        model = FastLanguageModel.for_inference(model)
        return model, tokenizer

    def huggingface_model_loader(self, model_name: str):
        model_path = os.getenv(f"{model_name.upper()}_MODEL_PATH")
        token = os.getenv("AUTH_TOKEN_HUGGINFACE")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, token=token, device_map="auto", torch_dtype=torch.float16
        )
        logger.debug(f"Loaded Huggingface model: {model_name}")
        return model, tokenizer

    def stable_diffusion_model_loader(self, model_name: str):
        """
        Load a Stable Diffusion model locally for image generation.

        Args:
            model_name (str): The name of the model to load (e.g., 'stable-diffusion-v1-5').

        Returns:
            pipeline: The loaded Stable Diffusion pipeline.
        """
        # model_path = os.getenv(f"{model_name.upper()}_MODEL_PATH")
        model_path = os.getenv("STABLE_DIFFUSION_MODEL_PATH")
        logger.debug(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Stable Diffusion model not found at {model_path}")

        # Load the Stable Diffusion model with half-precision
        pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipeline.to("cuda")  # Ensure the model is loaded onto the GPU
        logger.info(f"Loaded Stable Diffusion model from {model_path}")
        return pipeline


##############################################
# Function to calculate token count and split long prompts
##############################################
def count_tokens(prompt: str, tokenizer) -> int:
    """Return the number of tokens in a prompt using the given tokenizer."""
    tokens = tokenizer.encode(prompt)
    return len(tokens)


def split_prompt_into_batches(prompt: str, tokenizer, token_limit: int) -> list:
    """
    Splits a prompt into batches so that each batch's token count is within token_limit.
    Uses simple sentence splitting; you may enhance this algorithm if needed.
    """
    # First, split the prompt by period followed by a space
    sentences = prompt.split(". ")
    batches = []
    current_batch = ""
    for sentence in sentences:
        if current_batch:
            test_batch = current_batch + ". " + sentence
        else:
            test_batch = sentence
        if count_tokens(test_batch, tokenizer) <= token_limit:
            current_batch = test_batch
        else:
            if current_batch:
                batches.append(current_batch.strip())
            current_batch = sentence
    if current_batch:
        batches.append(current_batch.strip())
    return batches


# =============================================================================
# Unified functions for loading models.
# =============================================================================


def load_model_by_key(
    model_key: str, load_type: str = "api", loader_type: str = "huggingface", token: str = None
):
    """
    Load the appropriate model by key. For API-based models, use APIModuleLoader;
    for local models, use LLMLocalLoader.
    """
    if load_type == "api":
        return APIModuleLoader(model_key, api_key=token), None
    elif load_type == "local":
        loader = LLMLocalLoader(token=token)
        return loader.load_local_model(model_key, loader_type=loader_type)
    else:
        raise ValueError(f"Invalid load_type: {load_type}. Choose 'api' or 'local'.")


def load_all_models():
    """
    Return a dictionary mapping model keys to loader functions.
    """
    return {
        "gpt-3.5": lambda: APIModuleLoader("gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY")),
        "gpt-4": lambda: APIModuleLoader("gpt-4", api_key=os.getenv("OPENAI_API_KEY")),
        "google-gemini": lambda: APIModuleLoader("google-gemini", api_key=os.getenv("GOOGLE_API_KEY")),
        "claude-3": lambda: APIModuleLoader("claude-3", api_key=os.getenv("CLOUD_API_KEY")),
        "deepseek": lambda: APIModuleLoader("deepseek", api_key=os.getenv("DEEPSEEK_API_KEY")),
        "deepseek-un": lambda: LLMLocalLoader().unsloth_model_loader("deepseek"),
        "deepseek-hf": lambda: LLMLocalLoader().huggingface_model_loader("deepseek"),
        "llama-un": lambda: LLMLocalLoader().unsloth_model_loader("llama"),
        "llama-hf": lambda: LLMLocalLoader().huggingface_model_loader("llama"),
        "falcon-un": lambda: LLMLocalLoader().unsloth_model_loader("falcon"),
        "falcon-hf": lambda: LLMLocalLoader().huggingface_model_loader("falcon"),
        "mistral-un": lambda: LLMLocalLoader().unsloth_model_loader("mistral"),
        "mistral-hf": lambda: LLMLocalLoader().huggingface_model_loader("mistral"),
        "stable_diffusion": lambda: LLMLocalLoader().stable_diffusion_model_loader("stable_diffusion"),
    }


if __name__ == "__main__":
    # python .\speech_analyzer\loader.py
    # For testing, load one model and call its API.
    model_key = "google-gemini"  # Try changing to another key if desired.
    token = os.getenv("GOOGLE_API_KEY")
    # model_key = "claude-3"  # Try changing to another key if desired.
    # token = os.getenv("CLOUD_API_KEY")
    # model_key = "gpt-3.5-turbo"  # Try changing to another key if desired.
    # token=os.getenv("OPENAI_API_KEY")
    loader = APIModuleLoader(model_key, api_key=token)
    # loader = load_model_by_key(model_key, load_type="api", token=token)
    prompt = "What is the capital of France?"
    prompt = "Explain how AI works."
    try:
        generated_text = loader.call_api(prompt)
        logger.info(f"Generated text: {generated_text}")
    except Exception as e:
        logger.error(f"Error during API call: {e}")
