import os
import time
from loguru import logger
import openai  # For API calls

# from openai.error import (
#     RateLimitError,
#     APIError,
#     InvalidRequestError,
#     APIConnectionError,
#     ServiceUnavailableError,
#     OpenAIError,
#     NotFoundError,
# )
from dotenv import load_dotenv

# import torch
# import openai
# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv()

RateLimitError = openai._exceptions.RateLimitError
APIError = openai._exceptions.APIError
InvalidRequestError = openai._exceptions.BadRequestError
APIConnectionError = openai._exceptions.APIConnectionError
PermissionDeniedError = openai._exceptions.PermissionDeniedError
OpenAIError = openai._exceptions.OpenAIError
NotFoundError = openai._exceptions.NotFoundError
AuthenticationError = openai._exceptions.AuthenticationError
# and so on...

# Example API key for OpenAI GPT
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

logger.debug("OpenAI version:", openai.__version__)
logger.info("OpenAI module file:", openai.__file__)


def call_gpt_api(prompt: str, model_version: str) -> str:
    """
    Call the OpenAI API using the new client interface and return the generated text.
    Includes retries and fallback handling.

    Args:
        prompt (str): The prompt text.
        model_version (str): The model version to use (e.g., "gpt-4" or "gpt-3.5-turbo").

    Returns:
        str: The generated text from the model.
    """
    openai.api_key = OPENAI_API_KEY
    # The new API now uses a client-based interface; here we simply use openai.api_key.
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": prompt},
    # ]
    # Create an OpenAI client instance
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Prepare the messages list as required by the new API
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    max_retries = 5
    retries = 0

    while True:
        try:
            # old API
            # response = openai.ChatCompletion.create(
            #     model=model_version,
            #     messages=messages,
            #     max_tokens=1500,
            #     temperature=0.7,
            # )
            # return response.choices[0].message["content"].strip()
            # Call the new Chat API using the client
            response = client.chat.completions.create(
                model=model_version,
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
            )

            # Return the generated content (strip extra whitespace)
            return response.choices[0].message["content"].strip()

        except RateLimitError as e:
            retries += 1
            wait_time = 2**retries
            logger.warning(
                f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {retries}/{max_retries})"
            )
            if retries > max_retries:
                logger.error("Max retry attempts reached. Raising RateLimitError.")
                raise e
            time.sleep(wait_time)

        except NotFoundError as e:
            logger.error(f"Model {model_version} not found: {e}")
            # Fallback to gpt-3.5-turbo if the requested model isn’t available.
            if model_version.lower() != "gpt-3.5-turbo":
                logger.info("Falling back to gpt-3.5-turbo.")
                model_version = "gpt-3.5-turbo"
            else:
                raise e

        # except (APIError, InvalidRequestError, APIConnectionError, PermissionDeniedError, OpenAIError) as e:
        #     logger.error(f"OpenAI API error: {e}")
        #     raise e
        # except RateLimitError as e:
        #     print("Rate limit exceeded:", e)
        except APIError as e:
            print("OpenAI API error:", e)
        except InvalidRequestError as e:
            print("Invalid request:", e)
        except APIConnectionError as e:
            print("Connection error:", e)
        except AuthenticationError as e:
            print("AuthenticationError error:", e)
        except PermissionDeniedError as e:
            print("PermissionDeniedError:", e)
        # except NotFoundError as e:
        #     print("Model not found:", e)
        except OpenAIError as e:
            print("OpenAI error:", e)
        return ""


def call_gpt_old_api(prompt: str, model_version: str) -> str:
    """
    Call the OpenAI ChatCompletion API using the new 1.0.0 interface.

    Args:
        prompt (str): The prompt text.
        model_version (str): The OpenAI model version to use (e.g. "gpt-4").

    Returns:
        str: The generated text response.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        model=model_version,
        messages=messages,
        max_tokens=1500,
        temperature=0.7,
    )
    # Access the generated message content
    return response.choices[0].message["content"].strip()


def load_gpt_model(model_key: str):
    """
    Generator function to load a GPT model.
    If the key contains 'gpt', use OpenAI API; otherwise, use a local Hugging Face loader.
    """
    if "gpt" in model_key.lower():
        # For OpenAI API calls, we just return the model key and a GPT2 tokenizer as an example.
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # tokenizer = AutoTokenizer.from_pretrained(model_key)
        return model_key, tokenizer
        return model_key, None
    else:
        # Here you can add your custom loader (e.g. unsloth_loader or huggingface_loader)
        # from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_key)
        tokenizer = AutoTokenizer.from_pretrained(model_key)
        return model, tokenizer


def load_or_call_gpt_model(model_version) -> tuple[AutoModelForCausalLM | None, AutoTokenizer | None]:
    """
    Load the GPT model, either locally or via API depending on environment settings.

    Args:
        model_version (str): The model version (e.g., gpt-3.5, gpt-4, or local path).

    Returns:
        tuple: Model and tokenizer if local, or None for API calls.
    """
    use_local_model = os.getenv("USE_LOCAL_MODEL", "False").lower() == "true"

    if use_local_model:
        logger.debug(f"Loading local model: {model_version}")
        model = AutoModelForCausalLM.from_pretrained(model_version)
        tokenizer = AutoTokenizer.from_pretrained(model_version)
        return model, tokenizer
    else:
        # Return None if using API instead of local
        logger.debug(f"Using API version: {model_version}")
        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
        return None, None


def call_api_or_use_local(prompt, model, tokenizer):
    """
    Generate a summary using a local model or API, based on configuration.

    Args:
        prompt (str): The input prompt to summarize.
        model: The language model (can be None for API calls).
        tokenizer: The tokenizer (can be None for API calls).

    Returns:
        str: The generated summary.
    """
    use_local_model = os.getenv("USE_LOCAL_MODEL", "False").lower() == "true"

    if use_local_model:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate summary using the model
        outputs = model.generate(**inputs, max_length=1500)

        # Decode the generated tokens into text
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    else:
        # Call GPT API
        model_version = os.getenv("GPT_API_VERSION_1", "gpt-3.5")
        return call_gpt_api(prompt, model_version)


def run_with_multiple_gpt_versions(prompt):
    """
    Run a prompt with multiple GPT versions and return the results.

    Args:
        prompt (str): The input prompt.

    Returns:
        dict: Dictionary with GPT versions as keys and their responses as values.
    """
    results = {}

    # GPT API Versions
    api_versions = [os.getenv("GPT_API_VERSION_1", "gpt-3.5"), os.getenv("GPT_API_VERSION_2", "gpt-4")]

    for version in api_versions:
        print(f"Calling {version}...")
        model, tokenizer = load_gpt_model(version)
        summary = call_api_or_use_local(prompt, model, tokenizer)
        results[version] = summary

    return results


def generate_summary(prompt: str, model_or_key, tokenizer) -> str:
    """
    Generate a summary of the given prompt using either an OpenAI API model or a local model.

    Args:
        prompt (str): The text prompt to summarize.
        model_or_key: Either a string (model key for OpenAI) or a local model object.
        tokenizer: The tokenizer corresponding to the model.

    Returns:
        str: The generated summary text.
    """
    if isinstance(model_or_key, str) and "gpt" in model_or_key.lower():
        # Call the API version
        return call_gpt_api(prompt, model_or_key)
    else:
        # Use the local model to generate text
        # (Make sure the model is on the correct device)
        device = next(model_or_key.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model_or_key.generate(**inputs, max_length=1500)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


# def generate_summary(prompt: str, model_or_key, tokenizer) -> str:
#     if isinstance(model_or_key, str) and "gpt" in model_or_key.lower():
#         return call_gpt_api(prompt, model_or_key)
#     else:
#         inputs = tokenizer(prompt, return_tensors="pt").to(next(model_or_key.parameters()).device)
#         outputs = model_or_key.generate(**inputs, max_length=1500)
#         return tokenizer.decode(outputs[0], skip_special_tokens=True)


# def generate_summary(prompt: str, model_version: str, tokenizer) -> str:
#     """
#     Generate summary text by calling the GPT API.
#     """
#     return call_gpt_api(prompt, model_version)


if __name__ == "__main__":
    # Example usage of the function # TODO: add df parsing from csv
    test_prompt = "Explain the key differences between supervised and unsupervised learning."

    # Run with multiple GPT versions
    summaries = run_with_multiple_gpt_versions(test_prompt)

    # Print summaries from different models
    for model_version, summary in summaries.items():
        logger.debug(f"\nModel: {model_version}")
        logger.info(f"Summary: {summary}")
