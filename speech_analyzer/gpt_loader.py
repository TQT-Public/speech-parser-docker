import os
from loguru import logger
import openai  # For API calls
from dotenv import load_dotenv

# import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from speech_analyzer.dialogue_analyzer import load_dialogue_data
# from speech_analyzer.csv_loader import load_csv_data_for_model, load_csv_for_neural_network
# from speech_analyzer.dialogue_analyzer import run_with_multiple_gpt_versions

# Load environment variables
load_dotenv()

# Example API key for OpenAI GPT
openai.api_key = os.getenv("OPENAI_API_KEY")

# import openai
# from transformers import AutoTokenizer, AutoModelForCausalLM


# def load_gpt_model(api_version="gpt-3.5"):
#     """
#     Load GPT model either via OpenAI API or local model.

#     Args:
#         api_version (str): The version of the GPT model (e.g., gpt-3.5, gpt-4).

#     Returns:
#         tuple: GPT model and tokenizer (if local).
#     """
#     api_key = os.getenv("OPENAI_API_KEY")
#     if api_key and api_version.startswith("gpt"):
#         openai.api_key = api_key
#         logger.info(f"Using OpenAI API for model {api_version}")
#         return api_version, None
#     else:
#         logger.info(f"Using local GPT model {api_version}")
#         model = AutoModelForCausalLM.from_pretrained(api_version)
#         tokenizer = AutoTokenizer.from_pretrained(api_version)
#         return model, tokenizer


def load_gpt_model(model_version) -> tuple[AutoModelForCausalLM | None, AutoTokenizer | None]:
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


def call_gpt_api(prompt, model_version):
    """
    Call GPT API based on the given model version.

    Args:
        prompt (str): The prompt for GPT.
        model_version (str): The version of GPT (e.g., gpt-3.5, gpt-4).

    Returns:
        str: The generated response from the API.
    """
    response = openai.Completion.create(engine=model_version, prompt=prompt, max_tokens=1500)
    return response.choices[0].text.strip()


def generate_summary(prompt, model, tokenizer):
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
        summary = generate_summary(prompt, model, tokenizer)
        results[version] = summary

    return results


if __name__ == "__main__":
    # Example usage of the function
    test_prompt = "Explain the key differences between supervised and unsupervised learning."

    # Run with multiple GPT versions
    summaries = run_with_multiple_gpt_versions(test_prompt)

    # Print summaries from different models
    for model_version, summary in summaries.items():
        logger.debug(f"\nModel: {model_version}")
        logger.info(f"Summary: {summary}")
