import requests
from loguru import logger

DEEPSEEK_API_KEY = "your_deepseek_api_key"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/completions"


def call_backup_llm_api(prompt):
    # Placeholder for alternative API like DeepSeek
    response = call_deepseek_api(prompt)
    return response.get("summary", "")


def call_deepseek_api(prompt: str, model: str = "deepseek-base"):
    """
    Calls the DeepSeek API to get a completion (summary or other LLM response) for a given prompt.

    Args:
        prompt (str): The input text to send to the API.
        model (str): The DeepSeek model to use (e.g., "deepseek-base").

    Returns:
        str: The generated response from DeepSeek or error message if the call fails.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1500,  # You can adjust this based on the API limit
        "temperature": 0.7,
        "top_p": 0.9,
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Assuming the API returns 'choices' similar to OpenAI
        return result.get("choices", [{}])[0].get("text", "")
    except requests.RequestException as e:
        logger.error(f"Error calling DeepSeek API: {e}")
        return f"Error: {e}"


# def generate_summary(prompt: str, model_or_key: str):
#     """
#     Generate a summary using the specified AI model or key.
#     """
#     if "gpt" in model_or_key.lower():
#         return call_gpt_api(prompt, model_or_key)
#     elif "deepseek" in model_or_key.lower():
#         return call_deepseek_api(prompt, model_or_key)
#     else:
#         # Default fallback or error handling for unknown models
#         return "Model not supported."
