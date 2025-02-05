import os

# from transformers import AutoTokenizer
# from .loader import TOKEN_LIMITS

# ---- Helper function to split prompt into batches based on token limits ----


def calculate_token_count(text: str, tokenizer) -> int:
    """Calculate the number of tokens in text using the tokenizer."""
    tokens = tokenizer.encode(text)
    return len(tokens)


def split_prompt_into_batches(
    prompt: str, tokenizer, token_limit: int = int(os.getenv("TOKEN_LIMIT", "4096"))
) -> list:
    """
    Splits a long prompt (dialogue text) into smaller batches that do not exceed the token limit.
    The prompt is split by line breaks (assuming each line is a segment).
    """
    segments = prompt.splitlines()
    batches = []
    current_batch = ""
    current_token_count = 0

    for segment in segments:
        tokens_in_segment = calculate_token_count(segment, tokenizer)
        if current_token_count + tokens_in_segment > token_limit:
            batches.append(current_batch.strip())
            current_batch = segment
            current_token_count = tokens_in_segment
        else:
            if current_batch:
                current_batch += " " + segment
            else:
                current_batch = segment
            current_token_count += tokens_in_segment

    if current_batch:
        batches.append(current_batch.strip())

    return batches


# Example usage:
if __name__ == "__main__":
    prompt = "Your very long transcription text goes here..."
    batches = split_prompt_into_batches(prompt, model_name="gpt-3.5-turbo", token_limit=4000)
    for i, batch in enumerate(batches, start=1):
        print(f"Batch {i}:\n{batch}\n")


# def split_prompt_into_batches(prompt: str, model_name: str, token_limit: int = None):
#     """
#     Splits the prompt into smaller batches so that each batch's token count is below the token_limit.
#     Uses the GPT2 tokenizer by default.

#     Args:
#         prompt (str): The full prompt text.
#         model_name (str): The model identifier (used to load the appropriate tokenizer).
#         token_limit (int): The maximum allowed token count per batch. If None, it uses a default limit.

#     Returns:
#         list: A list of prompt chunks.
#     """
#     # For simplicity, we use GPT2 tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     if token_limit is None:
#         # Use default limits based on our TOKEN_LIMITS dictionary
#         token_limit = TOKEN_LIMITS.get(model_name, 4096) - 100  # Reserve some margin

#     # Tokenize the full prompt
#     tokens = tokenizer.encode(prompt)
#     if len(tokens) <= token_limit:
#         return [prompt]

#     # Split tokens into chunks and decode back into strings
#     batches = []
#     for i in range(0, len(tokens), token_limit):
#         batch_tokens = tokens[i : i + token_limit]
#         batch_text = tokenizer.decode(batch_tokens, skip_special_tokens=True)
#         batches.append(batch_text)

#     return batches
