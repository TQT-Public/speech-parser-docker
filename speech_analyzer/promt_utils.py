# import os


# from transformers import AutoTokenizer
# from .loader import TOKEN_LIMITS
# from speech_parser.utils.env import env_as_int


def calculate_token_count(text, tokenizer=None):
    """
    Calculate the number of tokens in a text using a tokenizer if available.
    Otherwise, estimate using a refined approximation.
    """
    if tokenizer:
        return len(tokenizer.encode(text))  # Use actual tokenization
    else:
        return len(text) // 4  # Approximate: 1 token ≈ 4 characters


def split_prompt_into_batches(text, token_limit, tokenizer):
    """
    Splits text into batches where NO batch exceeds token_limit tokens.

    Args:
        text (str): Full input text.
        token_limit (int): Max tokens per batch.
        tokenizer (object): Preloaded tokenizer.

    Returns:
        list: List of token-limited text batches.
    """
    if not tokenizer:
        raise ValueError("Tokenizer is required but not found!")

    tokens = tokenizer.encode(text, add_special_tokens=False)  # Convert text to token IDs
    batches = []
    current_batch = []

    for token in tokens:
        if len(current_batch) + 1 > token_limit:
            batches.append(tokenizer.decode(current_batch))  # Convert back to text
            current_batch = [token]  # Start new batch
        else:
            current_batch.append(token)

    if current_batch:
        batches.append(tokenizer.decode(current_batch))

    return batches


def ensure_batches_fit(batches, token_limit, tokenizer):
    """Recursively ensures all batches fit within token limit."""
    final_batches = []
    for batch in batches:
        token_count = len(tokenizer.encode(batch, add_special_tokens=False))
        if token_count > token_limit:
            # If batch is still too large, split further
            split_batches = split_prompt_into_batches(batch, token_limit, tokenizer)
            final_batches.extend(split_batches)
        else:
            final_batches.append(batch)
    return final_batches


def split_prompt_into_batches_strict(text, token_limit, tokenizer):
    """
    Precisely splits text into batches where NO batch exceeds token_limit tokens.

    Args:
        text (str): The full input text.
        token_limit (int): Maximum tokens allowed per batch.
        tokenizer (object): Local tokenizer for accurate splitting.

    Returns:
        list: List of strictly token-limited text batches.
    """
    if not tokenizer:
        raise ValueError("Tokenizer is required but not found!")

    tokens = tokenizer.encode(text, add_special_tokens=False)  # Convert text to token IDs
    batches = []
    current_batch = []
    current_token_count = 0

    for token in tokens:
        if current_token_count + 1 > token_limit:  # If adding a token exceeds limit
            batches.append(tokenizer.decode(current_batch))  # Convert token IDs back to text
            current_batch = [token]
            current_token_count = 1
        else:
            current_batch.append(token)
            current_token_count += 1

    if current_batch:
        batches.append(tokenizer.decode(current_batch))

    return batches


def split_prompt_into_batches_chars(text, token_limit, tokenizer=None):
    """
    Splits text into smaller batches, ensuring that no batch exceeds the token limit.

    Args:
        text (str): The full text to be split.
        token_limit (int): Maximum token limit per batch.
        tokenizer (object, optional): Tokenizer for accurate token counting.

    Returns:
        list: List of properly sized text batches.
    """
    words = text.split()
    batches = []
    current_batch = []
    current_token_count = 0

    for word in words:
        estimated_tokens = len(word) // 4  # Fallback estimate (1 token ≈ 4 chars)
        if tokenizer:
            estimated_tokens = len(tokenizer.encode(word))

        # If adding this word exceeds the limit, finalize the current batch
        if current_token_count + estimated_tokens > token_limit:
            if current_batch:
                batches.append(" ".join(current_batch))

            # Start a new batch
            current_batch = [word]
            current_token_count = estimated_tokens

            # Edge case: If a single word itself exceeds the limit, split it
            while current_token_count > token_limit:
                # Split the word into chunks that fit within the token limit
                chunk_size = min(len(word) // 2, token_limit)  # Conservative split
                batch_chunk = word[:chunk_size]
                batches.append(batch_chunk)
                word = word[chunk_size:]
                current_token_count = len(word) // 4  # Update count for remaining part

        else:
            current_batch.append(word)
            current_token_count += estimated_tokens

    if current_batch:
        batches.append(" ".join(current_batch))

    return batches


# ---- Helper function to split prompt into batches based on token limits ----
def calculate_token_count_no_tokenizer(text, token_limit):
    """
    Calculate the approximate number of tokens in a text based on character count.
    This assumes that, on average, 1 token is approximately 3 characters.

    Args:
        text (str): The text for which to calculate tokens.
        token_limit (int): The maximum token limit for the AI model.

    Returns:
        int: Approximate number of tokens based on character count.
    """
    char_count = len(text)
    token_count = char_count // 3  # Approximation: 1 token ≈ 3 characters
    return token_count
    word_count = len(text.split())  # Simple approximation: 1 word ~ 1 token
    return word_count  # Approximate number of tokens based on word count


def split_prompt_into_batches_no_tokenizer(text, token_limit):
    """
    Split text into smaller batches based on an estimated token count using character count as an approximation.

    Args:
        text (str): The full text to be split.
        token_limit (int): The maximum token limit for the AI model.

    Returns:
        list: List of text batches.
    """
    # char_limit = (
    #     token_limit * 4
    # )  # Approximate tokens: 1 token ≈ 4 characters (this can vary depending on language/model)
    char_limit = token_limit * 3  # A more conservative approximation: 1 token ≈ 3 characters on average
    words = text.split()
    batches = []
    current_batch = []
    current_char_count = 0

    for word in words:
        word_length = len(word) + 1  # Including the space
        current_char_count += word_length
        # current_char_count += len(word) + 1  # Include space between words

        if current_char_count > char_limit:
            batches.append(" ".join(current_batch))
            current_batch = [word]
            # current_char_count = len(word) + 1  # Reset count for new batch
            current_char_count = word_length  # Reset char count for the new batch
        else:
            current_batch.append(word)

    if current_batch:
        batches.append(" ".join(current_batch))

    return batches


def split_prompt_into_batches_tokenizer(prompt, token_limit, tokenizer):
    """
    Splits the transcriptions (which may not contain punctuation) into manageable batches based on token limits.

    Args:
        prompt (str): The full transcription or dialogue text.
        tokenizer: The tokenizer used for counting tokens and generating batches.
        token_limit (int): The maximum number of tokens allowed for each batch a.k.a. TOKEN_LIMIT.

    Returns:
        List[str]: A list of batched text segments.
    """
    # Split the prompt using a combination of line breaks and spaces as a rough segmentation method
    segments = prompt.splitlines()  # Split by lines first (assuming each line represents a dialogue/segment)

    batches = []
    current_batch = ""
    current_token_count = 0

    for segment in segments:
        # Tokenize the current segment
        tokens_in_segment = tokenizer(segment, return_tensors="pt")["input_ids"].size(1)

        # If adding this segment exceeds the token limit, start a new batch
        if current_token_count + tokens_in_segment > token_limit:
            batches.append(current_batch.strip())
            current_batch = segment  # Start a new batch with the current segment
            current_token_count = tokens_in_segment
        else:
            current_batch += f" {segment}"  # Add the segment to the current batch
            current_token_count += tokens_in_segment

    # Add the last batch if it's not empty
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
