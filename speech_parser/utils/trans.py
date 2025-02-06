import asyncio
from pathlib import Path
from googletrans import Translator
from loguru import logger

from speech_parser.audio_processing.save_results import save_summary
from speech_parser.utils.config import (
    AI_MODEL_NAME,
    AUDIO_FILE_NAME,
    OUTPUT_DIR,
)

import httpx
import urllib.parse


async def translate_summary_to_russian_urllib(summary):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "en",  # Source language: English
        "tl": "ru",  # Target language: Russian
        "hl": "ru",  # Language for the result interface
        "dt": "t",  # Return translation
        "ie": "UTF-8",  # Input encoding
        "oe": "UTF-8",  # Output encoding
    }

    translated_summary = ""
    max_length = 1000  # Limit the query size

    try:
        # Split summary if it exceeds max length
        summary_chunks = [summary[i : i + max_length] for i in range(0, len(summary), max_length)]

        async with httpx.AsyncClient() as client:
            for chunk in summary_chunks:
                params["q"] = urllib.parse.quote(chunk)
                response = await client.get(url, params=params)

                if response.status_code == 200:
                    translated_data = response.json()
                    translated_text = "".join([item[0] for item in translated_data[0]])
                    # translated_summary += translated_text
                    translated_summary += urllib.parse.unquote(translated_text)
                else:
                    raise ValueError(f"Error during translation: {response.status_code} {response.text}")

        # Ensure that translated summary is not empty
        if translated_summary.strip():
            return translated_summary
        else:
            logger.error("Translation returned an empty result.")
            return None

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None


async def translate_summary_to_russian(summary):
    translator = Translator()
    translated = await translator.translate(summary, src="en", dest="ru")
    return translated.text


async def translate(summary="Your summary text"):
    # python -m speech_parser.utils.trans

    # summary = "Your summary text"
    # Translate the summary to Russian
    russian_summary = await translate_summary_to_russian_urllib(summary)
    # russian_summary = await translate_summary_to_russian(summary)

    # Save the Russian translation
    save_summary(
        russian_summary, Path(AUDIO_FILE_NAME).stem, OUTPUT_DIR, ai_model_key=AI_MODEL_NAME, translated=True
    )


if __name__ == "__main__":
    asyncio.run(translate())
