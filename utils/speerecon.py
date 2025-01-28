import os
import sys
import json
import vosk
import wave
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake  # For key point extraction (RAKE)

# Ensure nltk data is available
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")


def extract_keywords_rake(text):
    """Extract key points from text using RAKE."""
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()


def process_audio_file(file_path, model_path):
    """Process the audio file and generate transcriptions."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Load Vosk model
    print(f"Loading Vosk model from: {model_path}")
    model = vosk.Model(model_path)

    # Open WAV file
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() > 48000:
        print("Audio file must be WAV format mono PCM.")
        return

    # Adjust decoder settings for better accuracy
    rec = vosk.KaldiRecognizer(
        model,
        wf.getframerate(),
        json.dumps(
            {
                "beam": 15,  # Wider beam width for more accurate results
                "max-active": 10000,  # Increase active states
                "lattice-beam": 8,  # Wider search space
                "silence-phones": "1:2:3:4:5:6:7:8:9:10",
            }
        ),
    )

    # Process audio
    transcription = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcription.append(result["text"])

    # Capture final partial results
    final_result = json.loads(rec.FinalResult())
    transcription.append(final_result.get("text", ""))

    wf.close()
    full_text = " ".join(transcription)
    return full_text


def save_transcriptions(transcription, output_folder):
    """Save transcriptions to output files."""
    os.makedirs(output_folder, exist_ok=True)

    # Filtered dialogue output (basic transcription)
    filtered_dialogue_path = os.path.join(output_folder, "filtered_dialogue.txt")
    with open(filtered_dialogue_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    # Key point extraction for summary
    key_points = extract_keywords_rake(transcription)
    discussion_summary_path = os.path.join(output_folder, "discussion_summary.txt")
    with open(discussion_summary_path, "w", encoding="utf-8") as f:
        f.write("Key Points:\n")
        f.write("\n".join(key_points))

    print(f"Filtered dialogue saved to: {filtered_dialogue_path}")
    print(f"Discussion summary saved to: {discussion_summary_path}")


def main():
    """Main execution function."""
    if len(sys.argv) < 2:
        print("Usage: python speerecon.py <audio_file_path>")
        return

    audio_file = sys.argv[1]
    model_path = "vosk-model-ru-0.42"  # Adjust path to your model directory
    output_folder = "./output"

    print(f"Processing file: {audio_file}")

    # Generate transcription
    transcription = process_audio_file(audio_file, model_path)
    if transcription:
        print(f"Transcription completed. Saving results...")
        save_transcriptions(transcription, output_folder)
    else:
        print("Transcription failed.")


if __name__ == "__main__":
    main()
