import sys
import os
import wave
import time
import vosk
import json
import csv

# ffmpeg -i D:\A-Samples\recorder-H1N1\2024-witcher-miro\ZOOM0065.WAV -ac 1 -ar 16000 -sample_fmt s16 D:\A-Samples\recorder-H1N1\2024-witcher-miro\ZOOM0065_converted.WAV

# Initialize the CSV file for saving progress
csv_file = "time_output_ffmpeg5.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "time", "last_partial"])


def save_progress(id, elapsed_time, last_partial):
    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([id, elapsed_time, last_partial])


def print_progress(current_time, total_time, partial_text, start_time, id_counter):
    percent_complete = (current_time / total_time) * 100
    elapsed_time = time.time() - start_time
    estimated_total_time = (elapsed_time / (current_time / total_time)) if current_time > 0 else 0
    remaining_time = estimated_total_time - elapsed_time

    # Convert times to minutes/hours format
    elapsed_minutes = elapsed_time // 60
    estimated_hours = estimated_total_time // 3600
    estimated_minutes = (estimated_total_time % 3600) // 60

    remaining_hours = remaining_time // 3600
    remaining_minutes = (remaining_time % 3600) // 60

    print(
        f"Progress: {percent_complete:.2f}% | Elapsed: {int(elapsed_minutes)}min | Last partial: '{partial_text}' "
        f"| Estimated remaining: {int(remaining_hours)}h {int(remaining_minutes)}min"
    )

    # Save progress to CSV file
    save_progress(id_counter, int(elapsed_minutes), partial_text)


def transcribe_audio(file_path):
    wf = wave.open(file_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        return

    model = vosk.Model("vosk-model-ru-0.42")
    rec = vosk.KaldiRecognizer(model, wf.getframerate())

    total_duration = wf.getnframes() / wf.getframerate()  # Total audio length in seconds
    processed_duration = 0  # Track the time processed
    start_time = time.time()  # To estimate the remaining time
    id_counter = 0  # Counter for CSV entry ID

    while True:
        data = wf.readframes(4000)  # Process audio in chunks
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            processed_duration += len(data) / wf.getframerate()  # Update the processed duration
            partial_text = result.get("text", "")
            print_progress(processed_duration, total_duration, partial_text, start_time, id_counter)
            id_counter += 1
        else:
            partial_result = json.loads(rec.PartialResult())
            partial_text = partial_result.get("partial", "")
            print_progress(processed_duration, total_duration, partial_text, start_time, id_counter)

    # Final result
    result = json.loads(rec.FinalResult())
    final_text = result.get("text", "")
    print(f"Final transcription: {final_text}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_ffmpeg5.py <audiofile>")
        exit(1)

    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Audio file '{audio_file}' not found!")
        exit(1)

    transcribe_audio(audio_file)
