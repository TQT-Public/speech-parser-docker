# Audio Processing and Transcription with Vosk

## Overview

This project processes long WAV files, transcribes them using the Vosk model, and outputs filtered dialogues with speaker identification.

This project provides tools for speaker diarization, audio transcription, and filtering using Vosk and Pyannote audio models.

## Features

- Diarization of speakers in an audio file
- Transcription of speaker dialogues
- Filtering of short audio segments
- Saving filtered dialogue and summary

## Project Structure

- **/audio_processing**: Handles audio conversion, transcription, and speaker identification.
- **/models**: Handles downloading Vosk models.
- **/utils**: Contains helper functions.
- **main.py**: The main script to run the transcription process.

## How to Run

1. Install dependencies from `requirements.txt` or use `poetry install`.
2. Place your audio files in the `sources` directory.
3. Adjust .env
4. Run the main script:

```bash
cp .env.default .env
python main.py
```

## Project Setup for Docker

### Build docker container and run Fetcher

```bash
cd speech-parser-docker
cp ./configs/.env.default .env
docker build --pull --rm -f "Dockerfile" -t speech-parser-gpu:latest "."
docker compose up -d
```

```bash
docker build --pull --rm -f "Dockerfile" -t speech-parser-gpu:latest "."
```

```bash
docker compose up -d
```

### One-liner To attach to virtual console, then run python3 main.py

```bash
cp ./configs/.env.default .env && docker build --pull --rm -f "Dockerfile" -t speech-parser-gpu:latest "."
```

### Interactive mode

```bash
# docker exec -it speech-parser-gpu "watch -n 0.2 nvidia-smi"
docker compose run speech-parser-gpu:latest /bin/bash -c "nano .env && python3 main.py"
```

### Monitor Docker CUDA resources

```bash
docker exec -it speech-parser-gpu "watch -n 0.2 nvidia-smi"
```

#### Or attach to container exec and RUN `watch -n 0.2 nvidia-smi`

```bash
watch -n 0.2 nvidia-smi
```

### Clear after Docker (CUDA+torch WSL image is about 60 Gb on disk)

```bash
.\scripts\docker-prune.bat
```

#### Then, close Docker engine to release docker.VHDX memory

#### Rename `docker-prune.bat` to `docker-prune.sh` on UNIX

### Speech Parser with Vosk and Pyannote

#### This project performs speaker diarization and transcription using Vosk and Pyannote. It processes audio files, identifies speakers, and saves transcriptions

### Prerequisites

1. **Python 3.8+**: Ensure Python is installed.
2. **Vosk Model**: Download the required Vosk model.
3. **API Key**: Obtain an API key from Pyannote if using their services.

### Setup

1. **Clone the repository**

    ```bash
    git clone https://github.com/your-username/speech-parser.git
    cd speech-parser
    ```

2. **Install Dependencies** Make sure you have `conda` or `virtualenv` installed.

    ```bash
    conda create --name speech-parser python=3.11
    conda activate speech-parser
    pip install -r requirements.txt
    ```

    **Python3.9** and **Python3.11** supported

3. **Setup Environment Variables**
    - Copy the `.env.sample` file and rename it to `.env`.
    - Create a .env file in the project root. You can use the .env.sample as a reference
    - Edit the `.env` file with your local paths and settings.

    ```bash
    cp .env.sample .env
    nano .env
    ```

    ```bash
    WORKSPACE=D:/A-Samples/recorder-H1N1/2024-witcher-miro
    VOSK_MODEL_PATH=./models
    MODEL_NAME=vosk-model-ru-0.42
    AUDIO_FILE=ZOOM0067.wav
    OUTPUT_DIR=./output
    OUTPUT_DIR_PARTS=./audio_files/parts
    AUDIO_WORKSPACE=./audio_files
    ASSIGNSPEAKERS=False
    FILTER_UNNECESSARY_RTTM=True
    MIN_RTTM_DURATION=2.0
    ```

4. **Download the Vosk Model** Run the following command to download the Vosk model:

    ```bash
    python -m models.download_model
    ```

5. **Get API Key (if required)** For Pyannote API, you need an API key:
    - Register at [Pyannote](https://pyannote.github.io).
    - Add the key to `.env` under `PYANNOTE_API_KEY`.

### Features [m]

- Speaker Diarization: Identify speakers in audio files.
- Transcription: Transcribe audio segments for each speaker.
- Filtered Transcription: Filters out short audio segments based on the threshold set in .env.
- Assign Real Speaker Names: Option to assign real names to identified speakers.

## Environment Variables

### The following variables are configured in the .env file

```bash
WORKSPACE: Path to your working directory.
VOSK_MODEL_PATH: Path to the Vosk model.
MODEL_NAME: Name of the Vosk model.
AUDIO_FILE: Name of the audio file to process.
OUTPUT_DIR: Directory for output files.
OUTPUT_DIR_PARTS: Directory for split audio parts.
AUDIO_WORKSPACE: Path to your audio workspace.
ASSIGNSPEAKERS: Whether to assign real speaker names (True or False).
FILTER_UNNECESSARY_RTTM: Whether to filter out short RTTM segments (True or False).
MIN_RTTM_DURATION: Minimum duration in seconds for keeping segments.
```

### Troubleshooting

#### If you encounter issues such as missing files, ensure the directories are properly configured in your .env file

#### 1. **Optimize Docker Resource Usage (WSL2 Configuration)**

- Modify the `.wslconfig` file to allocate more memory and CPU to Docker. This should help prevent memory exhaustion when the larger models are being loaded.

Example `.wslconfig` configuration:

```ini
[wsl2]
memory=8GB  # Allocate 8 GB of memory
processors=4  # Use 4 processors (adjust as per your system)
swap=4GB  # Set swap size to 4 GB
```

- Save (`touch .wslconfig` if no any) this file in your Windows user profile directory (`C:\Users\<YourUser>\.wslconfig`) and restart WSL with:

```bash
wsl --shutdown
```

Then, **restart Docker**.

Also look at set config at  `docker-compose.yaml` configuration, end of the file
