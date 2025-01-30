
# Speech Parser: Audio Processing and Transcription with Vosk and AI Models

## Overview

This project provides tools for processing long audio files, transcribing them using the Vosk model, and performing advanced tasks like speaker diarization, dialogue filtering, and summarization using AI models like Mistral, LLaMA, Falcon, and DeepSeek.

## Features

- **Audio Processing**:
  - Split long audio files into manageable segments.
  - Convert audio formats using FFmpeg.
- **Transcription**:
  - Transcribe audio using the Vosk model.
  - Identify and label speakers using Pyannote.
- **AI Integration**:
  - Summarize dialogues using AI models (Mistral, LLaMA, Falcon, DeepSeek).
  - Filter out short or unnecessary segments.
- **Docker Support**:
  - Run the entire pipeline in a Docker container for easy setup and reproducibility.

---

## Table of Contents

- [Speech Parser: Audio Processing and Transcription with Vosk and AI Models](#speech-parser-audio-processing-and-transcription-with-vosk-and-ai-models)
  - [Overview](#overview)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setting Up the Project](#setting-up-the-project)
    - [Installing FFmpeg](#installing-ffmpeg)
    - [Setting Up CUDA (Optional)](#setting-up-cuda-optional)
  - [Running the Project](#running-the-project)
    - [Using Docker](#using-docker)
    - [Running Locally](#running-locally)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Downloading Models](#downloading-models)
  - [AI Features](#ai-features)
    - [Using AI Models](#using-ai-models)
    - [Troubleshooting AI](#troubleshooting-ai)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Docker-Specific Issues](#docker-specific-issues)
  - [Advanced Usage](#advanced-usage)
    - [Customizing AI Models](#customizing-ai-models)
    - [Optimizing Performance](#optimizing-performance)

---

## Installation

### Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.9+**: Required for running the scripts.
- **Docker**: Required for running the project in a containerized environment.
- **FFmpeg**: Required for audio processing.
- **CUDA Toolkit** (Optional): Required for GPU acceleration with AI models.

### Setting Up the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/TQT-Public/speech-parser-docker.git
   cd speech-parser-docker
   ```

2. **Install Dependencies**:

   - **Using Docker**:
     Build and run the Docker container:

     ```bash
     docker-compose up --build
     ```

   - **Running Locally**:
     Create a Python virtual environment and install dependencies:

     ```bash
     python3 -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     pip install -r requirements.txt
     ```

### Installing FFmpeg

FFmpeg is required for audio processing. Follow the instructions for your operating system:

- **Windows**: Download from [FFmpeg's website](https://ffmpeg.org/download.html) and add it to your system's PATH.
- **Linux**: Install via package manager:

  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

- **macOS**: Install via Homebrew:

  ```bash
  brew install ffmpeg
  ```

### Setting Up CUDA (Optional)

If you plan to use GPU acceleration for AI models, install the CUDA Toolkit and cuDNN:

1. **Install CUDA Toolkit**:
   - Download from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
2. **Install cuDNN**:
   - Download from [NVIDIA's website](https://developer.nvidia.com/cudnn-downloads).
3. **Verify Installation**:
   - Run `nvidia-smi` to check GPU availability.
   - Install PyTorch with CUDA support:

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

---

## Running the Project

### Using Docker

1. **Build and Run the Docker Container**:

   ```bash
   docker-compose up --build
   ```

2. **Attach to the Container**:

   ```bash
   docker exec -it speech-parser-gpu /bin/bash
   ```

3. **Run the Script**:

   ```bash
   python main.py
   ```

### Running Locally

1. **Activate the Virtual Environment**:

   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Run the Script**:

   ```bash
   python main.py
   ```

---

## Configuration

### Environment Variables

Configure the project by editing the `.env` file. Here’s an example:

```bash
# General Settings
MAX_PROCESSES=3
DRY_RUN=False
ASSIGNSPEAKERS=False
FILTER_UNNECESSARY_RTTM=True
MIN_RTTM_DURATION=2.0

# File Paths
AUDIO_FILE_NAME="ZOOM0067.wav"
WORKSPACE="./sources"
VOSK_MODEL_PATH="./models"
MODEL_NAME="vosk-model-ru-0.42"
OUTPUT_DIR="./output"
OUTPUT_DIR_PARTS="./audio_files/parts"
AUDIOWORKSPACE="./audio_files"
```

### Downloading Models

Download the required models and place them in the correct directories:

1. **Vosk Model**:
   - Download from [Vosk's website](https://alphacephei.com/vosk/models).
   - Place it in `./models/vosk/vosk-model-ru-0.42`.

2. **AI Models**:
   - Use the provided scripts to download models like Mistral, LLaMA, Falcon, and DeepSeek.

---

## AI Features

### Using AI Models

The project supports AI models for dialogue summarization. Configure the model in `.env`:

```bash
AI_MODEL_NAME=mistral
```

### Troubleshooting AI

- **CUDA Errors**: Ensure CUDA and cuDNN are correctly installed.
- **Model Loading Issues**: Verify the model paths and permissions.
- **Docker-Specific Issues**: Check Docker resource allocation and file permissions.

---

## Troubleshooting

### Common Issues

- **Missing Files**: Ensure all paths in `.env` are correct.
- **FFmpeg Errors**: Verify FFmpeg installation and PATH configuration.

### Docker-Specific Issues

- **Resource Allocation**: Adjust `.wslconfig` for WSL2 or Docker settings for resource limits.
- **File Permissions**: Ensure files inside Docker have correct permissions.

---

## Advanced Usage

### Customizing AI Models

You can customize the AI models by modifying the `.env` file and downloading additional models.

### Optimizing Performance

- **GPU Acceleration**: Use CUDA for faster inference.
- **Batch Processing**: Adjust `BATCH_SIZE` in `.env` for larger datasets.

---
