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
docker build --pull --no-cache --rm -f "Dockerfile" -t speech-parser-gpu:latest "."
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

## Troubleshooting

### If you encounter issues such as missing files, ensure the directories are properly configured in your .env file

### 1. **Optimize Docker Resource Usage (WSL2 Configuration)**

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

### Re-download Vosk inside Docker

```bash
# Inside Docker, re-download the Vosk model:
wget https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip
unzip vosk-model-ru-0.42.zip -d /app/models

```

### Ensure Correct File Permissions

#### Make sure that the files in `/app/models/vosk-model-ru-0.42/graph/words.txt` and the other model files have the correct read permissions

### In Docker, check file permissions

```bash
ls -l /app/models/vosk-model-ru-0.42/graph/words.txt

```

### Ensure it has read permissions for all users, like

```bash
-rw-r--r-- 1 root root 12345 Jan 25 12:34 words.txt

```

### Clean up the old build

```bash
docker-compose down --rmi all --volumes --remove-orphans

```

### Check Python version of libraries

```bash
python -c "import numpy; print(numpy.__version__)"

```

### Move and Delete AI folders

```bash
# Watch for (copy=0, delete=1) == just delete
# in .env => MODEL_ENABLE_COPY=True
#            MODEL_MOVE_AND_DELETE=True
python -m scripts.move_model_out llama

```

### Check File Encoding

```bash
file -i /app/models/vosk-model-ru-0.42/graph/words.txt
# /app/models/vosk-model-ru-0.42/graph/words.txt: text/plain; charset=utf-8

```

### Use an absolute path in Docker

#### In Docker, relative paths may cause issues. To avoid any confusion, ensure you are using an absolute path to the Vosk model in your Docker container

```bash
# Docker File Paths
AUDIO_FILE_NAME = "/app/sources/ZOOM0067.wav"
WORKSPACE="/app/sources"
VOSK_MODEL_PATH="/app/models"
MODEL_NAME="vosk-model-ru-0.42"
OUTPUT_DIR="/app/output"
OUTPUT_DIR_PARTS="/app/audio_files/parts"
AUDIOWORKSPACE="/app/audio_files"
CONFIG_PATH="/app/configs"
SCRIPT_PATH="/app/scripts"
```

#### If the environment variable is set like this in .env

```bash
VOSK_MODEL_PATH=/app/models/vosk-model-ru-0.42
```

#### Update your code to ensure that the path is absolute

```bash
model_path = Path(VOSK_MODEL_PATH_ENV).resolve()  # Ensures it's an absolute path
```

### Ensure Docker container uses the same Vosk model version as local

#### If the model works fine outside of Docker, ensure that the version of Vosk inside Docker is the same as your local setup. If the wrong version is being installed in Docker, it could cause incompatibility with the model files

#### Specify the exact version of Vosk in your requirements.txt or Dockerfile

```bash
vosk==0.3.32
```

#### Then rebuild your Docker container to ensure the correct version is installed

```bash
docker-compose build
```

## Installation

Here’s an updated **README** section for installing `ffmpeg` on **Windows**, **Linux**, and **macOS**, along with a general revision of the installation instructions to ensure clarity and accuracy. You can integrate this into your repository's `README.md`.

---

## Installation Instructions

### Prerequisites (all OS)

Before setting up the project, ensure that you have the following prerequisites installed on your system:

- Docker (for running in a containerized environment)
- Python 3.9 or higher (for local development outside Docker)
- Git
- FFmpeg (for audio processing)
- Vosk model (download and place in the correct directory)

### **Step 0: Set Up CUDA Toolkit locally**

1. **Install CUDA Toolkit**:
   - Download and install the CUDA Toolkit compatible with your GPU (GTX 1070 Ti supports CUDA 11.x).
   - Follow the official instructions: [CUDA Toolkit Documentation](https://developer.nvidia.com/cuda-downloads).

2. **Install cuDNN**:
   - Download and install cuDNN (CUDA Deep Neural Network library) for your CUDA version.
   - Follow the official instructions: [cuDNN Documentation](https://developer.nvidia.com/cudnn-downloads): [All NVIDIA Downloads](https://developer.nvidia.com/downloads).

3. **Verify Installation**:
   - Run `nvidia-smi` to check if your GPU is recognized.
   - Install PyTorch with CUDA support:

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

   - Verify PyTorch can use CUDA:

     ```python
     import torch
     print(torch.cuda.is_available())  # Should return True
     ```

---

### Installing FFmpeg

FFmpeg is required for converting audio files to the appropriate format for Vosk.

#### **Windows**

To install FFmpeg on Windows:

1. **Download FFmpeg**:
   - Visit the official FFmpeg website: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - Click on the "Windows" logo and follow the instructions for downloading the build.

2. **Set FFmpeg Path**:
   - Unzip the downloaded file.
   - Move the `bin` folder (which contains `ffmpeg.exe`) to a permanent location.
   - Add the path to `ffmpeg.exe` in the system environment variables:
     - Search for "Environment Variables" in Windows.
     - Edit the `Path` variable in the "System variables" section.
     - Add the full path to the `bin` directory containing `ffmpeg.exe`.

3. **Verify Installation**:
   Open Command Prompt and run:

   ```sh
   ffmpeg -version
   ```

   You should see the FFmpeg version output.

#### **Linux (Ubuntu/Debian-based)**

To install FFmpeg on Linux:

1. **Update package lists**:

   ```sh
   sudo apt update
   ```

2. **Install FFmpeg**:

   ```sh
   sudo apt install ffmpeg
   ```

3. **Verify Installation**:

   ```sh
   ffmpeg -version
   ```

#### **macOS**

To install FFmpeg on macOS:

1. **Using Homebrew** (recommended):
   - If you don’t have Homebrew installed, install it first:

     ```sh
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```

2. **Install FFmpeg**:

   ```sh
   brew install ffmpeg
   ```

3. **Verify Installation**:

   ```sh
   ffmpeg -version
   ```

### Setting Up the Project

1. **Clone the repository**:

   ```sh
   git clone https://github.com/TQT-Public/speech-parser-docker.git
   cd speech-parser-docker
   ```

2. **Install dependencies**:

   - **If using Docker**:

     - Build and run the Docker container:

       ```sh
       docker-compose up --build
       ```

   - **If running locally** (outside Docker):
     - Create a Python virtual environment:

       ```sh
       python3 -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       ```

     - Install Python dependencies:

       ```sh
       pip install -r requirements.txt
       ```

### Running the Project

#### **Using Docker**

If you have Docker installed, simply run:

```sh
docker-compose up
```

This will pull the necessary Docker image, build the environment, and run the application.

#### **Running Locally** (without Docker)

1. Ensure FFmpeg is installed and added to your system’s path.
2. Activate your Python virtual environment (if not already activated):

   ```sh
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Run the script:

   ```sh
   python main.py
   ```

### Downloading Vosk Models

To run the speech recognition system, you will need to download and use the Vosk model. The model should be placed inside the `models` directory of the project.

1. **Download Vosk model**:
   - Download a Vosk model from [https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).
   - Unzip the model and place it in the `models` directory, e.g., `models/vosk-model-ru-0.42`.

2. **Ensure that the `.env` file** correctly points to the Vosk model directory:

   ```bash
   VOSK_MODEL_PATH="./models"
   MODEL_NAME="vosk-model-ru-0.42"
   ```

---

This updated README ensures that all users across Windows, Linux, and macOS have clear instructions for installing FFmpeg, setting up the environment, and running the project.

## AI features

### **Downloading AI Models**

Update `.env` variables

```bash
# Model Names
VOSK_MODEL_NAME=vosk-model-ru-0.42
LLAMA_MODEL_NAME=llama-2-7b
MISTRAL_MODEL_NAME=mistral-7b-v0.1
FALCON_MODEL_NAME=falcon-7b
DEEPSEEK_MODEL_NAME=DeepSeek-R1-Distill-Llama-8B
STABLE_DIFFUSION_MODEL_NAME=stable-diffusion-v1-5

# Model Paths
MODELS_DIR=./models
VOSK_MODEL_PATH=${MODELS_DIR}/vosk/${VOSK_MODEL_NAME}
LLAMA_MODEL_PATH=${MODELS_DIR}/ai/llama/${LLAMA_MODEL_NAME}
MISTRAL_MODEL_PATH=${MODELS_DIR}/ai/mistral/${MISTRAL_MODEL_NAME}
FALCON_MODEL_PATH=${MODELS_DIR}/ai/falcon/${FALCON_MODEL_NAME}
DEEPSEEK_MODEL_PATH=${MODELS_DIR}/ai/deepseek/${DEEPSEEK_MODEL_NAME}
STABLE_DIFFUSION_MODEL_PATH=${MODELS_DIR}/stable_diffusion/${STABLE_DIFFUSION_MODEL_NAME}
```

### To download the models into the correct subfolders, use the following commands one-liners

1. **DeepSeek-R1-Distill-Llama-8B**:

   ```bash
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('unsloth/DeepSeek-R1-Distill-Llama-8B').save_pretrained('./models/ai/deepseek/DeepSeek-R1-Distill-Llama-8B')"
   ```

   ```bash
   python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('unsloth/DeepSeek-R1-Distill-Llama-8B'); tokenizer = AutoTokenizer.from_pretrained('unsloth/DeepSeek-R1-Distill-Llama-8B'); model.save_pretrained('./models/ai/deepseek/DeepSeek-R1-Distill-Llama-8B'); tokenizer.save_pretrained('./models/ai/deepseek/DeepSeek-R1-Distill-Llama-8B')"
   ```

2. **LLaMA 2**:
    For detailed info and access look at [Llama page](https://huggingface.co/meta-llama/Llama-2-7b-hf)

   ```bash
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token='your_API_key').save_pretrained('./models/ai/llama/llama-2-7b')"
   ```

   ```bash
   python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token='your_API_key'); tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token='your_API_key'); model.save_pretrained('./models/ai/llama/llama-2-7b'); tokenizer.save_pretrained('./models/ai/llama/llama-2-7b')"
    ```

3. **Mistral 7B**:
    For detailed info and access look at [Mistral page](https://huggingface.co/mistralai/Mistral-7B-v0.1)

   ```bash
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', use_auth_token='your_API_key').save_pretrained('./models/ai/mistral/mistral-7b-v0.1')"
   ```

    ```bash
   python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', token='your_API_key'); tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', token='your_API_key'); model.save_pretrained('./models/ai/mistral/mistral-7b-v0.1'); tokenizer.save_pretrained('./models/ai/mistral/mistral-7b-v0.1')"
   ```

4. **Falcon 7B**:

   ```bash
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('tiiuae/falcon-7b').save_pretrained('./models/ai/falcon/falcon-7b')"
    ```

    ```bash
   python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-7b'); tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-7b'); model.save_pretrained('./models/ai/falcon/falcon-7b'); tokenizer.save_pretrained('./models/ai/falcon/falcon-7b')"
   ```

5. **Stable Diffusion**:

   ```bash
   python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5').save_pretrained('./models/stable_diffusion/stable-diffusion-v1-5')"
   ```

## Trouble-shooting AI

### Registry setting to enable long paths (Windows)

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### Find `site-packages\unsloth\models\llama.py` if facing errors with `LlamaModel_fast_forward_inference`

```bash
conda env list
cd <path_to_conda_env>/lib/python3.x/site-packages/unsloth/models/
ls -l | grep llama.py

```

`<path_to_conda_env>` on Windows: `{CURRENT_USER}\.conda\envs\unsloth_env\Lib\site-packages\unsloth\models\llama.py`

```python llama.py
# in -> def LlamaModel_fast_forward_inference (line ~900)
# Ensure hidden_states are cast to a valid PyTorch dtype
if isinstance(self.config.torch_dtype, str):
    if self.config.torch_dtype == 'float32':
        dtype = torch.float32
    elif self.config.torch_dtype == 'float16':
        dtype = torch.float16
    else:
        raise ValueError(f"Invalid torch_dtype: {self.config.torch_dtype}")
else:
    dtype = self.config.torch_dtype

# Cast hidden_states to the correct dtype
hidden_states = hidden_states.to(dtype)
# hidden_states = hidden_states.to(self.config.torch_dtype) # Original
```

### No module named "triton" Windows issue

#### Look at [link](https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp310-cp310-win_amd64.whl) and [link](https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp310-cp310-win_amd64.whl)

```bash
pip install triton-3.0.0-cp310-cp310-win_amd64.whl
```

### Correct ENV for conda+unsloth

```bash
conda create --name unsloth_env python=3.11 pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
conda activate unsloth_env
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes torchaudio
pip install python-dotenv loguru tqdm pyannote-audio xformers --index-url https://download.pytorch.org/whl/cu121
pip install vosk diffusers pydub
```
