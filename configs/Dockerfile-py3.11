# Use an NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common wget curl git ffmpeg nano unzip && \
    apt-get install -y nvidia-container-toolkit && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    apt-get clean

# Verify the Python version
RUN python3 --version

# Install pip manually for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install tqdm loguru python-dotenv torch torchaudio pandas numpy scipy matplotlib plotly scikit-learn pydub pyannote-audio aiohttp vosk nltk rake-nltk requests transformers diffusers

# Install additional dependencies for Stable Diffusion (if needed)
RUN pip install accelerate safetensors

# Set the working directory and copy project files
WORKDIR /app
COPY . /app

# Expose ports if needed (e.g., for a Vosk server)
EXPOSE 2700

# Set the default command to run your application
CMD ["python3", "main.py"]

# [Troubleshooting]
# CMD ["/bin/bash"] # To load to console