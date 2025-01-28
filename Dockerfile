# # # Use an NVIDIA CUDA base image with Ubuntu 20.04
# FROM ubuntu:20.04 - Try variations if facing troubles
# FROM nvidia/cuda:11.8.0-base-ubuntu20.04
# FROM nvidia/cuda:11.0-base
# FROM nvidia/cuda:11.4-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and set it as the default Python version
RUN apt-get update && \
    apt-get install -y software-properties-common wget curl git ffmpeg nano unzip && \
    apt-get install -y nvidia-container-toolkit && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-venv python3.9-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    apt-get clean


# # # Alternative: Python 3.11
# RUN apt-get update && \
#     apt-get install -y software-properties-common wget curl git ffmpeg nano unzip && \
#     apt-get install -y nvidia-container-toolkit && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.11 python3.11-venv python3.11-dev && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
#     apt-get clean
# - - [Optional]
#     && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
#     && python3 -m pip install --no-cache-dir html5lib \
#     && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Verify the Python version
RUN python3 --version

# Install pip manually for Python 3.9
RUN apt-get update && \
    apt-get install -y python3-pip && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    apt-get clean

# Install Python dependencies (Poetry\requirements.txt)
# RUN python3 -m pip install poetry==1.6.1
# RUN poetry lock --no-interaction && poetry install --no-root
# OR
RUN pip install tqdm loguru python-dotenv torch torchaudio pandas numpy scipy matplotlib plotly scikit-learn pydub pyannote-audio aiohttp vosk nltk rake-nltk requests
# Or
# # Install Vosk and other dependencies
# RUN pip install -r requirements.txt
# Set the working directory and copy project files
WORKDIR /app
COPY . /app

# Ensure Poetry dependencies are installed

# Expose ports if needed (e.g., for a Vosk server)
EXPOSE 2700

# RUN nvidia-smi
# Set the default command to run your application
CMD ["python3", "main.py"]
# [Troubleshooting ]
# CMD ["/bin/bash"] # To load to console
# # COPY .venv /app/.venv
# # ENV PATH="/app/.venv/bin:$PATH"