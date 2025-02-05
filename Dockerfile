# Use the official NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Miniconda
RUN apt-get update && \
    apt-get install -y software-properties-common wget curl git ffmpeg nano unzip && \
    apt-get install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/* && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Copy the environment file and create the Conda environment
COPY environment.yml /app/environment.yml
WORKDIR /app
RUN conda env create -f environment.yml && conda clean -afy
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Install additional pip dependencies
RUN pip install unsloth loguru python-dotenv torch torchaudio pandas numpy scipy matplotlib plotly scikit-learn pydub pyannote-audio aiohttp vosk nltk rake-nltk requests transformers diffusers anthropic deepseek

# Set the working directory and copy project files
COPY . /app

# Expose required port (if needed)
EXPOSE 2700

# Set default command
CMD ["python", "main.py"]
