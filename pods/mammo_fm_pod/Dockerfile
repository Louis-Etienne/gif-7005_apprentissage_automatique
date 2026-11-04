# ============================================================
# Mammo-FM GPU Environment
#
# Python: 3.9.21
# PyTorch: 2.2.2
# CUDA: 11.8
# Ubuntu: 22.04
# ============================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL maintainer="Mammo-FM Research Environment"
LABEL description="Reproducible CUDA 11.8 environment for Mammo-FM"

# ------------------------------------------------------------
# Build arguments
# ------------------------------------------------------------

ARG MAMMO_FM_REF=main
ARG CONDA_ENV_NAME=Mammo-FM

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV NLTK_DATA=/opt/nltk_data

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    cmake \
    ninja-build \
    libaio-dev \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install Miniconda
# ------------------------------------------------------------

ENV CONDA_DIR=/opt/conda

RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN conda config --set always_yes yes && \
    conda config --set channel_priority flexible && \
    conda install -n base -c conda-forge mamba && \
    conda clean -afy

# ------------------------------------------------------------
# Clone Mammo-FM
# ------------------------------------------------------------

WORKDIR /opt

RUN git clone \
    https://github.com/batmanlab/Mammo-FM.git \
    /opt/Mammo-FM && \
    git -C /opt/Mammo-FM checkout "${MAMMO_FM_REF}"

WORKDIR /opt/Mammo-FM

# ------------------------------------------------------------
# Prepare upstream Conda environment
#
# Keep the original environment specifications, except:
# - Remove the CUDA 12.6 metadata pin.
# - Remove the machine-specific prefix.
# - Defer Flash Attention until PyTorch is installed.
# - Remove CUDA 12 pip runtime packages because this image
#   targets the CUDA 11.8 PyTorch build.
# ------------------------------------------------------------

RUN cp environment.yml environment.docker.yml && \
    sed -i \
      -e '/cuda-version=12\.6/d' \
      -e '/^prefix:/d' \
      -e '/flash-attn==/d' \
      -e '/nvidia-.*-cu12==/d' \
      environment.docker.yml

# ------------------------------------------------------------
# Create Mammo-FM environment
# ------------------------------------------------------------

RUN mamba env create \
    -n ${CONDA_ENV_NAME} \
    -f environment.docker.yml && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=Mammo-FM
ENV PATH="${CONDA_DIR}/envs/Mammo-FM/bin:${PATH}"

# ------------------------------------------------------------
# Install Flash Attention
#
# Install after PyTorch is available.
# CUDA development toolkit is provided by the base image.
# ------------------------------------------------------------

ENV MAX_JOBS=4

RUN python -c "import torch; print(torch.__version__, torch.version.cuda)" && \
    pip install --no-cache-dir \
      flash-attn==2.7.3 \
      --no-build-isolation

# ------------------------------------------------------------
# Prepare NLTK resources
# ------------------------------------------------------------

RUN mkdir -p ${NLTK_DATA} && \
    python -c "import nltk; nltk.download('punkt_tab', download_dir='${NLTK_DATA}')"

# ------------------------------------------------------------
# Create workspace directories
# ------------------------------------------------------------

RUN mkdir -p \
    /workspace/data \
    /workspace/checkpoints \
    /workspace/outputs

# ------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------

WORKDIR /opt/Mammo-FM

ENV PYTHONPATH=/opt/Mammo-FM

# Default interactive shell
CMD ["/bin/bash"]