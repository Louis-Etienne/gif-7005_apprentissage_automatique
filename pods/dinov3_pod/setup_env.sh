#!/usr/bin/env bash
# DINOv3's torch.hub code needs Python >= 3.10, the Mammo-FM env is 3.9 -> separate env with the
# Mammo-FM python deps (pinned like Mammo-FM/environment.yml where it matters: numpy<2 for imgaug).
set -euo pipefail

ENV_PATH="${ENV_PATH:-/workspace/conda/dinov3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

source /workspace/conda/miniconda3/etc/profile.d/conda.sh
conda create -y -p "$ENV_PATH" python=3.11
conda activate "$ENV_PATH"

pip install --index-url "$TORCH_INDEX" torch==2.5.1 torchvision==0.20.1
pip install numpy==1.26.4 pandas opencv-python-headless==4.10.0.84 imgaug==0.4.0 albumentations==2.0.0 \
    tensorboard nltk transformers matplotlib pillow

python -c "import torch, imgaug, albumentations; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Activate with: conda activate $ENV_PATH"
