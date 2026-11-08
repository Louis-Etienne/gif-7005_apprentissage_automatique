#!/usr/bin/env bash
# DINOv2 ViT-B/14 + Mammo-FM RetinaNet recipe (same hyper-parameters as pods/mammo_fm_pod/launch_training.sh).
# Uses the Mammo-FM conda env and imports Mammo-FM's dataset / RetinaNet / evaluation code.
set -euo pipefail

python -u /workspace/gif-7005_apprentissage_automatique/pods/dinov2_pod/train_detector.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --data-dir /workspace/data/vindr_png \
  --img-dir /workspace/data/vindr_png/images_png \
  --csv-file /workspace/data/vindr_png/vindr_detection_v1_folds.csv \
  --dataset ViNDr \
  --arch dinov2_vitb14 \
  --dinov2-cache-dir /workspace/checkpoints \
  --epochs 120 \
  --batch-size 7 \
  --freeze_backbone n \
  --data_frac 1.0 \
  --concepts Mass \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive n \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2 \
  --resume auto \
  --checkpoints /workspace/outputs/checkpoints \
  --output_path /workspace/outputs/logs \
  --tensorboard-path /workspace/outputs/tensorboard
