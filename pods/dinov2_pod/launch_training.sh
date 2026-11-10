#!/usr/bin/env bash
# DINOv2 ViT-B/14 + Mammo-FM RetinaNet recipe (same hyper-parameters as pods/mammo_fm_pod/launch_training.sh).
# Uses the Mammo-FM conda env and imports Mammo-FM's dataset / RetinaNet / evaluation code.
set -euo pipefail

## BASE MODEL
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
  --checkpoints /workspace/outputs/dinov2_base_mass/checkpoints \
  --output_path /workspace/outputs/dinov2_base_mass/logs \
  --tensorboard-path /workspace/outputs/dinov2_base_mass/tensorboard


## SMALL MODEL
python -u /workspace/gif-7005_apprentissage_automatique/pods/dinov2_pod/train_detector.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --data-dir /workspace/data/vindr_png \
  --img-dir /workspace/data/vindr_png/images_png \
  --csv-file /workspace/data/vindr_png/vindr_detection_v1_folds.csv \
  --dataset ViNDr \
  --arch dinov2_vits14 \
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
  --checkpoints /workspace/outputs/dinov2_small_mass/checkpoints \
  --output_path /workspace/outputs/dinov2_small_mass/logs \
  --tensorboard-path /workspace/outputs/dinov2_small_mass/tensorboard


  ## SMALL MODEL - suspicious_calcification
python -u /workspace/gif-7005_apprentissage_automatique/pods/dinov2_pod/train_detector.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --data-dir /workspace/data/vindr_png \
  --img-dir /workspace/data/vindr_png/images_png \
  --csv-file /workspace/data/vindr_png/vindr_detection_v1_folds.csv \
  --dataset ViNDr \
  --arch dinov2_vits14 \
  --dinov2-cache-dir /workspace/checkpoints \
  --epochs 120 \
  --batch-size 7 \
  --freeze_backbone n \
  --data_frac 1.0 \
  --concepts "Suspicious Calcification" \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive n \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2 \
  --resume auto \
  --checkpoints /workspace/outputs/dinov2_small_suspicious_calcification/checkpoints \
  --output_path /workspace/outputs/dinov2_small_suspicious_calcification/logs \
  --tensorboard-path /workspace/outputs/dinov2_small_suspicious_calcification/tensorboard