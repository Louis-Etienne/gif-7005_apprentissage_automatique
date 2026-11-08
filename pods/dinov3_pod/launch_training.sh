#!/usr/bin/env bash
# DINOv3 ViT-B/16 + Mammo-FM RetinaNet recipe (same hyper-parameters as pods/mammo_fm_pod/launch_training.sh).
# Needs a Python >= 3.10 env (see setup_env.sh); imports Mammo-FM's dataset / RetinaNet / evaluation code.
#
# DINOv3 weights are gated: request access from Meta, then on the FIRST run set DINOV3_WEIGHTS to the
# emailed URL for dinov3_vitb16 (or a downloaded .pth). They are saved in /workspace/checkpoints and
# reused automatically afterwards (DINOV3_WEIGHTS can then be left empty).
set -euo pipefail

DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-}"
WEIGHTS_ARGS=()
if [[ -n "$DINOV3_WEIGHTS" ]]; then WEIGHTS_ARGS=(--dinov3-weights "$DINOV3_WEIGHTS"); fi

python -u /workspace/gif-7005_apprentissage_automatique/pods/dinov3_pod/train_detector.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --data-dir /workspace/data/vindr_png \
  --img-dir /workspace/data/vindr_png/images_png \
  --csv-file /workspace/data/vindr_png/vindr_detection_v1_folds.csv \
  --dataset ViNDr \
  --arch dinov3_vitb16 \
  --dinov3-cache-dir /workspace/checkpoints \
  ${WEIGHTS_ARGS[@]+"${WEIGHTS_ARGS[@]}"} \
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
  --checkpoints /workspace/outputs/dinov3_mass/checkpoints \
  --output_path /workspace/outputs/dinov3_mass/logs \
  --tensorboard-path /workspace/outputs/dinov3_mass/tensorboard
