#!/usr/bin/env bash
# Evaluate the DINOv2 detectors trained by launch_training.sh (best-mAP checkpoint of each run).
# Same env as training (Mammo-FM conda env); --arch must match the training run.
set -euo pipefail

## BASE MODEL
python -u /workspace/gif-7005_apprentissage_automatique/pods/dinov2_pod/evaluate_detector.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --dataset ViNDr \
  --data-dir /workspace/data/vindr_png \
  --img-dir /workspace/data/vindr_png/images_png \
  --csv-file vindr_detection_v1_folds.csv \
  --arch dinov2_vitb14 \
  --dinov2-cache-dir /workspace/checkpoints \
  --concepts Mass \
  --chk-pt-path "/workspace/outputs/dinov2_base_mass/checkpoints/ViNDr/Concept-Detector/dinov2_vitb14/lr_5e-05_epochs_120_concepts_Mass_alpha_0.25_gamma_2.0_score_th_0.2_data_frac_1.0_freeze_backbone_n/dinov2_vitb14_seed_10_fold_0_best_auroc.pth" \
  --batch-size 7 \
  --num-workers 4 \
  --iou-threshold 0.5 \
  --score-threshold 0.2 \
  --max-detections 100 \
  --visualization-dir /workspace/eval/dinov2_base_mass \
  --num-examples 30

## SMALL MODEL
python -u /workspace/gif-7005_apprentissage_automatique/pods/dinov2_pod/evaluate_detector.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --dataset ViNDr \
  --data-dir /workspace/data/vindr_png \
  --img-dir /workspace/data/vindr_png/images_png \
  --csv-file vindr_detection_v1_folds.csv \
  --arch dinov2_vits14 \
  --dinov2-cache-dir /workspace/checkpoints \
  --concepts Mass \
  --chk-pt-path "/workspace/outputs/dinov2_small_mass/checkpoints/ViNDr/Concept-Detector/dinov2_vits14/lr_5e-05_epochs_120_concepts_Mass_alpha_0.25_gamma_2.0_score_th_0.2_data_frac_1.0_freeze_backbone_n/dinov2_vits14_seed_10_fold_0_best_auroc.pth" \
  --batch-size 7 \
  --num-workers 4 \
  --iou-threshold 0.5 \
  --score-threshold 0.2 \
  --max-detections 100 \
  --visualization-dir /workspace/eval/dinov2_small_mass \
  --num-examples 30

## SMALL MODEL - suspicious_calcification
python -u /workspace/gif-7005_apprentissage_automatique/pods/dinov2_pod/evaluate_detector.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --dataset ViNDr \
  --data-dir /workspace/data/vindr_png \
  --img-dir /workspace/data/vindr_png/images_png \
  --csv-file vindr_detection_v1_folds.csv \
  --arch dinov2_vits14 \
  --dinov2-cache-dir /workspace/checkpoints \
  --concepts "Suspicious Calcification" \
  --chk-pt-path "/workspace/outputs/dinov2_small_suspicious_calcification/checkpoints/ViNDr/Concept-Detector/dinov2_vits14/lr_5e-05_epochs_120_concepts_Suspicious Calcification_alpha_0.25_gamma_2.0_score_th_0.2_data_frac_1.0_freeze_backbone_n/dinov2_vits14_seed_10_fold_0_best_auroc.pth" \
  --batch-size 7 \
  --num-workers 4 \
  --iou-threshold 0.5 \
  --score-threshold 0.2 \
  --max-detections 100 \
  --visualization-dir /workspace/eval/dinov2_small_suspicious_calcification \
  --num-examples 30
