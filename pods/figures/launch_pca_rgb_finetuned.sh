#!/usr/bin/env bash
# PCA-RGB figure of the fine-tuned detector backbones: a single figure with rows grouped by concept,
# each row using the backbones fine-tuned on that concept (best-mAP checkpoint in
# /workspace/outputs_finetune/<model>_<task>), the PCA being fitted per checkpoint. Rows of a concept a model
# has no checkpoint for are left black; models without any checkpoint are left out.
#
# Usage: bash launch_pca_rgb_finetuned.sh [OUTPUT_DIR]   (default /workspace/eval/pca_rgb_finetuned)
# Writes OUTPUT_DIR/pca_rgb_comparison.png. Run in the dinov3 env.
set -euo pipefail

OUTPUT_DIR=${1:-/workspace/eval/pca_rgb_finetuned}
OUTPUTS_DIR=/workspace/outputs_finetune
MODELS=(mammo_fm mammo_clip dinov2_small dinov2_base dinov3_small dinov3_base)
TASKS=(mass suspicious_calcification focal_asymmetry)
declare -A CONCEPTS=(
  [mass]="Mass"
  [suspicious_calcification]="Suspicious Calcification"
  [focal_asymmetry]="Focal Asymmetry"
)

models=()
ckpts=()
for model in "${MODELS[@]}"; do
  found=0
  for task in "${TASKS[@]}"; do
    ckpt=$(find "$OUTPUTS_DIR/${model}_${task}" -name "*_best_auroc.pth" 2>/dev/null | sort | head -n 1)
    if [[ -n "$ckpt" ]]; then
      ckpts+=("$model@${CONCEPTS[$task]}=$ckpt")
      found=1
    else
      echo "[skip] $model: no fine-tuned checkpoint for $task"
    fi
  done
  if [[ $found -eq 1 ]]; then models+=("$model"); fi
done
if [[ ${#models[@]} -eq 0 ]]; then
  echo "No fine-tuned checkpoints in $OUTPUTS_DIR"
  exit 1
fi

concepts=()
for task in "${TASKS[@]}"; do concepts+=("${CONCEPTS[$task]}"); done

echo "=== ${models[*]}"
python -u /workspace/gif-7005_apprentissage_automatique/pods/figures/make_pca_rgb_figure.py \
  --mammo-fm-codebase /workspace/Mammo-FM/src/codebase \
  --concepts "${concepts[@]}" --num-images 2 --grid 48 28 \
  --models "${models[@]}" \
  --detector-ckpt "${ckpts[@]}" \
  --output-dir "$OUTPUT_DIR"
