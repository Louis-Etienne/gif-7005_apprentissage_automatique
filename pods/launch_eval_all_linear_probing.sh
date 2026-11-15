#!/usr/bin/env bash
# Evaluate every linear-probing run trained by launch_training_all_linear_probing.sh (best-mAP
# checkpoint of each /workspace/outputs_linear_probing/<model>_<task>) into
# /workspace/eval/metrics_linear_probing/<model>_<task> (+ all_metrics.json), using launch_eval_all.sh.
#
# Usage:
#   bash launch_eval_all_linear_probing.sh                  # every run
#   bash launch_eval_all_linear_probing.sh dinov3 mass      # only runs whose folder name contains all the filters
#   SKIP_EXISTING=1 bash launch_eval_all_linear_probing.sh  # skip runs that already have metrics.json
set -uo pipefail

PODS_DIR=/workspace/gif-7005_apprentissage_automatique/pods
OUTPUTS_DIR=${OUTPUTS_DIR:-/workspace/outputs_linear_probing}
EVAL_DIR=${EVAL_DIR:-/workspace/eval/metrics_linear_probing}

OUTPUTS_DIR="$OUTPUTS_DIR" EVAL_DIR="$EVAL_DIR" bash "$PODS_DIR/launch_eval_all.sh" "$@"
