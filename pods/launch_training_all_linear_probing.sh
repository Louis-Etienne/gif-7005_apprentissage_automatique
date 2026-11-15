#!/usr/bin/env bash
# Linear-probing training of every model on the 3 concepts: same recipe as the fine-tuning runs
# (pods/*/launch_training.sh) but with the backbone frozen (--freeze_backbone y), so only the detector
# head is trained. Each run goes to /workspace/outputs_linear_probing/<model>_<task>/{checkpoints,logs,tensorboard}.
# Once a run has finished all its epochs, only its *_best_auroc.pth and last-epoch *_epoch_<EPOCHS-1>.pth
# checkpoints are kept. A failing run is logged and skipped; a summary is printed at the end.
# Evaluate the runs afterwards with launch_eval_all_linear_probing.sh.
#
# Usage:
#   bash launch_training_all_linear_probing.sh                 # every model x concept
#   bash launch_training_all_linear_probing.sh dinov3 mass     # only runs whose name contains all the filters
#   SKIP_EXISTING=1 bash launch_training_all_linear_probing.sh # skip runs that already have a *_best_auroc.pth
# DINOv3 weights are gated: on the first run set DINOV3_WEIGHTS (see pods/dinov3_pod/launch_training.sh).
set -uo pipefail

PODS_DIR=/workspace/gif-7005_apprentissage_automatique/pods
OUTPUTS_DIR=${OUTPUTS_DIR:-/workspace/outputs_linear_probing}
LOG_DIR=${LOG_DIR:-$OUTPUTS_DIR/logs_launch}
MAMMO_FM_CODEBASE=/workspace/Mammo-FM/src/codebase
DINOV3_PYTHON=/workspace/conda/dinov3/bin/python
MAMMO_FM_PYTHON=/workspace/conda/mammo_fm/bin/python
SKIP_EXISTING=${SKIP_EXISTING:-0}
DINOV3_WEIGHTS="${DINOV3_WEIGHTS:-}"
EPOCHS=120

MODELS=(mammo_fm mammo_clip dinov2_small dinov2_base dinov3_small dinov3_base)
TASKS=(mass suspicious_calcification focal_asymmetry)
declare -A CONCEPTS=(
  [mass]="Mass"
  [suspicious_calcification]="Suspicious Calcification"
  [focal_asymmetry]="Focal Asymmetry"
)

COMMON_ARGS=(
  --data-dir /workspace/data/vindr_png
  --img-dir /workspace/data/vindr_png/images_png
  --csv-file /workspace/data/vindr_png/vindr_detection_v1_folds.csv
  --dataset ViNDr
  --epochs "$EPOCHS"
  --batch-size 7
  --freeze_backbone y
  --data_frac 1.0
  --print-freq 5000
  --log-freq 300
  --running-interactive n
  --focal-alpha 0.25
  --focal-gamma 2.0
  --score-threshold 0.2
)

# Keep only *_best_auroc.pth and the last-epoch checkpoint of a finished run (also drops *_last.pth).
prune_checkpoints() {
  local ckpt_dir=$1 last_epoch="_epoch_$((EPOCHS - 1)).pth"
  if [[ -z $(find "$ckpt_dir" -name "*$last_epoch" | head -n 1) ]]; then
    echo "    [keep all] no *$last_epoch, run not finished"
    return
  fi
  local n
  n=$(find "$ckpt_dir" -name "*.pth" ! -name "*_best_auroc.pth" ! -name "*$last_epoch" -print -delete | wc -l)
  echo "    pruned $n checkpoints (kept best_auroc + epoch $((EPOCHS - 1)))"
}

mkdir -p "$LOG_DIR"
succeeded=()
failed=()
skipped=()

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    run="${model}_${task}"
    for filter in "$@"; do
      [[ "$run" == *"$filter"* ]] || continue 2
    done

    run_dir="$OUTPUTS_DIR/$run"
    if [[ "$SKIP_EXISTING" == "1" ]] && [[ -n $(find "$run_dir" -name "*_best_auroc.pth" 2>/dev/null | head -n 1) ]]; then
      echo "[skip] $run: checkpoint already in $run_dir"
      skipped+=("$run (already trained)")
      continue
    fi

    case "$model" in
      mammo_fm)
        python_bin=$MAMMO_FM_PYTHON
        cmd=("$MAMMO_FM_CODEBASE/train_detector.py" --arch breast_clip_det_b5
             --clip_chk_pt_path /workspace/checkpoints/Mammo-FM_BatmanlabTrained_CLIP.tar)
        ;;
      mammo_clip)
        python_bin=$MAMMO_FM_PYTHON
        cmd=("$MAMMO_FM_CODEBASE/train_detector.py" --arch breast_clip_det_b5
             --clip_chk_pt_path /workspace/checkpoints/b5-model-best-epoch-7.tar)
        ;;
      dinov2_*)
        python_bin=$MAMMO_FM_PYTHON
        arch=$([[ "$model" == *_small ]] && echo dinov2_vits14 || echo dinov2_vitb14)
        cmd=("$PODS_DIR/dinov2_pod/train_detector.py" --mammo-fm-codebase "$MAMMO_FM_CODEBASE"
             --arch "$arch" --dinov2-cache-dir /workspace/checkpoints --resume auto)
        ;;
      dinov3_*)
        python_bin=$DINOV3_PYTHON
        arch=$([[ "$model" == *_small ]] && echo dinov3_vits16 || echo dinov3_vitb16)
        cmd=("$PODS_DIR/dinov3_pod/train_detector.py" --mammo-fm-codebase "$MAMMO_FM_CODEBASE"
             --arch "$arch" --dinov3-cache-dir /workspace/checkpoints --resume auto)
        if [[ -n "$DINOV3_WEIGHTS" ]]; then cmd+=(--dinov3-weights "$DINOV3_WEIGHTS"); fi
        ;;
    esac

    log="$LOG_DIR/$run.log"
    echo "=== $run | concept=${CONCEPTS[$task]} | backbone frozen"
    echo "    log: $log"
    if "$python_bin" -u "${cmd[@]}" "${COMMON_ARGS[@]}" \
        --concepts "${CONCEPTS[$task]}" \
        --checkpoints "$run_dir/checkpoints" \
        --output_path "$run_dir/logs" \
        --tensorboard-path "$run_dir/tensorboard" > "$log" 2>&1; then
      echo "    done"
      prune_checkpoints "$run_dir/checkpoints"
      succeeded+=("$run")
    else
      echo "    FAILED (see $log), checkpoints kept"
      failed+=("$run")
    fi
  done
done

echo
echo "===================== Summary ====================="
echo "Succeeded (${#succeeded[@]}):"; printf '  %s\n' "${succeeded[@]}"
echo "Failed (${#failed[@]}):";       printf '  %s\n' "${failed[@]}"
echo "Skipped (${#skipped[@]}):";     printf '  %s\n' "${skipped[@]}"

[[ ${#failed[@]} -eq 0 ]]
