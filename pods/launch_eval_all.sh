#!/usr/bin/env bash
# Evaluate every trained detector in /workspace/outputs_finetune (best-mAP checkpoint of each run).
#
# Each run folder /workspace/outputs_finetune/<model>_<task> is evaluated into /workspace/eval/metrics_finetune/<model>_<task>.
# The pod, --arch and --concepts are read from the folder and checkpoint names, so new runs are
# picked up automatically. A failing run is logged and skipped; a summary is printed at the end and
# every metrics_finetune/<run>/metrics.json is merged into metrics_finetune/all_metrics.json (model x task mAP table + per-run details).
#
# Usage:
#   bash launch_eval_all.sh                  # every run
#   bash launch_eval_all.sh dinov3 mass      # only runs whose folder name contains all the filters
#   SKIP_EXISTING=1 bash launch_eval_all.sh  # skip runs that already have metrics_finetune/<run>/metrics.json
set -uo pipefail

PODS_DIR=/workspace/gif-7005_apprentissage_automatique/pods
OUTPUTS_DIR=${OUTPUTS_DIR:-/workspace/outputs_finetune}
EVAL_DIR=${EVAL_DIR:-/workspace/eval/metrics_finetune}
LOG_DIR=${LOG_DIR:-$EVAL_DIR/logs}
MAMMO_FM_CODEBASE=/workspace/Mammo-FM/src/codebase
DINOV3_PYTHON=/workspace/conda/dinov3/bin/python
MAMMO_FM_PYTHON=/workspace/conda/mammo_fm/bin/python
SKIP_EXISTING=${SKIP_EXISTING:-0}

COMMON_ARGS=(
  --dataset ViNDr
  --data-dir /workspace/data/vindr_png
  --img-dir /workspace/data/vindr_png/images_png
  --csv-file vindr_detection_v1_folds.csv
  --batch-size 7
  --num-workers 4
  --iou-threshold 0.5
  --score-threshold 0.2
  --max-detections 100
  --num-examples 30
)

mkdir -p "$LOG_DIR"
succeeded=()
failed=()
skipped=()

for run_dir in "$OUTPUTS_DIR"/*/; do
  run=$(basename "$run_dir")

  for filter in "$@"; do
    [[ "$run" == *"$filter"* ]] || continue 2
  done

  ckpt=$(find "$run_dir" -name "*_best_auroc.pth" | sort | head -n 1)
  if [[ -z "$ckpt" ]]; then
    echo "[skip] $run: no *_best_auroc.pth"
    skipped+=("$run (no checkpoint)")
    continue
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$EVAL_DIR/$run/metrics.json" ]]; then
    echo "[skip] $run: $EVAL_DIR/$run/metrics.json exists"
    skipped+=("$run (already evaluated)")
    continue
  fi

  # e.g. .../lr_5e-05_epochs_120_concepts_Suspicious Calcification_alpha_0.25_.../dinov3_vits16_seed_10_fold_0_best_auroc.pth
  concept=$(sed -E 's/.*_concepts_(.*)_alpha_.*/\1/' <<< "$(dirname "$ckpt")")
  arch=$(basename "$ckpt" | sed -E 's/_seed_.*//')

  case "$run" in
    dinov3_*)
      python_bin=$DINOV3_PYTHON
      cmd=("$PODS_DIR/dinov3_pod/evaluate_detector.py" --mammo-fm-codebase "$MAMMO_FM_CODEBASE"
           --arch "$arch" --dinov3-cache-dir /workspace/checkpoints)
      ;;
    dinov2_*)
      python_bin=$MAMMO_FM_PYTHON
      cmd=("$PODS_DIR/dinov2_pod/evaluate_detector.py" --mammo-fm-codebase "$MAMMO_FM_CODEBASE"
           --arch "$arch" --dinov2-cache-dir /workspace/checkpoints)
      ;;
    mammo_fm_*)
      python_bin=$MAMMO_FM_PYTHON
      cmd=("$PODS_DIR/mammo_fm_pod/evaluate_detector.py" --arch "$arch" --freeze-backbone n
           --clip-chk-pt-path /workspace/checkpoints/Mammo-FM_BatmanlabTrained_CLIP.tar)
      ;;
    mammo_clip_*)
      python_bin=$MAMMO_FM_PYTHON
      cmd=("$PODS_DIR/mammo_fm_pod/evaluate_detector.py" --arch "$arch" --freeze-backbone n
           --clip-chk-pt-path /workspace/checkpoints/b5-model-best-epoch-7.tar)
      ;;
    *)
      echo "[skip] $run: unknown model family"
      skipped+=("$run (unknown model)")
      continue
      ;;
  esac

  log="$LOG_DIR/$run.log"
  echo "=== $run | arch=$arch | concept=$concept"
  echo "    checkpoint: $ckpt"
  echo "    log: $log"
  if PYTHONPATH="$MAMMO_FM_CODEBASE${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -u "${cmd[@]}" \
      "${COMMON_ARGS[@]}" \
      --concepts "$concept" \
      --chk-pt-path "$ckpt" \
      --visualization-dir "$EVAL_DIR/$run" > "$log" 2>&1; then
    map=$(grep -oE '"mean_ap": [0-9.eE+-]+' "$EVAL_DIR/$run/metrics.json" 2>/dev/null | awk '{print $2}')
    echo "    done: mAP=${map:-?}"
    succeeded+=("$run mAP=${map:-?}")
  else
    echo "    FAILED (see $log)"
    failed+=("$run")
  fi
done

echo
echo "===================== Summary ====================="
echo "Succeeded (${#succeeded[@]}):"; printf '  %s\n' "${succeeded[@]}"
echo "Failed (${#failed[@]}):";       printf '  %s\n' "${failed[@]}"
echo "Skipped (${#skipped[@]}):";     printf '  %s\n' "${skipped[@]}"

# Joint metrics over every metrics_finetune/<model>_<task>/metrics.json (also the runs not re-evaluated this time).
"$DINOV3_PYTHON" - "$EVAL_DIR" <<'EOF'
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
tasks = ["mass", "suspicious_calcification", "focal_asymmetry"]
runs, table = {}, {}
for path in sorted(eval_dir.glob("*/metrics.json")):
    run = path.parent.name
    task = next((t for t in tasks if run.endswith(f"_{t}")), None)
    if task is None:
        continue
    model = run[: -len(task) - 1]
    metrics = json.loads(path.read_text())
    runs[run] = {"model": model, "task": task, **metrics}
    table.setdefault(model, {})[task] = metrics["mean_ap"]

output = eval_dir / "all_metrics.json"
output.write_text(json.dumps({"mean_ap": table, "runs": runs}, indent=4))

print(f"\nJoint metrics saved to {output}")
print(f"{'mAP':<16}" + "".join(f"{t:>26}" for t in tasks))
for model, aps in sorted(table.items()):
    print(f"{model:<16}" + "".join(f"{aps[t]:>26.4f}" if t in aps else f"{'-':>26}" for t in tasks))
EOF

[[ ${#failed[@]} -eq 0 ]]
