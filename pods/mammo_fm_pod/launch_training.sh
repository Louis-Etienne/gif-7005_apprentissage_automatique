nvidia-smi

python /workspace/Mammo-FM/src/codebase/train_detector.py \
  --data-dir /workspace/data/vindr_png \
  --img-dir /workspace/data/vindr_png/images_png \
  --csv-file /workspace/data/vindr_png/vindr_detection_v1_folds.csv \
  --dataset ViNDr \
  --arch breast_clip_det_b5 \
  --epochs 120 \
  --batch-size 7 \
  --freeze_backbone n \
  --data_frac 1.0 \
  --concepts Mass \
  --clip_chk_pt_path /workspace/checkpoints/Mammo-FM_BatmanlabTrained_CLIP.tar \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive n \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2 \
  --checkpoints /workspace/outputs/checkpoints \
  --output_path /workspace/outputs/logs \
  --tensorboard-path /workspace/outputs/tensorboard

  python /workspace/Mammo-FM/src/codebase/train_detector.py --data-dir /workspace/data/vindr_png --img-dir /workspace/data/vindr_png/images_png --csv-file /workspace/data/vindr_png/vindr_detection_v1_folds.csv --dataset ViNDr --arch breast_clip_det_b5 --epochs 120 --batch-size 7 --freeze_backbone n --data_frac 1.0 --concepts Mass --clip_chk_pt_path /workspace/checkpoints/Mammo-FM_BatmanlabTrained_CLIP.tar --print-freq 5000 --log-freq 300 --running-interactive n --focal-alpha 0.25 --focal-gamma 2.0 --score-threshold 0.2 --checkpoints /workspace/outputs/checkpoints --output_path /workspace/outputs/logs --tensorboard-path /workspace/outputs/tensorboard
