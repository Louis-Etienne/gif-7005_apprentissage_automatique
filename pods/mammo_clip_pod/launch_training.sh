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
  --clip_chk_pt_path /workspace/checkpoints/b5-model-best-epoch-7.tar \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive n \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2 \
  --checkpoints /workspace/outputs/mammo_clip_mass/checkpoints \
  --output_path /workspace/outputs/mammo_clip_mass/logs \
  --tensorboard-path /workspace/outputs/mammo_clip_mass/tensorboard


  ## Suspicious Calcification
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
  --concepts "Suspicious Calcification" \
  --clip_chk_pt_path /workspace/checkpoints/b5-model-best-epoch-7.tar \
  --print-freq 5000 \
  --log-freq 300 \
  --running-interactive n \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --score-threshold 0.2 \
  --checkpoints /workspace/outputs/mammo_clip_suspicious_calcification/checkpoints \
  --output_path /workspace/outputs/mammo_clip_suspicious_calcification/logs \
  --tensorboard-path /workspace/outputs/mammo_clip_suspicious_calcification/tensorboard