conda env config vars set \
    PYTHONPATH=/workspace/Mammo-FM/src/codebase

python -u /workspace/gif-7005_apprentissage_automatique/pods/mammo_fm_pod/evaluate_detector.py \
    --dataset ViNDr \
    --data-dir /workspace/data/vindr_png \
    --img-dir /workspace/data/vindr_png/images_png \
    --csv-file vindr_detection_v1_folds.csv \
    --arch breast_clip_det_b5 \
    --concepts Mass \
    --chk-pt-path /workspace/outputs/mammo_clip_mass/checkpoints/ViNDr/Concept-Detector/breast_clip_det_b5/lr_5e-05_epochs_120_concepts_Mass_alpha_0.25_gamma_2.0_score_th_0.2_data_frac_1.0_freeze_backbone_n/breast_clip_det_b5_seed_10_fold_0_best_auroc.pth \
    --clip-chk-pt-path /workspace/checkpoints/b5-model-best-epoch-7.tar \
    --freeze-backbone n \
    --batch-size 7 \
    --num-workers 4 \
    --iou-threshold 0.5 \
    --score-threshold 0.2 \
    --max-detections 100 \
    --visualization-dir /workspace/eval/mammo_clip_mass \
    --num-examples 30

    ## Suspicious Calcification
    python -u /workspace/gif-7005_apprentissage_automatique/pods/mammo_fm_pod/evaluate_detector.py \
    --dataset ViNDr \
    --data-dir /workspace/data/vindr_png \
    --img-dir /workspace/data/vindr_png/images_png \
    --csv-file vindr_detection_v1_folds.csv \
    --arch breast_clip_det_b5 \
    --concepts "Suspicious Calcification" \
    --chk-pt-path /workspace/outputs/mammo_clip_suspicious_calcification/checkpoints/ViNDr/Concept-Detector/breast_clip_det_b5/lr_5e-05_epochs_120_concepts_Suspicious Calcification_alpha_0.25_gamma_2.0_score_th_0.2_data_frac_1.0_freeze_backbone_n/breast_clip_det_b5_seed_10_fold_0_best_auroc.pth \
    --clip-chk-pt-path /workspace/checkpoints/b5-model-best-epoch-7.tar \
    --freeze-backbone n \
    --batch-size 7 \
    --num-workers 4 \
    --iou-threshold 0.5 \
    --score-threshold 0.2 \
    --max-detections 100 \
    --visualization-dir /workspace/eval/mammo_clip_suspicious_calcification \
    --num-examples 30