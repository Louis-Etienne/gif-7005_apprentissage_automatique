#!/usr/bin/env python3
"""
Evaluate a DINOv2 + RetinaNet concept detector checkpoint written by train_detector.py.

Counterpart of pods/mammo_fm_pod/evaluate_detector.py: same ViNDr test split and preprocessing,
same Mammo-FM evaluation (Detectors.experiments.evaluate, VOC-style AP), a metrics.json and PNG
examples (ground truth in green, predictions in red) in --visualization-dir.

The detector is rebuilt with train_detector.build_dinov2_retinanet, so --arch, --dinov2-layers,
--vit-input-size and --amp must match the training run. The checkpoint holds every weight
(ViT included), so no pretrained DINOv2 weights are downloaded.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from train_detector import (DEFAULT_MAMMO_FM_CODEBASE, DINOV2_ARCHS, NullLogger, build_dinov2_retinanet,
                            import_mammo_fm)


def load_model(args, device, RetinaNet):
    print(f"Initializing {args.arch} detector...")
    # Training-only settings: they do not change the architecture or inference.
    build_args = argparse.Namespace(**vars(args), dinov2_weights=None, dinov2_pretrained="n",
                                    freeze_backbone="n", grad_checkpointing="n",
                                    focal_alpha=0.25, focal_gamma=2.0)
    model = build_dinov2_retinanet(build_args, RetinaNet)

    checkpoint = torch.load(args.chk_pt_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"Checkpoint loaded successfully: {args.chk_pt_path}")
    if "epoch" in checkpoint:
        print(f"epoch: {int(checkpoint['epoch']) + 1}")
    for key in ["MAP", "best_MAP"]:
        if key in checkpoint:
            print(f"{key}: {checkpoint[key]:.4f}")

    return model, checkpoint


def get_validation_dataset(args, get_dataset):
    args.data_dir = Path(args.data_dir)
    csv_path = args.data_dir / args.csv_file
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"Loading CSV: {csv_path}")
    args.df = pd.read_csv(csv_path).fillna(0)

    if args.dataset.lower() != "vindr":
        raise ValueError(f"Only ViNDr is supported, got {args.dataset}")
    # Same split as train_detector.py, whose validation set is the ViNDr test split.
    args.df = args.df.head(2254)
    args.train_folds = args.df[args.df["split"] == "training"].reset_index(drop=True)
    args.valid_folds = args.df[args.df["split"] == "test"].reset_index(drop=True)

    # Required by Mammo-FM dataset utilities.
    args.model_type = "Concept-Detector"
    args.cur_fold = 0
    # Only used by the training augmentations, which get_dataset builds even with train=False.
    args.alpha = 10
    args.sigma = 15

    valid_loader, valid_dataset = get_dataset(args, train=False)
    print(f"Validation images: {len(valid_dataset)}")
    print(f"Validation batches: {len(valid_loader)}")
    return valid_dataset


def save_prediction_examples(model, dataset, args, device):
    output_dir = Path(args.visualization_dir)
    print(f"\nSaving PNG examples to: {output_dir}")

    saved = 0
    with torch.no_grad():
        for index in range(min(len(dataset), args.num_examples)):
            sample = dataset[index]
            image_tensor = sample["image"]

            # Undo the mean/std normalisation to get a displayable grayscale image.
            image = image_tensor.cpu().numpy()[0] * args.std + args.mean
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")
            draw = ImageDraw.Draw(image)

            # Ground truth boxes: green
            for x1, y1, x2, y2, label in sample["target"]["boxes"].cpu().numpy():
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
                draw.text((x1, max(0, y1 - 15)), f"GT: {args.concepts[int(label)]}", fill="lime")

            # Model predictions: red
            scores, labels, boxes = model(image_tensor.unsqueeze(0).to(device))
            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            boxes = boxes.detach().cpu().numpy()

            valid_indices = np.where(scores > args.score_threshold)[0]
            valid_indices = valid_indices[np.argsort(-scores[valid_indices])][:args.max_detections]
            for idx in valid_indices:
                x1, y1, x2, y2 = boxes[idx]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0, y1 - 15)), f"{args.concepts[int(labels[idx])]}: {scores[idx]:.3f}",
                          fill="red")

            filename = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{sample['study_id']}_{sample['image_id']}")
            output_path = output_dir / f"{index:05d}_{filename}.png"
            image.save(output_path)
            saved += 1
            print(f"Saved: {output_path}")

    print(f"\nSaved {saved} visualization examples.")


def evaluate_checkpoint(args):
    mfm = import_mammo_fm(args.mammo_fm_codebase)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.amp == "y" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("[warn] GPU has no bf16 support, disabling --amp")
        args.amp = "n"

    output_dir = Path(args.visualization_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset
    valid_dataset = get_validation_dataset(args, mfm["get_dataset"])

    # 2. Model
    model, checkpoint = load_model(args, device, mfm["RetinaNet"])

    # 3. Evaluate
    print("\nStarting detection evaluation...")
    average_precisions, mean_ap = mfm["evaluate"](
        val_dataset=valid_dataset,
        concepts=args.concepts,
        logger=NullLogger(),
        epoch_num=0,
        retinanet=model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )

    metrics = {
        "dataset": args.dataset,
        "architecture": args.arch,
        "checkpoint": args.chk_pt_path,
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None,
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "mean_ap": float(mean_ap),
        "average_precisions": {
            args.concepts[label]: {"AP": float(ap), "annotations": int(num_annotations)}
            for label, (ap, num_annotations) in average_precisions.items()
        },
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    # 4. Results
    print("\n" + "=" * 60)
    print(f"{args.arch.upper() + ' EVALUATION':^60}")
    print("=" * 60)
    for label, (ap, num_annotations) in average_precisions.items():
        print(f"{args.concepts[label]:<30}AP: {ap:.4f} Annotations: {int(num_annotations)}")
    print("-" * 60)
    print(f"mAP@{args.iou_threshold}: {mean_ap:.4f}")
    print("=" * 60)

    # 5. Export visual examples
    save_prediction_examples(model=model, dataset=valid_dataset, args=args, device=device)

    return average_precisions, mean_ap


def config():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Dataset
    parser.add_argument("--mammo-fm-codebase", default=os.environ.get("MAMMO_FM_CODEBASE", DEFAULT_MAMMO_FM_CODEBASE),
                        type=str, help="Path to Mammo-FM/src/codebase (also settable with $MAMMO_FM_CODEBASE)")
    parser.add_argument("--dataset", default="ViNDr", type=str)
    parser.add_argument("--data-dir", default="/workspace/data/vindr_png", type=str)
    parser.add_argument("--img-dir", default="/workspace/data/vindr_png/images_png", type=str)
    parser.add_argument("--csv-file", default="vindr_detection_v1_folds.csv", type=str)

    # Model: must match the training run
    parser.add_argument("--arch", default="dinov2_vitb14", type=str, choices=sorted(DINOV2_ARCHS))
    parser.add_argument("--concepts", nargs="+", default=["Mass"])
    parser.add_argument("--dinov2-layers", default=None, type=str,
                        help="3 comma-separated block indices for C3,C4,C5 (default: 50%%,75%%,100%% depth)")
    parser.add_argument("--vit-input-size", default=0, type=int,
                        help="Side fed to the ViT (multiple of 14). 0 = ceil(--resize/14)*14 (518 for 512)")
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--amp", default="n", type=str, choices=["y", "n"],
                        help="bf16 autocast for the ViT only")
    parser.add_argument("--dinov2-hub-dir", default=None, type=str,
                        help="Local clone of facebookresearch/dinov2 (torch.hub source='local'); default: GitHub")
    parser.add_argument("--dinov2-cache-dir", default="/workspace/checkpoints", type=str,
                        help="Directory holding the torch.hub code cached by train_detector.py")

    # Checkpoint
    parser.add_argument("--chk-pt-path", required=True, type=str,
                        help="Detector checkpoint, e.g. <run>/dinov2_vitb14_seed_10_fold_0_best_auroc.pth")

    # Normalization: same as train_detector.py defaults
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)

    # Dataloader
    parser.add_argument("--batch-size", default=7, type=int)
    parser.add_argument("--num-workers", default=4, type=int)

    # Evaluation
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--score-threshold", default=0.2, type=float)
    parser.add_argument("--max-detections", default=100, type=int)

    # Visualization
    parser.add_argument("--visualization-dir", default="/workspace/eval/dinov2", type=str)
    parser.add_argument("--num-examples", default=30, type=int)

    args = parser.parse_args()
    if not Path(args.chk_pt_path).is_file():
        raise FileNotFoundError(f"--chk-pt-path not found: {args.chk_pt_path}")
    if args.resize % 32 != 0:
        raise ValueError(f"--resize must be a multiple of 32 so C3/C4/C5 match the FPN, got {args.resize}")
    if args.dinov2_hub_dir and not Path(args.dinov2_hub_dir, "hubconf.py").is_file():
        raise FileNotFoundError(f"--dinov2-hub-dir has no hubconf.py: {args.dinov2_hub_dir}")
    return args


if __name__ == "__main__":
    evaluate_checkpoint(config())
