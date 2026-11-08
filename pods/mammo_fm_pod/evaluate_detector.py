import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import json

from PIL import Image, ImageDraw, ImageFont

from Datasets.dataset_utils import get_dataset
from Detectors.retinanet.detector_model import RetinaNet_efficientnet
from Detectors.experiments import evaluate


class NullLogger:
    def add_scalar(self, *args, **kwargs):
        pass


def load_model(args, device):

    print("Initializing Mammo-FM detector...")

    model = RetinaNet_efficientnet(
        num_classes=len(args.concepts),
        model_type=args.arch,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        clip_chk_pt=args.clip_chk_pt_path,
        freeze_backbone=args.freeze_backbone,
    )

    checkpoint = torch.load(
        args.chk_pt_path,
        map_location=device
    )

    state_dict = checkpoint.get("state_dict", checkpoint)

    state_dict = {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print("Checkpoint loaded successfully.")

    for key in ["MAP", "best_MAP"]:
        if key in checkpoint:
            print(f"{key}: {checkpoint[key]:.4f}")

    return model


def get_validation_dataset(args):

    args.data_dir = Path(args.data_dir)

    csv_path = args.data_dir / args.csv_file

    print(f"Loading CSV: {csv_path}")

    args.df = pd.read_csv(csv_path).fillna(0)

    if args.dataset.lower() == "vindr":

        # Match Mammo-FM training implementation.
        args.df = args.df.head(2254)

        args.train_folds = (
            args.df[args.df["split"] == "training"]
            .reset_index(drop=True)
        )

        args.valid_folds = (
            args.df[args.df["split"] == "test"]
            .reset_index(drop=True)
        )

    # Required by Mammo-FM dataset utilities.
    args.model_type = "Concept-Detector"
    args.cur_fold = 0

    # IMPORTANT:
    # Preserve the training preprocessing.
    # train_detector.py defaults to resize=512.
    # This is not an additional evaluation resize.
    args.resize = 512

    args.alpha = 50
    args.sigma = 5
    
    valid_loader, valid_dataset = get_dataset(
        args,
        train=False
    )

    print(f"Validation images: {len(valid_dataset)}")
    print(f"Validation batches: {len(valid_loader)}")

    return valid_dataset


def save_prediction_examples(
    model,
    dataset,
    args,
    device
):

    output_dir = Path(args.visualization_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving PNG examples to: {output_dir}")

    saved = 0

    with torch.no_grad():

        for index in range(min(len(dataset), args.num_examples)):

            sample = dataset[index]

            image_tensor = sample["image"]

            # Recover normalized image to displayable grayscale.
            image = (
                image_tensor.cpu().numpy()[0]
                * args.std
                + args.mean
            )

            image = np.clip(image * 255, 0, 255).astype(np.uint8)

            # Convert to RGB for drawing.
            image = Image.fromarray(image).convert("RGB")

            draw = ImageDraw.Draw(image)

            # -----------------------------------------------
            # Ground truth boxes: GREEN
            # -----------------------------------------------

            gt_boxes = sample["target"]["boxes"].cpu().numpy()

            for box in gt_boxes:

                x1, y1, x2, y2, label = box

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="lime",
                    width=2
                )

                draw.text(
                    (x1, max(0, y1 - 15)),
                    f"GT: {args.concepts[int(label)]}",
                    fill="lime"
                )

            # -----------------------------------------------
            # Model predictions: RED
            # -----------------------------------------------

            input_tensor = image_tensor.unsqueeze(0).to(device)

            scores, labels, boxes = model(input_tensor)

            scores = scores[0].cpu().numpy()
            labels = labels[0].cpu().numpy()
            boxes = boxes[0].cpu().numpy()

            valid_indices = np.where(
                scores > args.score_threshold
            )[0]

            valid_indices = valid_indices[
                np.argsort(-scores[valid_indices])
            ][:args.max_detections]

            for idx in valid_indices:

                x1, y1, x2, y2 = boxes[idx]

                score = scores[idx]
                label = int(labels[idx])

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3
                )

                draw.text(
                    (x1, max(0, y1 - 15)),
                    f"{args.concepts[label]}: {score:.3f}",
                    fill="red"
                )

            # -----------------------------------------------
            # Save PNG
            # -----------------------------------------------

            study_id = sample["study_id"]
            image_id = sample["image_id"]

            filename = re.sub(
                r"[^a-zA-Z0-9_-]",
                "_",
                f"{study_id}_{image_id}"
            )

            output_path = output_dir / f"{index:05d}_{filename}.png"

            image.save(output_path)

            saved += 1

            print(f"Saved: {output_path}")

    print(f"\nSaved {saved} visualization examples.")


def evaluate_checkpoint(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # 1. Dataset
    valid_dataset = get_validation_dataset(args)

    # 2. Model
    model = load_model(args, device)

    # 3. Evaluate
    print("\nStarting detection evaluation...")

    average_precisions, mean_ap = evaluate(
        val_dataset=valid_dataset,
        concepts=args.concepts,
        logger=NullLogger(),
        epoch_num=0,
        retinanet=model,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )

    import json

    # Save evaluation metrics
    metrics = {
        "dataset": args.dataset,
        "architecture": args.arch,
        "checkpoint": args.chk_pt_path,
        "iou_threshold": args.iou_threshold,
        "mean_ap": float(mean_ap),
        "average_precisions": {
            args.concepts[label]: {
                "AP": float(ap),
                "annotations": int(num_annotations)
            }
            for label, (ap, num_annotations) in average_precisions.items()
        }
    }

    metrics_path = Path(args.visualization_dir) / "metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to: {metrics_path}")

    # 4. Results
    print("\n" + "=" * 60)
    print("             MAMMO-FM EVALUATION")
    print("=" * 60)

    for label, (ap, num_annotations) in average_precisions.items():

        print(
            f"{args.concepts[label]:<30}"
            f"AP: {ap:.4f} "
            f"Annotations: {int(num_annotations)}"
        )

    print("-" * 60)
    print(f"mAP@{args.iou_threshold}: {mean_ap:.4f}")
    print("=" * 60)

    # 5. Export visual examples
    save_prediction_examples(
        model=model,
        dataset=valid_dataset,
        args=args,
        device=device
    )

    return average_precisions, mean_ap


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="ViNDr"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/data/vindr_png"
    )

    parser.add_argument(
        "--img-dir",
        type=str,
        default="/workspace/data/vindr_png/images_png"
    )

    parser.add_argument(
        "--csv-file",
        type=str,
        default="vindr_detection_v1_folds.csv"
    )

    # Model
    parser.add_argument(
        "--arch",
        type=str,
        default="breast_clip_det_b5"
    )

    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["Mass"]
    )

    # Checkpoints
    parser.add_argument(
        "--chk-pt-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--clip-chk-pt-path",
        type=str,
        default="/workspace/checkpoints/Mammo-FM_BatmanlabTrained_CLIP.tar"
    )

    parser.add_argument(
        "--freeze-backbone",
        type=str,
        default="n"
    )

    # Detector parameters
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0
    )

    # Normalization: same as train_detector.py defaults
    parser.add_argument("--mean", type=float, default=0.3089279)
    parser.add_argument("--std", type=float, default=0.25053555408335154)

    # Dataloader
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)

    # Evaluation
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-detections", type=int, default=100)

    # Visualization
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default="/workspace/outputs/evaluation_examples"
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=30
    )

    args = parser.parse_args()

    evaluate_checkpoint(args)
