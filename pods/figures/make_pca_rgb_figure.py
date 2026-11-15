#!/usr/bin/env python3
"""
Dense-feature comparison figure: project each backbone's last feature map to RGB with PCA.

Same protocol as the DINOv3 paper figure: every model is forwarded at the input size that gives the
same token grid (--grid), so all feature maps have the same resolution. The PCA is fitted per model on
the breast-tissue tokens pooled over all selected images (consistent colours across images), and the
RGB maps are resized back to each image's original resolution.

End feature maps:
  Mammo-FM / Mammo-CLIP  EfficientNet-B5 C5 (stride 32, 2048 ch), pretrained CLIP image encoder
  DINOv2                 last ViT block (patch 14), pretrained hub weights
  DINOv3                 last ViT block (patch 16), pretrained weights
--detector-ckpt KEY=PATH swaps in the fine-tuned backbone of a detector checkpoint; KEY@CONCEPT=PATH
only for the rows of that concept (one fine-tuned backbone per concept in a single figure, the PCA
being fitted per checkpoint; rows of a concept without a checkpoint are left black).

Run in the DINOv3 env (Python >= 3.10, pods/dinov3_pod/setup_env.sh) with the Mammo-FM codebase.
"""

import argparse
import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

FIGURES_DIR = Path(__file__).resolve().parent
PODS_DIR = FIGURES_DIR.parent
DEFAULT_EVAL_DIR = FIGURES_DIR.parents[2] / "eval"
CHECKPOINTS_DIR = Path("/workspace/checkpoints")
DEFAULT_MAMMO_FM_CODEBASE = "/workspace/Mammo-FM/src/codebase"

MAMMO_MEAN, MAMMO_STD = 0.3089279, 0.25053555408335154
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
STRIDES = {"effnet": 32, "dinov2": 14, "dinov3": 16}


@dataclasses.dataclass(frozen=True)
class Backbone:
    key: str
    name: str
    family: str                       # "effnet", "dinov2" or "dinov3"
    source: str                       # CLIP checkpoint (effnet) or torch.hub arch (DINO)
    detector_ckpt: str | None = None  # fine-tuned backbone from a detector checkpoint


MODELS = [
    Backbone("mammo_fm", "MammoFM", "effnet", str(CHECKPOINTS_DIR / "Mammo-FM_BatmanlabTrained_CLIP.tar")),
    Backbone("mammo_clip", "Mammo-CLIP", "effnet", str(CHECKPOINTS_DIR / "b5-model-best-epoch-7.tar")),
    Backbone("dinov2_small", "DINOv2 Small", "dinov2", "dinov2_vits14"),
    Backbone("dinov2_base", "DINOv2 Base", "dinov2", "dinov2_vitb14"),
    Backbone("dinov3_small", "DINOv3 Small", "dinov3", "dinov3_vits16"),
    Backbone("dinov3_base", "DINOv3 Base", "dinov3", "dinov3_vitb16"),
]


# ----------------------------------------------------------------------------
# Backbones
# ----------------------------------------------------------------------------
def load_pod_module(pod):
    """Import pods/<pod>/train_detector.py under a unique name (both DINO pods use that file name)."""
    spec = importlib.util.spec_from_file_location(f"{pod}_train_detector", PODS_DIR / pod / "train_detector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_backbone(backbone, args):
    pretrained = backbone.detector_ckpt is None
    if backbone.family == "effnet":
        if args.mammo_fm_codebase not in sys.path:
            sys.path.insert(0, args.mammo_fm_codebase)
        from Detectors.retinanet.efficient_net import EfficientNet
        net = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1, model_type="breast_clip_det_b5",
                                           clip_chk_pt=backbone.source)
        net.source_layer_indexes = [26, 37]  # as in RetinaNet_efficientnet
    elif backbone.family == "dinov2":
        net = load_pod_module("dinov2_pod").DINOv2Backbone(
            arch=backbone.source, pretrained=pretrained, cache_dir=str(CHECKPOINTS_DIR)).vit
    else:
        net = load_pod_module("dinov3_pod").DINOv3Backbone(
            arch=backbone.source, pretrained=pretrained, cache_dir=str(CHECKPOINTS_DIR)).vit

    if backbone.detector_ckpt:
        # RetinaNet keeps its backbone under "efficientnet."; the DINO ViT sits under "efficientnet.vit."
        prefix = "efficientnet." if backbone.family == "effnet" else "efficientnet.vit."
        checkpoint = torch.load(backbone.detector_ckpt, map_location="cpu")
        state = {k.removeprefix("module."): v for k, v in checkpoint.get("state_dict", checkpoint).items()}
        net.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        print(f"{backbone.name}: fine-tuned backbone loaded from {backbone.detector_ckpt}")
    return net.eval()


@torch.no_grad()
def extract_features(net, family, x):
    """Last feature map as (h, w, C)."""
    if family == "effnet":
        feature = net(x)[3]  # (x, C3, C4, C5)
    else:
        feature = net.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
    return feature[0].permute(1, 2, 0).float().cpu()


# ----------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------
def gt_boxes(findings, image_id, concepts):
    """Ground-truth boxes (x1, y1, x2, y2, concept) of an image for the requested concepts.

    The resized_* columns are in the pixel space of the stored PNGs (the raw x/y columns are not).
    """
    boxes = []
    for r in findings[findings["image_id"] == image_id].itertuples():
        concept = next((c for c in concepts if c in str(r.finding_categories)), None)
        if concept is not None and r.resized_xmax > r.resized_xmin:
            boxes.append((r.resized_xmin, r.resized_ymin, r.resized_xmax, r.resized_ymax, concept))
    return boxes


def select_images(args):
    """(path, row label, GT boxes): --num-images test images per concept, no image used twice."""
    if args.images:
        return [(Path(p), "", []) for p in args.images]
    df = pd.read_csv(Path(args.data_dir) / args.csv_file, low_memory=False).head(2254)  # same subset as the detection runs
    findings = df[df["split"] == "test"]
    images = findings.drop_duplicates(["patient_id", "image_id"])
    box_concepts = [c for c in args.concepts if c]
    selected, used = [], set()
    for concept in args.concepts or [""]:
        rows = images[images["finding_categories"].str.contains(concept, regex=False)] if concept else images
        rows = rows[~rows["image_id"].isin(used)].head(args.num_images)
        if len(rows) < args.num_images:
            print(f"[warn] only {len(rows)} test images found for concept '{concept}'")
        used.update(rows["image_id"])
        selected += [(Path(args.img_dir) / str(r.patient_id) /
                      (r.image_id if str(r.image_id).endswith(".png") else f"{r.image_id}.png"), concept,
                      gt_boxes(findings, r.image_id, box_concepts))
                     for r in rows.itertuples()]
    if not selected:
        raise ValueError(f"No test images found for concepts {args.concepts}")
    return selected


def load_image(path):
    """Grayscale PNG -> (1, 1, H, W) tensor min-max scaled to [0, 1]."""
    image = torch.from_numpy(np.asarray(Image.open(path).convert("L"), dtype=np.float32))
    image = image - image.min()
    return (image / image.max().clamp_min(1e-6))[None, None]


def preprocess(image, size, normalization):
    x = F.interpolate(image, size=size, mode="bilinear", align_corners=False, antialias=True).repeat(1, 3, 1, 1)
    if normalization == "mammo":
        return (x - MAMMO_MEAN) / MAMMO_STD
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def tissue_mask(image, grid, threshold):
    """True where the grid cell covers breast tissue (mammogram background is black)."""
    return F.adaptive_avg_pool2d(image, grid)[0, 0] > threshold


# ----------------------------------------------------------------------------
# PCA -> RGB
# ----------------------------------------------------------------------------
def fit_pca(features, masks):
    """PCA on the tissue tokens of all images: mean, (C, 3) components and 1-99% range per component."""
    tokens = torch.cat([f[m] for f, m in zip(features, masks)])
    mean = tokens.mean(0)
    centered = tokens - mean
    _, eigenvectors = torch.linalg.eigh(centered.T @ centered)
    components = eigenvectors[:, -3:].flip(1)  # eigh sorts eigenvalues in ascending order
    projected = centered @ components
    low, high = torch.quantile(projected, 0.01, dim=0), torch.quantile(projected, 0.99, dim=0)
    return mean, components, low, high


def pca_rgb(feature, mask, pca):
    mean, components, low, high = pca
    rgb = (((feature - mean) @ components - low) / (high - low)).clamp(0, 1)
    rgb[~mask] = 0
    return rgb


def resize_rgb(rgb, size, mode):
    """(h, w, 3) grid -> (H, W, 3) uint8 image at the original resolution."""
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    up = F.interpolate(rgb.permute(2, 0, 1)[None], size=size, mode=mode, **kwargs)[0]
    return (up.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def concept_colors(concepts):
    return {c: tuple(int(255 * v) for v in plt.cm.tab10(i)[:3]) for i, c in enumerate(concepts)}


def draw_boxes(image, boxes, colors, width=4):
    """(1, 1, H, W) image in [0, 1] -> (H, W, 3) uint8 with the GT boxes outlined."""
    rgb = Image.fromarray((image[0, 0].numpy() * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for x1, y1, x2, y2, concept in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=colors[concept], width=width)
    return np.asarray(rgb)


def make_figure(images, row_labels, columns, output, dpi):
    n_rows, n_cols = len(images), len(columns) + 1
    aspect = images[0].shape[-2] / images[0].shape[-1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * aspect * n_rows + 0.3), squeeze=False)
    for row, image in enumerate(images):
        if row_labels[row]:
            axes[row, 0].set_ylabel(row_labels[row], fontsize=9, fontweight="bold")
        panels = [("Input", image[0, 0].numpy())] + [(name, rgbs[row]) for name, rgbs in columns]
        for col, (name, panel) in enumerate(panels):
            ax = axes[row, col]
            if panel.ndim == 2:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(name, fontsize=10, fontweight="bold")
    fig.tight_layout(w_pad=0.3, h_pad=0.3)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = tuple(args.grid)
    detector_ckpts = {}
    for item in args.detector_ckpt:
        target, path = item.split("=", 1)
        key, _, concept = target.partition("@")
        detector_ckpts.setdefault(key, {})[concept or None] = path
    unknown = set(detector_ckpts) - {m.key for m in MODELS}
    if unknown:
        raise ValueError(f"Unknown --detector-ckpt keys: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, row_labels, boxes = zip(*select_images(args))
    images = [load_image(p) for p in image_paths]
    masks = [tissue_mask(image, grid, args.tissue_threshold) for image in images]
    print(f"{len(images)} images, token grid {grid[0]}x{grid[1]}, device {device}")

    colors = concept_colors([c for c in args.concepts if c])
    gt_images = [draw_boxes(image, b, colors) for image, b in zip(images, boxes)]
    gt_dir = args.output_dir / "input_gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    for path, gt_image in zip(image_paths, gt_images):
        Image.fromarray(gt_image).save(gt_dir / f"{path.parent.name}_{path.stem}.png")
    columns = [("Input + GT", gt_images)]
    for backbone in MODELS:
        if backbone.key not in args.models:
            continue
        ckpts = detector_ckpts.get(backbone.key, {})
        per_concept = any(concept is not None for concept in ckpts)
        # Rows sharing a checkpoint go through the same network and PCA; None = pretrained weights.
        groups = {}
        for row, concept in enumerate(row_labels):
            ckpt = ckpts.get(concept, ckpts.get(None))
            if ckpt is None and per_concept:
                print(f"[skip] {backbone.name}: no fine-tuned checkpoint for '{concept}', row left black")
                continue
            groups.setdefault(ckpt, []).append(row)

        stride = STRIDES[backbone.family]
        size = (grid[0] * stride, grid[1] * stride)
        rgbs = [np.zeros((*image.shape[-2:], 3), dtype=np.uint8) for image in images]
        for ckpt, rows in groups.items():
            group_backbone = dataclasses.replace(backbone, detector_ckpt=ckpt)
            # Pretrained DINO expects ImageNet statistics; the Mammo backbones and every fine-tuned
            # backbone were trained with Mammo-FM's.
            normalization = "imagenet" if backbone.family != "effnet" and not ckpt else "mammo"
            print(f"\n{backbone.name}: input {size[0]}x{size[1]}, {normalization} normalization, {len(rows)} images")

            net = build_backbone(group_backbone, args).to(device)
            features = [extract_features(net, backbone.family, preprocess(images[r], size, normalization).to(device))
                        for r in rows]
            del net
            torch.cuda.empty_cache()
            if tuple(features[0].shape[:2]) != grid:
                raise RuntimeError(f"{backbone.name}: expected a {grid} grid, got {tuple(features[0].shape[:2])}")

            pca = fit_pca(features, [masks[r] for r in rows])
            for r, f in zip(rows, features):
                rgbs[r] = resize_rgb(pca_rgb(f, masks[r], pca), tuple(images[r].shape[-2:]), args.upsample)

        finetuned = any(ckpt is not None for ckpt in groups)
        model_dir = args.output_dir / (f"{backbone.key}_finetuned" if finetuned else backbone.key)
        model_dir.mkdir(parents=True, exist_ok=True)
        for path, rgb in zip(image_paths, rgbs):
            Image.fromarray(rgb).save(model_dir / f"{path.parent.name}_{path.stem}.png")
        columns.append((f"{backbone.name}\n(FT)" if finetuned else backbone.name, rgbs))

    make_figure(images, row_labels, columns, args.output_dir / "pca_rgb_comparison.png", args.dpi)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mammo-fm-codebase", default=os.environ.get("MAMMO_FM_CODEBASE", DEFAULT_MAMMO_FM_CODEBASE))
    parser.add_argument("--models", nargs="+", choices=[m.key for m in MODELS], default=[m.key for m in MODELS])
    parser.add_argument("--detector-ckpt", nargs="*", default=[], metavar="KEY[@CONCEPT]=PATH",
                        help="Use the fine-tuned backbone of a detector checkpoint, e.g. dinov3_base=/workspace/outputs_finetune/...pth, "
                             "or only for one concept's rows, e.g. \"dinov3_base@Suspicious Calcification=...pth\"")

    # Images: explicit PNGs, or the first --num-images test images showing each of --concepts
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--data-dir", default="/workspace/data/vindr_png")
    parser.add_argument("--img-dir", default="/workspace/data/vindr_png/images_png")
    parser.add_argument("--csv-file", default="vindr_detection_v1_folds.csv")
    parser.add_argument("--concepts", "--concept", nargs="*", default=["Mass"],
                        help="One or more ViNDr finding categories, e.g. Mass \"Suspicious Calcification\"; "
                             "none = any test image")
    parser.add_argument("--num-images", type=int, default=4, help="Images per concept")

    # PCA / rendering
    parser.add_argument("--grid", nargs=2, type=int, default=[48, 28], metavar=("H", "W"),
                        help="Token grid shared by all models (input = grid x 32 EfficientNet, x16 DINOv3, x14 DINOv2)")
    parser.add_argument("--tissue-threshold", type=float, default=0.05,
                        help="Grid cells whose mean intensity in [0, 1] is below this are background")
    parser.add_argument("--upsample", choices=["nearest-exact", "bilinear"], default="nearest-exact")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_DIR / "pca_rgb")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    main()