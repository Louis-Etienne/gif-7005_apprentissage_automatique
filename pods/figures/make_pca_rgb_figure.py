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
--detector-ckpt KEY=PATH swaps in the fine-tuned backbone of a detector checkpoint.

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
from PIL import Image

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
def select_images(args):
    if args.images:
        return [Path(p) for p in args.images]
    df = pd.read_csv(Path(args.data_dir) / args.csv_file).head(2254)  # same subset as the detection runs
    df = df[df["split"] == "test"]
    if args.concept:
        df = df[df["finding_categories"].str.contains(args.concept, regex=False)]
    df = df.drop_duplicates(["study_id", "image_id"]).head(args.num_images)
    return [Path(args.img_dir) / str(r.study_id) / (r.image_id if str(r.image_id).endswith(".png") else f"{r.image_id}.png")
            for r in df.itertuples()]


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
def make_figure(images, columns, output, dpi):
    n_rows, n_cols = len(images), len(columns) + 1
    aspect = images[0].shape[-2] / images[0].shape[-1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * aspect * n_rows + 0.3), squeeze=False)
    for row, image in enumerate(images):
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
    detector_ckpts = dict(item.split("=", 1) for item in args.detector_ckpt)
    unknown = set(detector_ckpts) - {m.key for m in MODELS}
    if unknown:
        raise ValueError(f"Unknown --detector-ckpt keys: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = select_images(args)
    images = [load_image(p) for p in image_paths]
    masks = [tissue_mask(image, grid, args.tissue_threshold) for image in images]
    print(f"{len(images)} images, token grid {grid[0]}x{grid[1]}, device {device}")

    columns = []
    for backbone in MODELS:
        if backbone.key not in args.models:
            continue
        backbone = dataclasses.replace(backbone, detector_ckpt=detector_ckpts.get(backbone.key))
        stride = STRIDES[backbone.family]
        size = (grid[0] * stride, grid[1] * stride)
        # Pretrained DINO expects ImageNet statistics; the Mammo backbones and every fine-tuned
        # backbone were trained with Mammo-FM's.
        normalization = "imagenet" if backbone.family != "effnet" and not backbone.detector_ckpt else "mammo"
        print(f"\n{backbone.name}: input {size[0]}x{size[1]}, {normalization} normalization")

        net = build_backbone(backbone, args).to(device)
        features = [extract_features(net, backbone.family, preprocess(image, size, normalization).to(device))
                    for image in images]
        del net
        torch.cuda.empty_cache()
        if tuple(features[0].shape[:2]) != grid:
            raise RuntimeError(f"{backbone.name}: expected a {grid} grid, got {tuple(features[0].shape[:2])}")

        pca = fit_pca(features, masks)
        rgbs = [resize_rgb(pca_rgb(f, m, pca), tuple(image.shape[-2:]), args.upsample)
                for f, m, image in zip(features, masks, images)]

        suffix = "_finetuned" if backbone.detector_ckpt else ""
        model_dir = args.output_dir / f"{backbone.key}{suffix}"
        model_dir.mkdir(parents=True, exist_ok=True)
        for path, rgb in zip(image_paths, rgbs):
            Image.fromarray(rgb).save(model_dir / f"{path.parent.name}_{path.stem}.png")
        columns.append((backbone.name + (" (fine-tuned)" if suffix else ""), rgbs))

    make_figure(images, columns, args.output_dir / "pca_rgb_comparison.png", args.dpi)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mammo-fm-codebase", default=os.environ.get("MAMMO_FM_CODEBASE", DEFAULT_MAMMO_FM_CODEBASE))
    parser.add_argument("--models", nargs="+", choices=[m.key for m in MODELS], default=[m.key for m in MODELS])
    parser.add_argument("--detector-ckpt", nargs="*", default=[], metavar="KEY=PATH",
                        help="Use the fine-tuned backbone of a detector checkpoint, e.g. dinov3_base=/workspace/outputs/...pth")

    # Images: explicit PNGs, or the first --num-images test images showing --concept
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--data-dir", default="/workspace/data/vindr_png")
    parser.add_argument("--img-dir", default="/workspace/data/vindr_png/images_png")
    parser.add_argument("--csv-file", default="vindr_detection_v1_folds.csv")
    parser.add_argument("--concept", default="Mass", help="'' = any test image")
    parser.add_argument("--num-images", type=int, default=4)

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