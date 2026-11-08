#!/usr/bin/env python3
"""
DINOv2 + RetinaNet concept detector (e.g. "Mass" on ViNDr-Mammo), trained with
the exact Mammo-FM detection recipe.

Mirror of Mammo-FM/src/codebase/train_detector.py + Detectors/experiments.py
(https://github.com/batmanlab/Mammo-FM). Only the image encoder is swapped:

    Mammo-FM : EfficientNet-B5 (Mammo-FM CLIP weights) -> C3, C4, C5 -> FPN -> RetinaNet
    here     : DINOv2 ViT/14                           -> C3, C4, C5 -> FPN -> RetinaNet

Everything else is imported *unchanged* from the Mammo-FM codebase so the
comparison between backbones is controlled:
  - data pipeline     Datasets.dataset_utils.get_dataset (same CSV, head(2254),
                      training/test split, imgaug augmentations, resize=512,
                      min-max + mean/std normalisation, collater)
  - detector          Detectors.retinanet.detector_model.RetinaNet
                      (PyramidFeatures P3-P7, cls/reg heads, anchors, focal loss,
                      box decoding, NMS)
  - evaluation        Detectors.experiments.evaluate (VOC-style AP @ IoU 0.5)
  - optimisation      Adam, constant lr, grad-clip 0.1, same logging / checkpoint
                      names and output directory layout (get_Paths).

DINOv2 -> C3/C4/C5 adapter (ViTDet-style "simple feature pyramid"):
  The 512x512 input is resized to a multiple of 14 (518 by default), three
  intermediate ViT blocks are taken (default: 50%, 75%, 100% depth, with the
  final LayerNorm), each token map is resampled to the exact stride-8/16/32
  grid of the 512 image (64/32/16 for 512) and projected with
  1x1 conv + GN + GELU + 3x3 conv + GN to (256, 512, 1024) channels.
  Boxes, anchors and FPN therefore live in the same 512x512 frame as Mammo-FM.

Requirements: the Mammo-FM conda env (imgaug, albumentations, tensorboard, ...)
and the Mammo-FM repo (pass --mammo-fm-codebase or put it on PYTHONPATH).
DINOv2 is loaded via torch.hub (facebookresearch/dinov2); use --dinov2-hub-dir
and/or --dinov2-weights for offline pods.
"""

import argparse
import collections
import gc
import math
import os
import pickle
import sys
import time
import traceback
import warnings
from contextlib import nullcontext
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

# imgaug 0.4.0 uses aliases removed from recent numpy; harmless on numpy 1.26.
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]
if not hasattr(np, "sctypes"):
    np.sctypes = {  # type: ignore[attr-defined]
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [bool, object, bytes, str, np.void],
    }

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.checkpoint import checkpoint

DEFAULT_MAMMO_FM_CODEBASE = "/workspace/Mammo-FM/src/codebase"


# ----------------------------------------------------------------------------
# Mammo-FM imports (resolved at runtime so the codebase path is configurable)
# ----------------------------------------------------------------------------
def import_mammo_fm(codebase):
    """Put the Mammo-FM codebase on sys.path and import the pieces we reuse."""
    if codebase:
        codebase = str(Path(codebase).expanduser().resolve())
        if not Path(codebase, "Detectors", "experiments.py").is_file():
            raise FileNotFoundError(
                f"--mammo-fm-codebase '{codebase}' does not look like Mammo-FM/src/codebase "
                "(missing Detectors/experiments.py)."
            )
        if codebase not in sys.path:
            sys.path.insert(0, codebase)
    try:
        from Datasets.dataset_utils import get_dataset
        from Detectors.experiments import evaluate
        from Detectors.retinanet.detector_model import RetinaNet
        from utils import get_Paths, seed_all, timeSince
    except ImportError as exc:
        raise ImportError(
            "Could not import the Mammo-FM codebase. Pass --mammo-fm-codebase "
            "/path/to/Mammo-FM/src/codebase (or set PYTHONPATH) and use the Mammo-FM "
            f"environment (imgaug, albumentations, tensorboard...). Original error: {exc}"
        ) from exc
    return dict(get_dataset=get_dataset, evaluate=evaluate, RetinaNet=RetinaNet,
                get_Paths=get_Paths, seed_all=seed_all, timeSince=timeSince)


# ----------------------------------------------------------------------------
# DINOv2 backbone producing (x, C3, C4, C5) like Mammo-FM's EfficientNet
# ----------------------------------------------------------------------------
DINOV2_ARCHS = {
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
    "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg",
}


def _adapter(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(32, out_ch),
    )


def _resample(feature, size):
    """Exact-size resampling of a token map: area pooling down, bilinear up."""
    if tuple(feature.shape[-2:]) == tuple(size):
        return feature
    if feature.shape[-2] >= size[0] and feature.shape[-1] >= size[1]:
        return F.adaptive_avg_pool2d(feature, size)
    return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)


def _extract_state_dict(obj):
    """Accept raw state dicts or common checkpoint wrappers (DINOv2 train ckpts...)."""
    if isinstance(obj, dict):
        for key in ("teacher", "model", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return _extract_state_dict(obj[key])
    state = {}
    for k, v in obj.items():
        for prefix in ("module.", "backbone.", "encoder."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        state[k] = v
    return state


# facebookresearch/dinov2 main switched to PEP 604 annotations (`float | None`) in June 2025,
# which breaks on Python < 3.10. Pin the last commit before that change (#528).
DINOV2_HUB_REPO = "facebookresearch/dinov2:b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8"


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT/14 encoder + simple feature pyramid -> (C5, C3, C4, C5)."""

    strides = (8, 16, 32)

    def __init__(self, arch="dinov2_vitb14", hub_repo=DINOV2_HUB_REPO, hub_source="github",
                 weights_path=None, pretrained=True, layers=None, out_channels=(256, 512, 1024),
                 vit_input_size=0, freeze_backbone="n", grad_checkpointing=False, amp=False,
                 cache_dir=None):
        super().__init__()
        if arch not in DINOV2_ARCHS:
            raise ValueError(f"--arch must be one of {sorted(DINOV2_ARCHS)}, got '{arch}'")

        # Persist the pretrained DINOv2 weights (and the hub code) in cache_dir so later
        # runs/pods reuse them instead of downloading again.
        cached_weights = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            if hub_source == "github":
                torch.hub.set_dir(str(cache_dir / "torch_hub"))
            cached_weights = cache_dir / f"{arch}_pretrain.pth"
            if pretrained and not weights_path and cached_weights.is_file():
                print(f"Using cached DINOv2 weights: {cached_weights}")
                weights_path = str(cached_weights)

        load_hub_weights = pretrained and not weights_path
        print(f"Loading {arch} from torch.hub ({hub_source}: {hub_repo}), "
              f"hub pretrained weights: {load_hub_weights}")
        # skip_validation: torch.hub's fork check only accepts branch/tag heads, so it rejects
        # a pinned historical commit SHA (even one that exists upstream).
        self.vit = torch.hub.load(hub_repo, arch, source=hub_source, pretrained=load_hub_weights,
                                  trust_repo=True,
                                  **({"skip_validation": True} if hub_source == "github" else {}))
        if load_hub_weights and cached_weights is not None:
            try:
                atomic_save(self.vit.state_dict(), cached_weights)
                print(f"Saved pretrained DINOv2 weights to {cached_weights}")
            except OSError as exc:
                print(f"[warn] could not save DINOv2 weights to {cached_weights}: {exc}")
        if weights_path:
            raw = torch.load(weights_path, map_location="cpu")
            state = _extract_state_dict(raw)
            missing, unexpected = self.vit.load_state_dict(state, strict=False)
            missing = [k for k in missing if k != "mask_token"]
            if missing:
                raise RuntimeError(f"DINOv2 weights {weights_path} are missing keys: {missing[:10]}...")
            if unexpected:
                print(f"[warn] ignored {len(unexpected)} unexpected keys from {weights_path} "
                      f"(e.g. {unexpected[:5]})")
            print(f"DINOv2 weights loaded from {weights_path}")
        if not pretrained and not weights_path:
            print("[warn] DINOv2 is randomly initialised (--dinov2-pretrained n)")

        self.patch_size = int(self.vit.patch_size)
        self.embed_dim = int(self.vit.embed_dim)
        depth = int(getattr(self.vit, "n_blocks", len(self.vit.blocks)))
        if layers is None:
            layers = [depth // 2 - 1, (3 * depth) // 4 - 1, depth - 1]
        layers = [int(i) for i in layers]
        if len(layers) != 3 or sorted(set(layers)) != layers or layers[0] < 0 or layers[-1] >= depth:
            raise ValueError(f"--dinov2-layers needs 3 strictly increasing block indices in [0, {depth - 1}], "
                             f"got {layers}")
        self.layers = layers
        self.vit_input_size = int(vit_input_size or 0)
        if self.vit_input_size and self.vit_input_size % self.patch_size:
            raise ValueError(f"--vit-input-size must be a multiple of {self.patch_size}")
        self.out_channels = tuple(int(c) for c in out_channels)
        self.amp = bool(amp)

        self.adapters = nn.ModuleList([_adapter(self.embed_dim, c) for c in self.out_channels])

        self.frozen = freeze_backbone == "y"
        print(f"Freezing the backbone of the image encode: {freeze_backbone}")
        if self.frozen:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()
        elif grad_checkpointing:
            self._enable_grad_checkpointing()

        print(f"DINOv2: depth={depth}, embed_dim={self.embed_dim}, taps={self.layers}, "
              f"C3/C4/C5 channels={self.out_channels}, vit_input_size="
              f"{self.vit_input_size or 'auto (ceil to multiple of %d)' % self.patch_size}, amp(bf16)={self.amp}")

    def _enable_grad_checkpointing(self):
        # Patch forward in place so state_dict keys stay identical.
        for blk in self.vit.blocks:
            original = blk.forward

            def forward(x, _orig=original, _blk=blk):
                if _blk.training and torch.is_grad_enabled():
                    return checkpoint(_orig, x, use_reentrant=False)
                return _orig(x)

            blk.forward = forward
        print("Gradient checkpointing enabled on DINOv2 blocks")

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def _vit_input(self, x):
        h, w = x.shape[-2:]
        if self.vit_input_size:
            th = tw = self.vit_input_size
        else:
            p = self.patch_size
            th, tw = math.ceil(h / p) * p, math.ceil(w / p) * p
        if (th, tw) != (h, w):
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        h, w = x.shape[-2:]
        vit_in = self._vit_input(x)
        use_amp = self.amp and x.is_cuda
        grad_ctx = torch.no_grad() if self.frozen else nullcontext()
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with grad_ctx, amp_ctx:
            feats = self.vit.get_intermediate_layers(vit_in, n=self.layers, reshape=True, norm=True)
        outs = []
        for feat, adapter, stride in zip(feats, self.adapters, self.strides):
            size = ((h + stride - 1) // stride, (w + stride - 1) // stride)  # == Anchors grid
            outs.append(adapter(_resample(feat.float(), size)))
        C3, C4, C5 = outs
        return C5, C3, C4, C5


def build_dinov2_retinanet(args, RetinaNet):
    layers = [int(v) for v in args.dinov2_layers.split(",")] if args.dinov2_layers else None
    backbone = DINOv2Backbone(
        arch=args.arch,
        hub_repo=args.dinov2_hub_dir or DINOV2_HUB_REPO,
        hub_source="local" if args.dinov2_hub_dir else "github",
        weights_path=args.dinov2_weights,
        pretrained=args.dinov2_pretrained == "y",
        layers=layers,
        vit_input_size=args.vit_input_size,
        freeze_backbone=args.freeze_backbone,
        grad_checkpointing=args.grad_checkpointing == "y",
        amp=args.amp == "y",
        cache_dir=args.dinov2_cache_dir or None,
    )
    # Mammo-FM RetinaNet, unchanged (FPN P3-P7, heads, anchors, focal loss, NMS).
    return RetinaNet(
        len(args.concepts), backbone, list(backbone.out_channels),
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
    )


# ----------------------------------------------------------------------------
# Training (mirror of Detectors/experiments.py: do_experiements + train_loop)
# ----------------------------------------------------------------------------
class NullLogger:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def make_logger(path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(path)
    except Exception as exc:  # tensorboard missing / unwritable dir
        print(f"[warn] TensorBoard disabled: {exc}")
        return NullLogger()


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _reseed_imgaug_worker(worker_id):
    import imgaug
    imgaug.seed(torch.initial_seed() % (2 ** 32))


def do_experiements(args, device, mfm):
    args.data_dir = Path(args.data_dir)
    csv_path = args.data_dir / args.csv_file
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    args.df = pd.read_csv(csv_path)
    print(csv_path)
    args.df = args.df.fillna(0)
    print(f"df shape: {args.df.shape}")

    if args.dataset.lower() == 'vindr':
        args.df = args.df.head(2254)
    args.model_base_name = args.arch
    mfm["seed_all"](args.seed)
    args.cur_fold = 0

    if args.dataset.lower() == 'vindr':
        args.train_folds = args.df[args.df['split'] == "training"].reset_index(drop=True)
        args.valid_folds = args.df[args.df['split'] == "test"].reset_index(drop=True)
    elif args.dataset.lower() == 'embed':
        args.df["anon_dicom_path"] = args.df["anon_dicom_path"].str.replace(
            "/ocean/projects/asc170022p/shared/Projects/breast-imaging/RSNA_Breast_Imaging/Dataset", str(args.data_dir))
        args.train_folds = args.df[args.df['fold'] != args.cur_fold].reset_index(drop=True)
        args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported dataset for detection: {args.dataset} (ViNDr or EMBED)")
    print(f'train: {args.train_folds.shape}', f'valid: {args.valid_folds.shape}')
    if len(args.train_folds) == 0 or len(args.valid_folds) == 0:
        raise ValueError("Empty train or validation split; check the CSV 'split'/'fold' columns.")

    if args.running_interactive:
        args.train_folds = args.train_folds.head(100)
        args.valid_folds = args.valid_folds.head(n=1000)

    train_loop(args, device, mfm)

    print(f'Checkpoints saved at: {args.chk_pt_path}')
    print(f'Outputs saved at: {args.output_path}')


def train_loop(args, device, mfm):
    print(f'\n================== fold: {args.cur_fold} training ======================')
    if args.data_frac < 1.0:
        args.train_folds = args.train_folds.sample(frac=args.data_frac, random_state=1, ignore_index=True)
    train_loader, valid_loader, valid_dataset = mfm["get_dataset"](args)
    if args.reseed_imgaug_workers == "y" and args.num_workers > 0:
        train_loader.worker_init_fn = _reseed_imgaug_worker
    print(f'train_loader: {len(train_loader)}', f'valid_loader: {len(valid_dataset)}')
    if len(train_loader) == 0:
        raise ValueError("Train loader is empty (fewer images than --batch-size with drop_last=True).")

    retinanet = build_dinov2_retinanet(args, mfm["RetinaNet"])
    print(retinanet)
    n_total = sum(p.numel() for p in retinanet.parameters())
    n_train = sum(p.numel() for p in retinanet.parameters() if p.requires_grad)
    print(f"Parameters: total={n_total / 1e6:.2f}M trainable={n_train / 1e6:.2f}M")
    retinanet.to(device)
    retinanet.training = True

    backbone_params = [p for n, p in retinanet.named_parameters()
                       if p.requires_grad and n.startswith("efficientnet.vit.")]
    other_params = [p for n, p in retinanet.named_parameters()
                    if p.requires_grad and not n.startswith("efficientnet.vit.")]
    param_groups = [{"params": other_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr or args.lr})
    optimizer = torch.optim.Adam(param_groups, lr=args.lr)
    logger = make_logger(args.tb_logs_path / f'fold{args.cur_fold}')

    ckpt_prefix = f'{args.model_base_name}_seed_{args.seed}_fold_{args.cur_fold}'
    last_ckpt_path = args.chk_pt_path / f'{ckpt_prefix}_last.pth'
    best_MAP = 0.
    start_epoch = 0
    resume_path = last_ckpt_path if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if resume_path is not None and resume_path.is_file():
        ckpt = torch.load(resume_path, map_location=device)
        retinanet.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_MAP = float(ckpt.get("best_MAP", 0.))
        print(f"Resumed from {resume_path}: next epoch {start_epoch + 1}, best mAP {best_MAP:.4f}")
    elif args.resume and args.resume != "auto":
        raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")

    loss_hist = collections.deque(maxlen=500)
    cls_loss_hist = collections.deque(maxlen=500)
    reg_loss_hist = collections.deque(maxlen=500)
    retinanet.train()

    for epoch_num in range(start_epoch, args.epochs):
        start_time = time.time()
        retinanet.train()

        start = time.time()
        epoch_loss = []
        epoch_cls_loss = []
        epoch_reg_loss = []
        consecutive_failures = 0
        for iter_num, data in enumerate(train_loader):
            try:
                optimizer.zero_grad(set_to_none=True)
                img = data["image"].to(device, non_blocking=True)
                bbox = data["res_bbox_tensor"].to(device, non_blocking=True)
                classification_loss, regression_loss = retinanet([img, bbox])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss {float(loss)} (cls={float(classification_loss)}, "
                                             f"reg={float(regression_loss)}); skipping step")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                consecutive_failures = 0
                loss_hist.append(float(loss))
                cls_loss_hist.append(float(classification_loss))
                reg_loss_hist.append(float(regression_loss))

                epoch_loss.append(float(loss))
                epoch_cls_loss.append(float(classification_loss))
                epoch_reg_loss.append(float(regression_loss))
                if iter_num % args.print_freq == 0 or iter_num == (len(train_loader) - 1):
                    print('Epoch: [{0}][{1}/{2}] '
                          'Elapsed {remain:s} '
                          'Classification loss: {cls_loss:1.5f} '
                          'Regression loss: {reg_loss:1.5f} '
                          'Running loss: {run_loss:1.5f} '.format(
                              epoch_num + 1, iter_num, len(train_loader),
                              remain=mfm["timeSince"](start, float(iter_num + 1) / len(train_loader)),
                              cls_loss=float(classification_loss),
                              reg_loss=float(regression_loss),
                              run_loss=np.mean(loss_hist)), flush=True)

                if iter_num % args.log_freq == 0 or iter_num == (len(train_loader) - 1):
                    index = iter_num + len(train_loader) * epoch_num
                    logger.add_scalar('train/iter_loss', np.mean(loss_hist), index)
                    logger.add_scalar('train/iter_cls_loss', np.mean(cls_loss_hist), index)
                    logger.add_scalar('train/iter_reg_loss', np.mean(reg_loss_hist), index)

                del classification_loss
                del regression_loss
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                # Mammo-FM skips failing batches; we do the same but stay loud and bounded.
                consecutive_failures += 1
                optimizer.zero_grad(set_to_none=True)
                print(f"[warn] epoch {epoch_num + 1} iter {iter_num} skipped: {e}", flush=True)
                if consecutive_failures == 1:
                    traceback.print_exc()
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                if consecutive_failures >= args.max_consecutive_failures:
                    raise RuntimeError(f"{consecutive_failures} consecutive failed iterations, aborting") from e
                continue

        if not epoch_loss:
            raise RuntimeError(f"Epoch {epoch_num + 1}: no successful training step")

        _, MAP = mfm["evaluate"](valid_dataset, logger=logger, epoch_num=epoch_num, concepts=args.concepts,
                                 retinanet=retinanet, iou_threshold=args.iou_threshold,
                                 score_threshold=args.score_threshold, max_detections=args.max_detections)
        MAP = float(MAP)
        logger.add_scalar('train/epoch_loss', np.mean(np.array(epoch_loss)), epoch_num)
        logger.add_scalar('train/epoch_cls_loss', np.mean(np.array(epoch_cls_loss)), epoch_num)
        logger.add_scalar('train/epoch_reg_loss', np.mean(np.array(epoch_reg_loss)), epoch_num)
        logger.add_scalar('valid/MAP', MAP, epoch_num)

        elapsed = time.time() - start_time
        print(
            f'Epoch {epoch_num + 1} - avg_train_loss: {np.mean(np.array(epoch_loss)):.4f},  '
            f'time: {elapsed:.0f}s, MAP: {MAP:.4f}', flush=True
        )

        state_dict = retinanet.state_dict()
        if args.save_all_epochs == "y":
            atomic_save({'state_dict': state_dict, 'MAP': MAP},
                        args.chk_pt_path / f'{ckpt_prefix}_epoch_{epoch_num}.pth')

        if best_MAP < MAP:
            best_MAP = MAP
            print(f'Epoch {epoch_num + 1} - Save Best mAP: {best_MAP:.4f} Model')
            atomic_save({'state_dict': state_dict, 'best_MAP': best_MAP, 'epoch': epoch_num},
                        args.chk_pt_path / f'{ckpt_prefix}_best_auroc.pth')

        atomic_save({'state_dict': state_dict, 'optimizer': optimizer.state_dict(), 'epoch': epoch_num,
                     'MAP': MAP, 'best_MAP': best_MAP}, last_ckpt_path)
        del state_dict
    print(f'[Fold{args.cur_fold}] Best MAP: {best_MAP:.4f}')

    logger.close()
    torch.cuda.empty_cache()
    gc.collect()


# ----------------------------------------------------------------------------
# CLI (same names/defaults as Mammo-FM train_detector.py + DINOv2 options)
# ----------------------------------------------------------------------------
def config():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tensorboard-path', metavar='DIR', default='/workspace/outputs/tensorboard',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR', default='/workspace/outputs/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output_path', metavar='DIR', default='/workspace/outputs/logs',
                        help='path to output logs')
    parser.add_argument("--data-dir", default="/workspace/data/vindr_png", type=str, help="Path to data file")
    parser.add_argument("--img-dir", default="images_png", type=str, help="Path to image file")
    parser.add_argument("--csv-file", default="vindr_detection_v1_folds.csv", type=str, help="data csv file")
    parser.add_argument("--dataset", default="ViNDr", type=str, help="Dataset name?")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    parser.add_argument("--freeze_backbone", default='n', type=str, choices=["y", "n"],
                        help="Freeze DINOv2? (y/n). Adapter, FPN and heads are always trained")
    parser.add_argument("--arch", default="dinov2_vitb14", type=str, choices=sorted(DINOV2_ARCHS),
                        help="DINOv2 variant (torch.hub entrypoint name)")
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--score-threshold", default=0.05, type=float)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--max-detections", default=100, type=int)
    parser.add_argument("--num_cycles", default=0.5, type=float)
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--focal-alpha", default=0.6, type=float)
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_folds", default=4, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=7, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--detection-threshold", default=0.3, type=float)
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--resize", default=512, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument('--model-type', default="Concept-Detector", type=str)
    parser.add_argument("--apex", default="y", type=str, help="Parsed for CLI parity; unused (as in Mammo-FM)")
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument("--concepts", default="Suspicious Calcification", type=str)

    # --- DINOv2 / robustness options (not in Mammo-FM) ---
    parser.add_argument("--mammo-fm-codebase", default=os.environ.get("MAMMO_FM_CODEBASE", DEFAULT_MAMMO_FM_CODEBASE),
                        type=str, help="Path to Mammo-FM/src/codebase (also settable with $MAMMO_FM_CODEBASE)")
    parser.add_argument("--dinov2-hub-dir", default=None, type=str,
                        help="Local clone of facebookresearch/dinov2 (torch.hub source='local'); default: GitHub")
    parser.add_argument("--dinov2-weights", default=None, type=str,
                        help="Local DINOv2 backbone .pth (e.g. dinov2_vitb14_pretrain.pth) instead of downloading")
    parser.add_argument("--dinov2-cache-dir", default="/workspace/checkpoints", type=str,
                        help="Where the pretrained DINOv2 weights (<arch>_pretrain.pth) and torch.hub code are "
                             "saved and reused on later runs. '' = default torch.hub cache")
    parser.add_argument("--dinov2-pretrained", default="y", type=str, choices=["y", "n"],
                        help="n = random init (ablation)")
    parser.add_argument("--dinov2-layers", default=None, type=str,
                        help="3 comma-separated block indices for C3,C4,C5 (default: 50%%,75%%,100%% depth)")
    parser.add_argument("--vit-input-size", default=0, type=int,
                        help="Side fed to the ViT (multiple of 14). 0 = ceil(--resize/14)*14 (518 for 512)")
    parser.add_argument("--backbone-lr", default=None, type=float,
                        help="Optional separate lr for DINOv2 weights (default: same as --lr, like Mammo-FM)")
    parser.add_argument("--amp", default="n", type=str, choices=["y", "n"],
                        help="bf16 autocast for the ViT only (FPN/heads/focal loss stay fp32)")
    parser.add_argument("--grad-checkpointing", default="n", type=str, choices=["y", "n"],
                        help="Activation checkpointing on ViT blocks to save memory")
    parser.add_argument("--save-all-epochs", default="y", type=str, choices=["y", "n"],
                        help="Save a checkpoint every epoch like Mammo-FM (n = only best + last)")
    parser.add_argument("--resume", default="", type=str,
                        help="'auto' = resume from <prefix>_last.pth in the run dir if present, or a path")
    parser.add_argument("--max-consecutive-failures", default=20, type=int,
                        help="Abort if this many training iterations fail in a row")
    parser.add_argument("--reseed-imgaug-workers", default="n", type=str, choices=["y", "n"],
                        help="Give each DataLoader worker its own imgaug seed (Mammo-FM does not)")
    return parser.parse_args()


def validate_args(args):
    if args.resize % 32 != 0:
        raise ValueError(f"--resize must be a multiple of 32 so C3/C4/C5 match the FPN, got {args.resize}")
    if args.model_type.lower() != "concept-detector":
        raise ValueError("--model-type must be Concept-Detector for detection")
    if not 0 < args.data_frac <= 1.0:
        raise ValueError("--data_frac must be in (0, 1]")
    if args.dinov2_weights and not Path(args.dinov2_weights).is_file():
        raise FileNotFoundError(f"--dinov2-weights not found: {args.dinov2_weights}")
    if args.dinov2_hub_dir and not Path(args.dinov2_hub_dir, "hubconf.py").is_file():
        raise FileNotFoundError(f"--dinov2-hub-dir has no hubconf.py: {args.dinov2_hub_dir}")


def main(args):
    mfm = import_mammo_fm(args.mammo_fm_codebase)
    validate_args(args)
    print("PyTorch Version:", torch.__version__)
    args.concepts = [args.concepts]
    print("Torchvision Version:", torchvision.__version__)
    mfm["seed_all"](args.seed)
    device = 'cuda' if torch.cuda.is_available() and args.device == "cuda" else 'cpu'
    if args.amp == "y" and device == "cuda" and not torch.cuda.is_bf16_supported():
        print("[warn] GPU has no bf16 support, disabling --amp")
        args.amp = "n"
    args.root = (f"lr_{args.lr}_epochs_{args.epochs}_concepts_{args.concepts[0]}_alpha_{args.focal_alpha}"
                 f"_gamma_{args.focal_gamma}_score_th_{args.score_threshold}_data_frac_{args.data_frac}"
                 f"_freeze_backbone_{args.freeze_backbone}")

    args.apex = True if args.apex == "y" else False
    args.running_interactive = True if args.running_interactive == "y" else False
    chk_pt_path, output_path, tb_logs_path = mfm["get_Paths"](args)
    args.chk_pt_path = chk_pt_path
    args.output_path = output_path
    args.tb_logs_path = tb_logs_path

    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    print("====================> Paths <====================")
    print(f"checkpoint_path: {chk_pt_path}")
    print(f"output_path: {output_path}")
    print(f"tb_logs_path: {tb_logs_path}")
    print('device:', device)
    print('torch version:', torch.__version__)
    print(f'No. of concept: {len(args.concepts)}')
    print("====================> Paths <====================")

    pickle.dump(args, open(os.path.join(output_path, f"seed_{args.seed}_train_configs.pkl"), "wb"))
    torch.cuda.empty_cache()

    do_experiements(args, device, mfm)


if __name__ == "__main__":
    main(config())
