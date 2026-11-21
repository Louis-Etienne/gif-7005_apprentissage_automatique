#!/usr/bin/env python3
"""Extraction exhaustive des embeddings DINOv2 pour RSNA.

Exemples :
    python -u scripts/extract_embeddings_dinov2_gpu.py --split all --batch-size 8
    python -u scripts/extract_embeddings_dinov2_gpu.py --split train --batch-size 8

Le modèle est gelé : il s'agit d'extraction de caractéristiques, pas de fine-tuning.
"""

import argparse
import csv
import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data/raw/train_images"
SPLIT_DIR = PROJECT_ROOT / "data/processed/splits"
EMBEDDING_ROOT = PROJECT_ROOT / "data/processed/embeddings/dinov2"
MODEL_NAME = "facebook/dinov2-small"
EMBEDDING_DIM = 384
EXPECTED_SPLITS = {"train": 38341, "validation": 8184, "test": 8181}
LOGGER = logging.getLogger("dinov2_rsna")


def parse_args():
    parser = argparse.ArgumentParser(description="Embeddings DINOv2 RSNA, toutes les images")
    parser.add_argument("--split", choices=["all", *EXPECTED_SPLITS], default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--verify-existing", action="store_true", help="Relire et vérifier les fichiers .npy existants (plus lent)")
    return parser.parse_args()


def prepare_image(dicom_path: Path) -> Image.Image:
    ds = pydicom.dcmread(dicom_path)
    pixels = ds.pixel_array
    if pixels.ndim != 2:
        raise ValueError(f"DICOM non 2D : {pixels.shape}")
    image = pixels.astype(np.float32)
    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        image = image.max() - image
    if not np.isfinite(image).all():
        raise ValueError("Pixels NaN/Inf")
    image_min, image_max = float(image.min()), float(image.max())
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)
    pil_image = Image.fromarray(np.rint(image * 255).astype(np.uint8))
    pil_image.thumbnail((518, 518), Image.Resampling.LANCZOS)
    return pil_image.convert("RGB")


def embedding_ok(path: Path, verify: bool) -> bool:
    if not path.is_file():
        return False
    if not verify:
        # NPY float32 : en-tête (~128 octets) + 384 * 4 octets.
        return path.stat().st_size >= EMBEDDING_DIM * 4
    try:
        arr = np.load(path, allow_pickle=False)
        return arr.shape == (EMBEDDING_DIM,) and arr.dtype == np.float32 and np.isfinite(arr).all()
    except (OSError, ValueError, EOFError):
        return False


def save_embedding(path: Path, embedding: np.ndarray):
    if embedding.shape != (EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError(f"Embedding invalide : {embedding.shape}")
    tmp = path.with_suffix(".npy.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, embedding.astype(np.float32))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_split(split: str) -> pd.DataFrame:
    csv_path = SPLIT_DIR / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, dtype={"patient_id": "string", "image_id": "string"})
    if not {"patient_id", "image_id"}.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes dans {csv_path}")
    if df[["patient_id", "image_id"]].isna().any().any():
        raise ValueError(f"Identifiants manquants dans {csv_path}")
    if df.duplicated(["patient_id", "image_id"]).any():
        raise ValueError(f"Doublons dans {csv_path}")
    expected = EXPECTED_SPLITS[split]
    if len(df) != expected:
        raise ValueError(f"{split} : {len(df)} lignes, attendu {expected}")
    return df


def infer_batch(images, paths, processor, model, device):
    """Échoue explicitement sur erreur GPU : fichiers déjà sauvés réutilisables."""
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        vectors = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
    if len(vectors) != len(paths):
        raise RuntimeError("Nombre d'embeddings différent du lot")
    for path, vector in zip(paths, vectors):
        save_embedding(path, vector)
    return len(paths)


def process_split(split, df, processor, model, device, args):
    output_dir = EMBEDDING_ROOT / split
    output_dir.mkdir(parents=True, exist_ok=True)
    error_path = output_dir / "errors.csv"
    # Les erreurs sont réécrites à chaque exécution; le bilan final recense les manquants.
    stats = {"new": 0, "existing": 0, "read_errors": 0}
    batch_images, batch_paths = [], []
    start = time.monotonic()
    with error_path.open("w", newline="", encoding="utf-8") as error_file:
        writer = csv.writer(error_file)
        writer.writerow(["patient_id", "image_id", "error"])
        bar = tqdm(df.itertuples(index=False), total=len(df), desc=split, unit="img")
        for index, row in enumerate(bar, 1):
            patient_id, image_id = str(row.patient_id), str(row.image_id)
            output_path = output_dir / f"{patient_id}_{image_id}.npy"
            if embedding_ok(output_path, args.verify_existing):
                stats["existing"] += 1
            else:
                dicom_path = IMAGE_DIR / patient_id / f"{image_id}.dcm"
                try:
                    image = prepare_image(dicom_path)
                except Exception as exc:
                    stats["read_errors"] += 1
                    writer.writerow([patient_id, image_id, repr(exc)])
                    error_file.flush()
                    LOGGER.error("Erreur DICOM %s/%s : %s", patient_id, image_id, exc)
                    continue
                batch_images.append(image)
                batch_paths.append(output_path)
                if len(batch_images) == args.batch_size:
                    stats["new"] += infer_batch(batch_images, batch_paths, processor, model, device)
                    batch_images.clear()
                    batch_paths.clear()
            if index % args.progress_every == 0:
                LOGGER.info("%s [%d/%d] nouveaux=%d existants=%d erreurs=%d (%.1f min)",
                            split, index, len(df), stats["new"], stats["existing"],
                            stats["read_errors"], (time.monotonic() - start) / 60)
        if batch_images:
            stats["new"] += infer_batch(batch_images, batch_paths, processor, model, device)

    # Vérification exhaustive de la présence et de la forme des embeddings.
    missing = []
    invalid = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Vérification {split}", unit="img"):
        patient_id, image_id = str(row.patient_id), str(row.image_id)
        path = output_dir / f"{patient_id}_{image_id}.npy"
        if not path.is_file():
            missing.append((patient_id, image_id))
        elif not embedding_ok(path, verify=True):
            invalid.append((patient_id, image_id))
    report = output_dir / "verification.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "image_id", "status"])
        writer.writerows((p, i, "missing") for p, i in missing)
        writer.writerows((p, i, "invalid") for p, i in invalid)
    LOGGER.info("BILAN %s : attendus=%d nouveaux=%d existants=%d erreurs_DICOM=%d manquants=%d invalides=%d",
                split, len(df), stats["new"], stats["existing"], stats["read_errors"], len(missing), len(invalid))
    return len(missing) + len(invalid)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.batch_size < 1 or args.progress_every < 1:
        raise ValueError("batch-size et progress-every doivent être >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    LOGGER.info("Périphérique : %s", device)
    if device.type == "cuda":
        LOGGER.info("GPU : %s", torch.cuda.get_device_name(0))
    splits = list(EXPECTED_SPLITS) if args.split == "all" else [args.split]
    datasets = {split: load_split(split) for split in splits}
    if args.split == "all":
        ids = [set(zip(df.patient_id, df.image_id)) for df in datasets.values()]
        patients = [set(df.patient_id) for df in datasets.values()]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] & ids[j] or patients[i] & patients[j]:
                    raise ValueError("Chevauchement entre les partitions")
    LOGGER.info("Images à traiter : %d", sum(len(df) for df in datasets.values()))
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    LOGGER.info("Modèle chargé : %s", MODEL_NAME)
    failures = 0
    for split, df in datasets.items():
        failures += process_split(split, df, processor, model, device, args)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f"Extraction incomplète : {failures} embeddings manquants/invalides")
    LOGGER.info("SUCCÈS : tous les embeddings demandés ont été vérifiés")


if __name__ == "__main__":
    main()
