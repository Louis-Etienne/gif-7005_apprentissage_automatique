#!/usr/bin/env python3
"""Extract frozen Mammo-FM pooled image features from all RSNA split DICOMs.

Usage:
  python -u scripts/extract_embeddings_mammofm.py --splits train validation test \
      2>&1 | tee -a logs/mammofm_embeddings.log

Does not read labels or calculate test-set performance.
"""
import argparse
import csv
import logging
import os
import time
from pathlib import Path

import mammofm as mf
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("mammofm_extract")
DIM = 2048


def valid_embedding(path: Path, verify: bool) -> bool:
    if not path.is_file():
        return False
    if not verify:
        # A 2048-element float32 .npy is slightly larger than 8192 bytes.
        return path.stat().st_size >= DIM * 4 + 100
    try:
        arr = np.load(path, allow_pickle=False)
        return arr.shape == (DIM,) and arr.dtype == np.float32 and bool(np.isfinite(arr).all())
    except (OSError, ValueError, EOFError):
        return False


def atomic_save(path: Path, vector: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, vector, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_ids(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV absent: {csv_path}")
    frame = pd.read_csv(csv_path, dtype={"patient_id": "string", "image_id": "string"}, usecols=["patient_id", "image_id"])
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"Split vide ou identifiants manquants: {csv_path}")
    if frame.duplicated(["patient_id", "image_id"]).any():
        raise ValueError(f"Doublons patient_id/image_id: {csv_path}")
    return frame


def process_split(split: str, args: argparse.Namespace) -> dict:
    frame = read_ids(args.split_root / f"{split}.csv")
    out_dir = args.output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    error_path = out_dir / "errors.csv"
    # Append errors rather than overwriting logs from a previous interrupted run.
    needs_header = not error_path.exists() or error_path.stat().st_size == 0
    counts = {"total": len(frame), "new": 0, "skipped": 0, "errors": 0}
    started = time.monotonic()
    LOGGER.info("Début %s : %d images", split, len(frame))
    with error_path.open("a", newline="", encoding="utf-8") as error_file:
        errors = csv.writer(error_file)
        if needs_header:
            errors.writerow(["patient_id", "image_id", "error"])
            error_file.flush()
        for index, (patient_id, image_id) in enumerate(frame.itertuples(index=False, name=None), 1):
            patient_id, image_id = str(patient_id), str(image_id)
            target = out_dir / f"{patient_id}_{image_id}.npy"
            if valid_embedding(target, verify=args.verify_existing):
                counts["skipped"] += 1
            else:
                source = args.image_root / patient_id / f"{image_id}.dcm"
                try:
                    if not source.is_file():
                        raise FileNotFoundError(str(source))
                    # Official Mammo-FM preprocessing for RSNA DICOMs.
                    features = mf.features(str(source), "rsna")
                    pooled = np.asarray(features["after_pooling"], dtype=np.float32)
                    if pooled.shape != (1, DIM):
                        raise ValueError(f"Unexpected pooled shape: {pooled.shape}")
                    vector = pooled[0]
                    if not np.isfinite(vector).all():
                        raise ValueError("Non-finite embedding")
                    atomic_save(target, vector)
                    counts["new"] += 1
                except Exception as exc:
                    counts["errors"] += 1
                    errors.writerow([patient_id, image_id, repr(exc)])
                    error_file.flush()
                    LOGGER.error("%s %s/%s : %s", split, patient_id, image_id, exc)
            if index % args.log_every == 0 or index == len(frame):
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0.0
                remaining = (len(frame) - index) / rate if rate else float("inf")
                LOGGER.info(
                    "%s %d/%d (%.1f%%) | nouveaux=%d existants=%d erreurs=%d | %.2f img/s | ETA %.1f min",
                    split, index, len(frame), 100 * index / len(frame),
                    counts["new"], counts["skipped"], counts["errors"], rate, remaining / 60,
                )
    LOGGER.info("Fin %s : %s", split, counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Full RSNA Mammo-FM pooled embedding extraction")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation", "test"])
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/train_images")
    parser.add_argument("--split-root", type=Path, default=ROOT / "data/processed/splits")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/embeddings/mammofm")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--verify-existing", action="store_true", help="Fully load/check existing npy files before skipping")
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    all_counts = [process_split(split, args) for split in args.splits]
    total_errors = sum(s["errors"] for s in all_counts)
    LOGGER.info("Toutes partitions terminées; erreurs lors de cette exécution : %d", total_errors)
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
