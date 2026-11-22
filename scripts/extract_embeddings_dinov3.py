#!/usr/bin/env python3
"""Embeddings RSNA DINOv3 ViT-S/16, sans entraînement ni consultation des labels.

Test: python -u scripts/extract_embeddings_dinov3.py --splits validation --max-images 2
Complet: python -u scripts/extract_embeddings_dinov3.py --splits train validation test
"""
import argparse
import csv
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'data/processed/splits'
DICOMS = ROOT / 'data/raw/train_images'
MODEL = ROOT / 'models/dinov3'
OUTPUT = ROOT / 'data/processed/embeddings/dinov3'
EXPECTED = {'train': 38341, 'validation': 8184, 'test': 8181}
LOG = logging.getLogger('dinov3')


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--splits', nargs='+', choices=list(EXPECTED), default=list(EXPECTED))
    p.add_argument('--max-images', type=int, default=None)
    p.add_argument('--progress-every', type=int, default=100)
    p.add_argument('--verify-existing', action='store_true')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return p.parse_args()


def prepare_dicom(path):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f'Image DICOM invalide : {path} ({image.shape})')
    if ds.get('PhotometricInterpretation') == 'MONOCHROME1':
        image = image.max() - image
    low, high = float(image.min()), float(image.max())
    image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    rgb = np.repeat(np.rint(image * 255).astype(np.uint8)[..., None], 3, axis=2)
    return Image.fromarray(rgb, mode='RGB')


def valid(path, dim, thorough=False):
    if not path.is_file():
        return False
    if not thorough:
        return path.stat().st_size >= dim * 4 + 128
    try:
        a = np.load(path, allow_pickle=False)
        return a.shape == (dim,) and a.dtype == np.float32 and np.isfinite(a).all()
    except (OSError, ValueError, EOFError):
        return False


def save(path, vector, dim):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape != (dim,) or not np.isfinite(vector).all():
        raise ValueError(f'Embedding invalide : {vector.shape}, attendu {(dim,)}')
    tmp = path.with_suffix('.npy.tmp')
    try:
        with tmp.open('wb') as f:
            np.save(f, vector)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    args = arguments()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.progress_every < 1 or (args.max_images is not None and args.max_images < 1):
        raise ValueError('Paramètres numériques invalides')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA demandé mais indisponible')
    device = torch.device('cuda' if (args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    if not (MODEL / 'config.json').is_file():
        raise FileNotFoundError(f'Checkpoint absent : {MODEL}. Exécuter download_dinov3.py')
    LOG.info('Chargement DINOv3 : %s ; device=%s', MODEL, device)
    processor = AutoImageProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    dim = int(model.config.hidden_size)
    LOG.info('Dimension embeddings : %d ; prétraitement : %s', dim, processor)

    tables = {}
    for split in args.splits:
        df = pd.read_csv(SPLITS / f'{split}.csv', dtype={'patient_id': str, 'image_id': str})
        if not {'patient_id', 'image_id'}.issubset(df.columns):
            raise ValueError(f'Colonnes manquantes dans {split}.csv')
        if len(df) != EXPECTED[split] or df[['patient_id', 'image_id']].isna().any().any():
            raise ValueError(f'Partition {split} inattendue : {len(df)} lignes')
        if df.duplicated(['patient_id', 'image_id']).any():
            raise ValueError(f'Doublons dans {split}')
        tables[split] = df.head(args.max_images) if args.max_images else df

    failures = 0
    for split, df in tables.items():
        folder = OUTPUT / split
        folder.mkdir(parents=True, exist_ok=True)
        new = old = errors = 0
        started = time.monotonic()
        with (folder / 'errors.csv').open('a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['patient_id', 'image_id', 'error'])
            for n, row in enumerate(df.itertuples(index=False), 1):
                patient, image_id = str(row.patient_id), str(row.image_id)
                target = folder / f'{patient}_{image_id}.npy'
                if valid(target, dim, args.verify_existing):
                    old += 1
                else:
                    source = DICOMS / patient / f'{image_id}.dcm'
                    try:
                        image = prepare_dicom(source)
                        inputs = processor(images=image, return_tensors='pt')
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.inference_mode():
                            result = model(**inputs)
                        # Représentation globale ViT : token CLS, sans tête de classification.
                        vector = result.last_hidden_state[:, 0, :].detach().float().cpu().numpy()[0]
                        save(target, vector, dim)
                        new += 1
                        del image, inputs, result, vector
                    except Exception as exc:
                        errors += 1
                        writer.writerow([patient, image_id, repr(exc)])
                        file.flush()
                        LOG.error('Erreur %s/%s : %s', patient, image_id, exc)
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            raise
                if n % args.progress_every == 0 or n == len(df):
                    LOG.info('%s %d/%d | nouveaux=%d existants=%d erreurs=%d | %.1f min',
                             split, n, len(df), new, old, errors,
                             (time.monotonic() - started) / 60)
        missing = sum(not valid(folder / f'{row.patient_id}_{row.image_id}.npy', dim, True)
                      for row in df.itertuples(index=False))
        LOG.info('BILAN %s : nouveaux=%d existants=%d erreurs=%d manquants/invalides=%d',
                 split, new, old, errors, missing)
        failures += missing
    if failures:
        raise SystemExit(f'Extraction incomplète : {failures} embeddings manquants/invalides')
    LOG.info('SUCCÈS : embeddings demandés présents et valides')


if __name__ == '__main__':
    main()
