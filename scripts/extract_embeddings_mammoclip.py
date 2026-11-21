#!/usr/bin/env python3
"""Extraire les embeddings Mammo-CLIP B5 des DICOM RSNA, sans utiliser les labels.

Test : python -u scripts/extract_embeddings_mammoclip.py --splits validation --max-images 2
Production : python -u scripts/extract_embeddings_mammoclip.py --splits train validation test
"""
import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'data/processed/splits'
DICOMS = ROOT / 'data/raw/train_images'
OUTPUT = ROOT / 'data/processed/embeddings/mammoclip'
CHECKPOINT = ROOT / 'models/mammoclip/Pre-trained-checkpoints/b5-model-best-epoch-7.tar'
EXPECTED = {'train': 38341, 'validation': 8184, 'test': 8181}
DIM = 2048
MEAN = 0.3089279
STD = 0.25053555408335154
LOG = logging.getLogger('mammoclip')


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--splits', nargs='+', choices=list(EXPECTED), default=['train', 'validation', 'test'])
    p.add_argument('--max-images', type=int, default=None, help='Essai rapide, par partition')
    p.add_argument('--height', type=int, default=1520)
    p.add_argument('--width', type=int, default=912)
    p.add_argument('--progress-every', type=int, default=100)
    p.add_argument('--verify-existing', action='store_true')
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    return p.parse_args()


def load_model(device):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    # Le checkpoint est un pickle : ne charger que des fichiers de provenance fiable.
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    cfg = ckpt['config']['model']['image_encoder']
    LOG.info('Configuration encodeur : %s', cfg)
    if cfg.get('name', '').lower() != 'tf_efficientnet_b5_ns-detect':
        raise ValueError(f"Encodeur inattendu : {cfg.get('name')}")
    sys.path.insert(0, str(ROOT / 'external/Mammo-CLIP/src/codebase'))
    from breastclip.model.modules import load_image_encoder
    encoder = load_image_encoder(cfg)
    weights = {key[len('image_encoder.'):]: value for key, value in ckpt['model'].items()
               if key.startswith('image_encoder.')}
    if not weights:
        raise RuntimeError('Aucun poids image_encoder.* dans le checkpoint')
    encoder.load_state_dict(weights, strict=True)
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    del ckpt
    LOG.info('Encodeur chargé avec %d tenseurs', len(weights))
    return encoder


def prepare(path, height, width):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f'Image invalide : {path}, forme={image.shape}')
    if ds.get('PhotometricInterpretation') == 'MONOCHROME1':
        image = image.max() - image
    low, high = float(image.min()), float(image.max())
    image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    # Adaptation DICOM -> RGB et dimensions H=1520, W=912.
    # Le dépôt officiel attend des PNG RSNA prétraités : vérifier leur pipeline
    # avant de considérer cette conversion comme une reproduction exacte.
    rgb = Image.fromarray(np.rint(image * 255).astype(np.uint8)).convert('RGB')
    rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    arr = (arr - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def valid(path, thorough=False):
    if not path.is_file():
        return False
    if not thorough:
        return path.stat().st_size >= DIM * 4
    try:
        a = np.load(path, allow_pickle=False)
        return a.shape == (DIM,) and a.dtype == np.float32 and np.isfinite(a).all()
    except (OSError, ValueError, EOFError):
        return False


def save(path, vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape != (DIM,) or not np.isfinite(vec).all():
        raise ValueError(f'Embedding incorrect : {vec.shape}')
    temp = path.with_suffix('.npy.tmp')
    try:
        with temp.open('wb') as f:
            np.save(f, vec)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    args = arguments()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if args.height < 1 or args.width < 1 or args.progress_every < 1:
        raise ValueError('Dimensions et fréquence de suivi doivent être positives')
    if args.max_images is not None and args.max_images < 1:
        raise ValueError('--max-images doit être positif')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA indisponible')
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    LOG.info('Device=%s ; image=%dx%d', device, args.height, args.width)
    tables = {}
    for split in args.splits:
        df = pd.read_csv(SPLITS / f'{split}.csv', dtype={'patient_id': 'string', 'image_id': 'string'})
        if not {'patient_id', 'image_id'}.issubset(df.columns):
            raise ValueError(f'Colonnes absentes : {split}')
        if len(df) != EXPECTED[split] or df[['patient_id', 'image_id']].isna().any().any():
            raise ValueError(f'Partition {split} incorrecte : {len(df)} lignes')
        if df.duplicated(['patient_id', 'image_id']).any():
            raise ValueError(f'Identifiants dupliqués : {split}')
        tables[split] = df.head(args.max_images) if args.max_images else df
    model = load_model(device)
    failures = 0
    for split, df in tables.items():
        folder = OUTPUT / split
        folder.mkdir(parents=True, exist_ok=True)
        count_new = count_old = count_errors = 0
        started = time.monotonic()
        # Éviter de détruire un ancien journal en cas de reprise.
        with (folder / 'errors.csv').open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['patient_id', 'image_id', 'error'])
            for n, row in enumerate(df.itertuples(index=False), 1):
                patient, image = str(row.patient_id), str(row.image_id)
                target = folder / f'{patient}_{image}.npy'
                if valid(target, args.verify_existing):
                    count_old += 1
                    continue
                source = DICOMS / patient / f'{image}.dcm'
                try:
                    tensor = prepare(source, args.height, args.width).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        result = model(tensor)
                    if isinstance(result, (tuple, list)):
                        result = result[0]
                    vector = result.detach().float().cpu().numpy()
                    if vector.shape != (1, DIM):
                        raise ValueError(f'Sortie inattendue : {vector.shape}')
                    save(target, vector[0])
                    count_new += 1
                    del tensor, result
                except Exception as e:
                    count_errors += 1
                    writer.writerow([patient, image, repr(e)])
                    f.flush()
                    LOG.error('Erreur %s/%s : %s', patient, image, e)
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        raise
                if n % args.progress_every == 0 or n == len(df):
                    LOG.info('%s %d/%d | nouveaux=%d existants=%d erreurs=%d | %.1f min',
                             split, n, len(df), count_new, count_old, count_errors,
                             (time.monotonic() - started) / 60)
        # Vérification des fichiers pour la portion demandée.
        missing = sum(not valid(folder / f'{row.patient_id}_{row.image_id}.npy', True)
                      for row in df.itertuples(index=False))
        LOG.info('BILAN %s : nouveaux=%d existants=%d erreurs=%d manquants/invalides=%d',
                 split, count_new, count_old, count_errors, missing)
        failures += missing
    if failures:
        raise SystemExit(f'Extraction incomplète : {failures} embeddings manquants/invalides')
    LOG.info('SUCCÈS : embeddings demandés présents et valides')


if __name__ == '__main__':
    main()
