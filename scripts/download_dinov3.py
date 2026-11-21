#!/usr/bin/env python3
"""Télécharger le checkpoint officiel DINOv3 ViT-S/16 (accès Hugging Face requis)."""
import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'facebook/dinov3-vits16-pretrain-lvd1689m'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=ROOT / 'models/dinov3')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info('Téléchargement de %s vers %s', REPO, args.output_dir)
    try:
        path = snapshot_download(
            repo_id=REPO,
            local_dir=str(args.output_dir),
            allow_patterns=['*.json', '*.safetensors', '*.safetensors.index.json', '*.txt'],
        )
    except Exception as exc:
        raise SystemExit(
            f'Téléchargement impossible : {exc}\n'
            'Vérifier que les conditions d’accès au modèle ont été acceptées '
            'et que `hf auth login` a été exécuté.'
        ) from exc
    logging.info('Checkpoint disponible dans %s', path)

if __name__ == '__main__':
    main()
