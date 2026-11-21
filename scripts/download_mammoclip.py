
"""Téléchargement du checkpoint préentraîné Mammo-CLIP EfficientNet-B5."""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "shawn24/Mammo-CLIP"
FILENAME = "Pre-trained-checkpoints/b5-model-best-epoch-7.tar"

OUTPUT_DIR = PROJECT_ROOT / "models" / "mammoclip"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Début du téléchargement Mammo-CLIP")
    logging.info("Dépôt : %s", REPO_ID)
    logging.info("Checkpoint : %s", FILENAME)

    checkpoint_path = OUTPUT_DIR / FILENAME

    if checkpoint_path.is_file():
        logging.info("Checkpoint déjà présent : %s", checkpoint_path)
        return

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=OUTPUT_DIR,
    )

    logging.info("Téléchargement terminé")
    logging.info("Fichier : %s", downloaded_path)
    logging.info(
        "Taille : %.2f Mo",
        Path(downloaded_path).stat().st_size / (1024**2),
    )


if __name__ == "__main__":
    main()
