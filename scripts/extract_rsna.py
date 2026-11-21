
"""
Extraction des mammographies RSNA depuis l'archive Kaggle.

Fonctionnalités :
- Extraction des DICOM d'entraînement uniquement
- Conservation des fichiers déjà extraits et valides
- Vérification de la taille et du CRC des fichiers extraits
- Reprise après interruption
- Journalisation de la progression

Exécution :
    python -u scripts/extract_rsna.py
"""

from pathlib import Path
from zipfile import ZipFile
import shutil
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = (
    PROJECT_ROOT
    / "data/raw/rsna-breast-cancer-detection.zip"
)

DESTINATION = PROJECT_ROOT / "data/raw"

EXPECTED_IMAGES = 54706
PROGRESS_INTERVAL = 500


def extract_rsna():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(
            f"Archive introuvable : {ZIP_PATH}"
        )

    extracted = 0
    skipped = 0
    start = time.time()

    print("=== Extraction RSNA ===", flush=True)
    print(f"Archive : {ZIP_PATH}", flush=True)
    print(f"Destination : {DESTINATION}", flush=True)

    with ZipFile(ZIP_PATH) as archive:
        dicoms = [
            info
            for info in archive.infolist()
            if info.filename.startswith("train_images/")
            and info.filename.endswith(".dcm")
        ]

        total = len(dicoms)
        print(f"Images trouvées : {total}", flush=True)

        if total != EXPECTED_IMAGES:
            raise RuntimeError(
                f"Nombre inattendu : {total}, "
                f"attendu : {EXPECTED_IMAGES}"
            )

        for index, info in enumerate(dicoms, start=1):
            target = DESTINATION / info.filename

            # Ne pas réextraire un fichier déjà présent
            # si sa taille correspond à celle de l'archive.
            if (
                target.is_file()
                and target.stat().st_size == info.file_size
            ):
                skipped += 1

            else:
                target.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                temporary = target.with_suffix(".dcm.part")

                try:
                    with archive.open(info) as source:
                        with open(temporary, "wb") as output:
                            shutil.copyfileobj(
                                source,
                                output,
                                length=1024 * 1024
                            )

                    if temporary.stat().st_size != info.file_size:
                        raise RuntimeError(
                            f"Taille incorrecte : {info.filename}"
                        )

                    temporary.replace(target)
                    extracted += 1

                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise

            if (
                index % PROGRESS_INTERVAL == 0
                or index == total
            ):
                elapsed = (time.time() - start) / 60

                print(
                    f"[{index}/{total}] "
                    f"Extraites : {extracted} | "
                    f"Existantes : {skipped} | "
                    f"Durée : {elapsed:.1f} min",
                    flush=True
                )

    print("\n=== Extraction terminée ===", flush=True)
    print(f"Nouvelles images : {extracted}")
    print(f"Images conservées : {skipped}")
    print(f"Total traité : {extracted + skipped}")


if __name__ == "__main__":
    extract_rsna()
