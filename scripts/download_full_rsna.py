
import argparse
import csv
import random
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/raw/train.csv"
OUTPUT_DIR = ROOT / "data/raw/train_images"
ERROR_LOG = ROOT / "data/processed/rsna_download_errors.csv"

COMPETITION = "rsna-breast-cancer-detection"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0)
    return parser.parse_args()


def valid_dicom(path):
    if not path.is_file() or path.stat().st_size < 132:
        return False

    # Vérification rapide de la signature DICOM.
    # Certains DICOM valides n'ont pas de préambule.
    with path.open("rb") as f:
        f.seek(128)
        signature = f.read(4)

    return signature == b"DICM"


def download_image(patient_id, image_id):
    relative_path = (
        f"train_images/{patient_id}/{image_id}.dcm"
    )

    destination = (
        OUTPUT_DIR / str(patient_id) / f"{image_id}.dcm"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)

    if valid_dicom(destination):
        return "skipped"

    with tempfile.TemporaryDirectory(
        prefix="rsna_",
        dir=ROOT / "data/raw",
    ) as temp:
        temp_dir = Path(temp)

        command = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            relative_path,
            "-p",
            str(temp_dir),
            "-q",
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip()
            )

        candidates = list(temp_dir.rglob("*.dcm"))

        if candidates:
            source = candidates[0]
            shutil.move(str(source), str(destination))
        else:
            archives = list(temp_dir.rglob("*.zip"))

            if not archives:
                raise FileNotFoundError(
                    "Aucun fichier DICOM ou ZIP téléchargé."
                )

            found = False

            for archive in archives:
                with zipfile.ZipFile(archive) as zf:
                    members = [
                        name for name in zf.namelist()
                        if name.endswith(f"/{image_id}.dcm")
                        or name == f"{image_id}.dcm"
                    ]

                    if members:
                        with zf.open(members[0]) as source:
                            with destination.open("wb") as target:
                                shutil.copyfileobj(source, target)
                        found = True
                        break

            if not found:
                raise FileNotFoundError(
                    f"Image {image_id} absente de l'archive."
                )

    if not valid_dicom(destination):
        destination.unlink(missing_ok=True)
        raise ValueError("Fichier DICOM invalide ou incomplet.")

    return "downloaded"


def main():
    args = parse_args()

    if args.max_retries < 1 or args.delay < 0:
        raise ValueError("Paramètres de téléchargement invalides.")

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(
        CSV_PATH,
        usecols=["patient_id", "image_id"],
        dtype=str,
    ).drop_duplicates()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)

    total = len(df)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"Images à traiter : {total}", flush=True)
    print(f"Destination : {OUTPUT_DIR}", flush=True)

    with ERROR_LOG.open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["patient_id", "image_id", "error"])

        for index, row in enumerate(df.itertuples(index=False), 1):
            patient_id = row.patient_id
            image_id = row.image_id

            last_error = None
            success = False

            for attempt in range(args.max_retries):
                try:
                    status = download_image(patient_id, image_id)

                    if status == "skipped":
                        skipped += 1
                    else:
                        downloaded += 1
                        time.sleep(args.delay)

                    success = True
                    break

                except Exception as error:
                    last_error = str(error)

                    wait = min(
                        30 * (2 ** attempt) + random.uniform(0, 5),
                        900,
                    )

                    print(
                        f"Erreur {patient_id}/{image_id} : "
                        f"{last_error}. Attente {wait:.0f}s",
                        flush=True,
                    )

                    time.sleep(wait)

            if not success:
                failed += 1
                writer.writerow(
                    [patient_id, image_id, last_error]
                )
                log.flush()

            if index % 100 == 0 or index == total:
                print(
                    f"[{index}/{total}] "
                    f"Téléchargées : {downloaded} | "
                    f"Existantes : {skipped} | "
                    f"Échecs : {failed}",
                    flush=True,
                )

    print("\n===== TÉLÉCHARGEMENT TERMINÉ =====")
    print(f"Téléchargées : {downloaded}")
    print(f"Déjà présentes : {skipped}")
    print(f"Échecs : {failed}")
    print(f"Journal des erreurs : {ERROR_LOG}")


if __name__ == "__main__":
    main()
