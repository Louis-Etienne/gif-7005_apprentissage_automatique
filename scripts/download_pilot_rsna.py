from pathlib import Path
import subprocess
import zipfile
import sys
import pandas as pd


# Chemins relatifs à la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Choisir le fichier CSV à partir d'un argument du terminal
if len(sys.argv) != 2:
    raise SystemExit(
        "Utilisation : python scripts/download_pilot_rsna.py "
        "<chemin_du_csv>"
    )

PILOT_CSV = Path(sys.argv[1]).resolve()

if not PILOT_CSV.is_file():
    raise FileNotFoundError(
        f"Fichier CSV introuvable : {PILOT_CSV}"
    )
    
DATA_DIR = PROJECT_ROOT / "data/raw"

COMPETITION = "rsna-breast-cancer-detection"


def download_image(patient_id, image_id):
    """Télécharge et extrait une mammographie RSNA."""

    patient_id = int(patient_id)
    image_id = int(image_id)

    image_dir = DATA_DIR / "train_images" / str(patient_id)
    image_dir.mkdir(parents=True, exist_ok=True)

    dicom_path = image_dir / f"{image_id}.dcm"
    zip_path = image_dir / f"{image_id}.dcm.zip"

    # Ne pas télécharger une image déjà présente
    if dicom_path.exists():
        print(f"Déjà présente : {dicom_path.name}")
        return True

    kaggle_file = f"train_images/{patient_id}/{image_id}.dcm"

    # Télécharger seulement si l'archive est absente
    if not zip_path.exists():
        command = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-f",
            kaggle_file,
            "-p",
            str(DATA_DIR),
        ]

        subprocess.run(command, check=True)

    # Extraire l'archive si elle existe
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [
                member for member in archive.infolist()
                if not member.is_dir()
                and Path(member.filename).name == f"{image_id}.dcm"
            ]

            if len(members) != 1:
                raise ValueError(
                    f"Fichier DICOM introuvable ou ambigu : {zip_path}"
                )

            with archive.open(members[0]) as source:
                with dicom_path.open("wb") as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)

    if not dicom_path.exists():
        raise FileNotFoundError(
            f"Image introuvable après téléchargement : {dicom_path}"
        )

    print(f"OK : {patient_id}/{image_id}.dcm")
    return True


def main():
    pilot_df = pd.read_csv(PILOT_CSV)

    print(f"Nombre d'images à traiter : {len(pilot_df)}")

    success_count = 0

    for index, row in pilot_df.iterrows():
        print(f"\nImage {index + 1}/{len(pilot_df)}")

        try:
            if download_image(row["patient_id"], row["image_id"]):
                success_count += 1
        except Exception as error:
            print(f"ERREUR : {error}")

    print(f"\nTéléchargements réussis : {success_count}/{len(pilot_df)}")


if __name__ == "__main__":
    main()