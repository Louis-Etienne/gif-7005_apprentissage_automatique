from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pydicom
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# 1. Définir les chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if len(sys.argv) != 3:
    raise SystemExit(
        "Utilisation : python scripts/extract_embeddings_dinov2.py "
        "<fichier_csv> <train|validation>"
    )

PILOT_CSV = Path(sys.argv[1]).resolve()
dataset_name = sys.argv[2]

if dataset_name not in ("train", "validation"):
    raise ValueError(
        "Le jeu de données doit être 'train' ou 'validation'."
    )

if not PILOT_CSV.is_file():
    raise FileNotFoundError(
        f"Fichier CSV introuvable : {PILOT_CSV}"
    )

IMAGE_DIR = PROJECT_ROOT / "data/raw/train_images"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/embeddings/dinov2"
    / dataset_name
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Charger DINOv2 une seule fois
device = torch.device("cpu")
model_name = "facebook/dinov2-small"

print("Chargement de DINOv2...")

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

print("Modèle chargé.")


# 3. Convertir un fichier DICOM en image RGB
def prepare_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path)

    image = dicom.pixel_array.astype(np.float32)

    # Inverser les intensités si nécessaire
    if dicom.get("PhotometricInterpretation") == "MONOCHROME1":
        image = image.max() - image

    # Normaliser les pixels entre 0 et 1
    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)

    # Convertir en image PIL
    image_uint8 = (image * 255).round().astype(np.uint8)
    pil_image = Image.fromarray(image_uint8)

    # Réduire la taille avant le prétraitement DINOv2
    pil_image.thumbnail((518, 518), Image.Resampling.LANCZOS)

    return pil_image.convert("RGB")


# 4. Extraire l'embedding
def extract_embedding(rgb_image):
    inputs = processor(
        images=rgb_image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model(**inputs)

        # Représentation du token CLS
        embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.cpu().numpy()


# 5. Parcourir les 40 mammographies
pilot_df = pd.read_csv(PILOT_CSV)

success_count = 0
errors = []

for index, row in pilot_df.iterrows():

    patient_id = int(row["patient_id"])
    image_id = int(row["image_id"])

    dicom_path = (
        IMAGE_DIR
        / str(patient_id)
        / f"{image_id}.dcm"
    )

    output_path = (
        OUTPUT_DIR
        / f"{patient_id}_{image_id}.npy"
    )

    print(
        f"[{index + 1}/{len(pilot_df)}] "
        f"Patient {patient_id}, image {image_id}",
        flush=True
    )

    try:
        if output_path.exists():
            existing = np.load(output_path)

            if existing.shape == (1, 384) and np.isfinite(existing).all():
                print("  Embedding déjà présent.")
                success_count += 1
                continue

        rgb_image = prepare_image(dicom_path)

        embedding = extract_embedding(rgb_image)

        if embedding.shape != (1, 384):
            raise ValueError(
                f"Dimensions inattendues : {embedding.shape}"
            )

        if not np.isfinite(embedding).all():
            raise ValueError("L'embedding contient des valeurs non finies.")

        np.save(output_path, embedding)

        print(f"  Embedding enregistré : {embedding.shape}")

        success_count += 1

    except Exception as error:
        print(f"  ERREUR : {error}")
        errors.append((patient_id, image_id, str(error)))


print("\nExtraction terminée.")
print(f"Embeddings disponibles : {success_count}/{len(pilot_df)}")
print(f"Erreurs : {len(errors)}")

for error in errors:
    print(error)