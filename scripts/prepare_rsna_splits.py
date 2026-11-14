
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data/raw/train.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Charger les données RSNA
df = pd.read_csv(INPUT_CSV)

# Une ligne par patiente
patient_df = (
    df.groupby("patient_id")["cancer"]
    .max()
    .reset_index()
)

# 70 % entraînement, 30 % temporaire
train_patients, temp_patients = train_test_split(
    patient_df,
    test_size=0.30,
    random_state=42,
    stratify=patient_df["cancer"]
)

# 15 % validation, 15 % test
val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42,
    stratify=temp_patients["cancer"]
)

# Construire les partitions d'images
splits = {
    "train": train_patients,
    "validation": val_patients,
    "test": test_patients,
}

for name, patients in splits.items():
    subset = df[df["patient_id"].isin(patients["patient_id"])].copy()

    output_path = OUTPUT_DIR / f"{name}.csv"
    subset.to_csv(output_path, index=False)

    print(
        f"{name}: {len(subset)} images, "
        f"{subset['patient_id'].nunique()} patientes, "
        f"{subset['cancer'].sum()} images positives"
    )

# Vérifier l'absence de chevauchement des patientes
train_ids = set(train_patients["patient_id"])
val_ids = set(val_patients["patient_id"])
test_ids = set(test_patients["patient_id"])

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

print("\nVérification réussie : aucune patiente partagée.")
