
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError


# ============================================================
# CONFIGURATION
# ============================================================

COMPETITION = "rsna-breast-cancer-detection"
PAGE_SIZE = 200
DELAY_BETWEEN_REQUESTS = 3
MAX_RETRIES = 8

CHECKPOINT = Path(
    "data/processed/rsna_storage_progress.json"
)


# ============================================================
# INITIALISATION
# ============================================================

api = KaggleApi()
api.authenticate()

CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

if CHECKPOINT.exists():
    state = json.loads(CHECKPOINT.read_text())

    print("Reprise d'une exécution précédente.")
    print(f"Pages déjà traitées : {state['page_number']}")
    print(f"Images déjà recensées : {state['total_images']}")
    print(
        f"Taille déjà calculée : "
        f"{state['total_bytes'] / 1024**3:.2f} Gio"
    )

else:
    state = {
        "page_token": None,
        "total_bytes": 0,
        "total_images": 0,
        "page_number": 0,
    }

    print("Nouvelle analyse du jeu RSNA.")


# ============================================================
# FONCTION : APPEL API AVEC RETENTATIVES
# ============================================================

def fetch_page(page_token):
    for attempt in range(MAX_RETRIES):
        try:
            return api.competition_list_files(
                COMPETITION,
                page_size=PAGE_SIZE,
                page_token=page_token,
            )

        except HTTPError as error:
            if (
                error.response is None
                or error.response.status_code != 429
            ):
                raise

            if attempt == MAX_RETRIES - 1:
                print(
                    "Nombre maximal de tentatives atteint. "
                    "La progression est conservée."
                )
                raise

            retry_after = error.response.headers.get(
                "Retry-After"
            )

            if retry_after and retry_after.isdigit():
                wait_seconds = max(
                    int(retry_after),
                    60 * (2 ** attempt),
                )
            else:
                wait_seconds = min(
                    60 * (2 ** attempt),
                    900,
                )

            print(
                f"Erreur 429 : limitation Kaggle. "
                f"Nouvelle tentative dans "
                f"{wait_seconds} secondes.",
                flush=True,
            )

            time.sleep(wait_seconds)


# ============================================================
# PARCOURS DES FICHIERS
# ============================================================

while True:

    response = fetch_page(state["page_token"])

    for file in response.files:

        if (
            file.name.startswith("train_images/")
            and file.name.endswith(".dcm")
        ):
            state["total_images"] += 1
            state["total_bytes"] += int(file.total_bytes)

    state["page_number"] += 1

    next_token = getattr(
        response,
        "next_page_token",
        None,
    )

    if (
        next_token is not None
        and next_token == state["page_token"]
    ):
        raise RuntimeError(
            "La pagination Kaggle n'avance plus."
        )

    state["page_token"] = next_token

    # Sauvegarde après chaque page
    temporary_checkpoint = CHECKPOINT.with_suffix(".tmp")
    temporary_checkpoint.write_text(
        json.dumps(state, indent=2)
    )
    temporary_checkpoint.replace(CHECKPOINT)

    # Affichage périodique
    if state["page_number"] % 25 == 0:
        print(
            f"Pages : {state['page_number']} | "
            f"Images : {state['total_images']} | "
            f"Taille : "
            f"{state['total_bytes'] / 1024**3:.2f} Gio",
            flush=True,
        )

    if not next_token:
        break

    # Pause pour réduire les risques de limitation API
    time.sleep(DELAY_BETWEEN_REQUESTS)


# ============================================================
# RÉSULTATS FINAUX
# ============================================================

print("\n" + "=" * 50)
print("RÉSULTAT FINAL — RSNA")
print("=" * 50)

print(f"Pages parcourues : {state['page_number']}")
print(f"Nombre de mammographies : {state['total_images']}")

print(
    f"Taille totale : "
    f"{state['total_bytes'] / 1024**3:.2f} Gio"
)

print(
    f"Taille totale : "
    f"{state['total_bytes'] / 1000**3:.2f} Go"
)

EXPECTED_IMAGES = 54706

if state["total_images"] == EXPECTED_IMAGES:
    print("\nVérification réussie : 54 706 images recensées.")
else:
    print(
        f"\nAttention : {state['total_images']} images "
        f"recensées sur {EXPECTED_IMAGES} attendues."
    )

print("=" * 50)
