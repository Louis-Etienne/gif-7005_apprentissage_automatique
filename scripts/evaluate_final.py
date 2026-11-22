#!/usr/bin/env python3
"""Evaluation finale RSNA: modèles et seuils figés sur validation, sans entraînement."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('dinov2', 'dinov3', 'mammofm', 'mammoclip')
SPLITS = ROOT / 'data/processed/splits'
OUT = ROOT / 'results/final_evaluation'


def main():
    split = pd.read_csv(SPLITS / 'test.csv', dtype={'patient_id': str, 'image_id': str})
    assert {'patient_id', 'image_id', 'cancer'}.issubset(split.columns)
    if split[['patient_id', 'image_id', 'cancer']].isna().any().any():
        raise ValueError('Valeurs manquantes dans test.csv')
    if split.duplicated(['patient_id', 'image_id']).any():
        raise ValueError('Images dupliquées dans test.csv')
    y = split.cancer.to_numpy(dtype=int)
    if len(split) != 8181 or int(y.sum()) != 177 or not np.isin(y, [0, 1]).all():
        raise ValueError(f'Test inattendu : n={len(split)}, positifs={int(y.sum())}')
    ids = list(zip(split.patient_id, split.image_id))
    records, predictions = [], {}
    for name in MODELS:
        print(f'\n=== {name} ===', flush=True)
        base = ROOT / 'results' / name
        config = json.loads((base / 'selection.json').read_text(encoding='utf-8'))
        threshold = float(config['threshold'])
        if not (0 < threshold < 1):
            raise ValueError(f'Seuil invalide pour {name}: {threshold}')
        pipe = joblib.load(base / 'model.joblib')  # fichiers locaux fiables uniquement
        if not hasattr(pipe, 'predict_proba') or 'scaler' not in pipe.named_steps:
            raise ValueError(f'Pipeline inattendu pour {name}')
        dim = int(pipe.n_features_in_)
        folder = ROOT / 'data/processed/embeddings' / name / 'test'
        X = np.empty((len(split), dim), dtype=np.float32)
        for i, (patient, image) in enumerate(ids):
            path = folder / f'{patient}_{image}.npy'
            arr = np.load(path, allow_pickle=False)
            if arr.shape == (1, dim):
                arr = arr[0]
            if arr.shape != (dim,) or not np.isfinite(arr).all():
                raise ValueError(f'Embedding invalide : {path} ; shape={arr.shape}, attendu={dim}')
            X[i] = arr
            if (i + 1) % 2000 == 0:
                print(f'  embeddings : {i+1}/{len(split)}', flush=True)
        probs = pipe.predict_proba(X)[:, list(pipe.classes_).index(1)]
        pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
        metrics = {
            'model': name, 'n': len(y), 'positives': int(y.sum()),
            'threshold': threshold,
            'auprc': float(average_precision_score(y, probs)),
            'auroc': float(roc_auc_score(y, probs)),
            'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'sensitivity_recall': float(recall_score(y, pred, zero_division=0)),
            'specificity': tn / (tn + fp),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'f2': float(fbeta_score(y, pred, beta=2, zero_division=0)),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        }
        records.append(metrics)
        predictions[name] = (probs, pred)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    # Écriture uniquement après réussite de toutes les évaluations
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (probs, pred) in predictions.items():
        split.assign(probability_cancer=probs, predicted_cancer=pred).to_csv(
            OUT / f'{name}_test_predictions.csv', index=False)
    pd.DataFrame(records).to_csv(OUT / 'comparison_test.csv', index=False)
    (OUT / 'comparison_test.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSUCCÈS : résultats finaux dans {OUT}', flush=True)


if __name__ == '__main__':
    main()
