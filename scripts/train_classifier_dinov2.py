#!/usr/bin/env python3
"""RSNA: linear probe DINOv2 embeddings, patient-disjoint train/validation.

Default: model selection on validation, NO test evaluation.
Optional --evaluate-test: one final locked-model test evaluation.
"""
import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score, fbeta_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger('dinov2_probe')


def parse_args():
    p = argparse.ArgumentParser(description='RSNA DINOv2 linear probe: train/validation, optional final test.')
    p.add_argument('--embedding-root', type=Path, default=ROOT/'data/processed/embeddings/dinov2')
    p.add_argument('--split-root', type=Path, default=ROOT/'data/processed/splits')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/dinov2')
    p.add_argument('--evaluate-test', action='store_true', help='Run final test ONCE after validation selection')
    return p.parse_args()


def load_split(split, split_root, embedding_root):
    csv_file = split_root/f'{split}.csv'
    if not csv_file.is_file():
        raise FileNotFoundError(f'CSV introuvable : {csv_file}')
    df = pd.read_csv(csv_file, dtype={'patient_id': str, 'image_id': str})
    required = {'patient_id', 'image_id', 'cancer'}
    if not required.issubset(df.columns):
        raise ValueError(f'{csv_file} doit contenir {sorted(required)} ; colonnes : {list(df.columns)}')
    if df[['patient_id', 'image_id', 'cancer']].isna().any().any():
        raise ValueError(f'Valeurs manquantes dans {csv_file}')
    if df.duplicated(['patient_id', 'image_id']).any():
        raise ValueError(f'Doublons patient_id/image_id dans {csv_file}')
    y = df['cancer'].to_numpy(dtype=np.int64)
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f'Valeurs cancer autres que 0/1 dans {csv_file}')
    X = np.empty((len(df), 384), dtype=np.float32)
    missing, invalid = [], []
    folder = embedding_root/split
    for i, row in enumerate(df.itertuples(index=False)):
        path = folder/f'{row.patient_id}_{row.image_id}.npy'
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            a = np.load(path, allow_pickle=False)
            if a.shape == (1, 384):
                a = a[0]
            if a.shape != (384,) or not np.isfinite(a).all():
                raise ValueError(f'shape={a.shape} ou valeurs non finies')
            X[i] = a
        except Exception as exc:
            invalid.append(f'{path}: {exc}')
        if (i+1) % 10000 == 0:
            LOG.info('%s : %d/%d embeddings lus', split, i+1, len(df))
    if missing or invalid:
        raise RuntimeError(f'{split}: {len(missing)} manquants, {len(invalid)} invalides. '
                           f'Exemples : {(missing+invalid)[:5]}')
    LOG.info('%s : %d images, %d cancers (%.2f%%), %d patients',
             split, len(df), int(y.sum()), 100*y.mean(), df.patient_id.nunique())
    return df, X, y


def check_patient_disjoint(*frames):
    sets = [set(df.patient_id) for df in frames]
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise ValueError(f'Fuite patients entre partitions {i} et {j}: {len(overlap)}')


def metrics_at(y, probs, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(threshold=float(threshold), n=int(len(y)), positives=int(y.sum()),
                auroc=float(roc_auc_score(y, probs)),
                auprc=float(average_precision_score(y, probs)),
                accuracy=float(accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)),
                sensitivity_recall=float(recall_score(y, pred, zero_division=0)),
                specificity=float(tn/(tn+fp)) if tn+fp else None,
                f1=float(f1_score(y, pred, zero_division=0)),
                f2=float(fbeta_score(y, pred, beta=2, zero_division=0)),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]])


def choose_threshold(y, probs):
    # Select threshold maximizing F2 on validation only; tie -> highest threshold.
    precision, recall, thresholds = precision_recall_curve(y, probs)
    if len(thresholds) == 0:
        return 0.5
    p, r = precision[:-1], recall[:-1]
    f2 = np.divide(5*p*r, 4*p+r, out=np.zeros_like(p), where=(4*p+r)>0)
    best = np.flatnonzero(np.isclose(f2, f2.max(), rtol=1e-10, atol=1e-12))
    return float(thresholds[best[-1]])


def save_plots(y, probs, threshold, output_dir, prefix):
    fpr, tpr, _ = roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, label=f'AUROC = {roc_auc_score(y, probs):.4f}')
    ax.plot([0,1], [0,1], '--', label='Hasard')
    ax.set(xlabel='Taux de faux positifs', ylabel='Sensibilité', title=f'ROC — {prefix}')
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir/f'{prefix}_roc.png', dpi=160); plt.close(fig)
    prec, rec, _ = precision_recall_curve(y, probs)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(rec, prec, label=f'AUPRC = {average_precision_score(y, probs):.4f}')
    ax.axhline(y.mean(), linestyle='--', label=f'Prévalence = {y.mean():.4f}')
    ax.set(xlabel='Sensibilité (Recall)', ylabel='Précision', title=f'Précision–rappel — {prefix}', xlim=(0,1), ylim=(0,1))
    ax.legend(); fig.tight_layout(); fig.savefig(output_dir/f'{prefix}_pr.png', dpi=160); plt.close(fig)
    cm = confusion_matrix(y, probs >= threshold, labels=[0,1])
    fig, ax = plt.subplots(figsize=(5,4))
    ax.imshow(cm, cmap='Blues')
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center', color='black')
    ax.set(xticks=[0,1], yticks=[0,1], xticklabels=['0','1'], yticklabels=['0','1'],
           xlabel='Prédiction', ylabel='Vérité', title=f'Confusion — {prefix}')
    fig.tight_layout(); fig.savefig(output_dir/f'{prefix}_confusion.png', dpi=160); plt.close(fig)


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir/'training.log', encoding='utf-8')])
    df_train, X_train, y_train = load_split('train', args.split_root, args.embedding_root)
    df_val, X_val, y_val = load_split('validation', args.split_root, args.embedding_root)
    check_patient_disjoint(df_train, df_val)
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError('Train et validation doivent contenir les deux classes.')

    candidates = []
    best = None
    # C: inverse de la régularisation. balanced compense la rareté de cancer=1.
    for class_weight in ('balanced', None):
        for C in (0.001, 0.01, 0.1, 1.0, 10.0):
            LOG.info('Entraînement : C=%s, class_weight=%s', C, class_weight)
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('classifier', LogisticRegression(C=C, class_weight=class_weight,
                                                               max_iter=3000, solver='lbfgs', random_state=42))])
            pipe.fit(X_train, y_train)
            probs = pipe.predict_proba(X_val)[:,1]
            ap = float(average_precision_score(y_val, probs))
            auc = float(roc_auc_score(y_val, probs))
            record = dict(C=C, class_weight=str(class_weight), val_auprc=ap, val_auroc=auc,
                          iterations=int(pipe.named_steps['classifier'].n_iter_[0]))
            candidates.append(record)
            LOG.info('Validation : AUPRC=%.5f AUROC=%.5f', ap, auc)
            # Tie-break AUROC, then earlier/simpler candidate.
            if best is None or (ap, auc) > (best['ap'], best['auc']):
                best = dict(model=pipe, probs=probs, ap=ap, auc=auc, config=record)

    pd.DataFrame(candidates).to_csv(args.output_dir/'hyperparameter_search.csv', index=False)
    threshold = choose_threshold(y_val, best['probs'])
    val_metrics = metrics_at(y_val, best['probs'], threshold)
    joblib.dump(best['model'], args.output_dir/'model.joblib')
    df_val.assign(probability_cancer=best['probs'], predicted_cancer=(best['probs']>=threshold).astype(int)).to_csv(
        args.output_dir/'validation_predictions.csv', index=False)
    save_json(args.output_dir/'selection.json', dict(model='facebook/dinov2-small',
              method='frozen embeddings + standardized logistic regression',
              selection_metric='validation AUPRC', threshold_selection='validation F2 maximum',
              best_config=best['config'], threshold=threshold,
              validation=val_metrics, train_images=len(y_train), validation_images=len(y_val)))
    save_plots(y_val, best['probs'], threshold, args.output_dir, 'validation')
    LOG.info('MEILLEUR MODÈLE : %s', best['config'])
    LOG.info('SEUIL validation F2 : %.6f', threshold)
    LOG.info('MÉTRIQUES validation : %s', json.dumps(val_metrics, ensure_ascii=False))

    if args.evaluate_test:
        LOG.warning('ÉVALUATION FINALE TEST : ne pas réutiliser les résultats pour ajuster le modèle.')
        df_test, X_test, y_test = load_split('test', args.split_root, args.embedding_root)
        check_patient_disjoint(df_train, df_val, df_test)
        probs_test = best['model'].predict_proba(X_test)[:,1]
        test_metrics = metrics_at(y_test, probs_test, threshold)
        save_json(args.output_dir/'test_metrics.json', test_metrics)
        df_test.assign(probability_cancer=probs_test,
                       predicted_cancer=(probs_test>=threshold).astype(int)).to_csv(
                           args.output_dir/'test_predictions.csv', index=False)
        save_plots(y_test, probs_test, threshold, args.output_dir, 'test')
        LOG.info('MÉTRIQUES TEST : %s', json.dumps(test_metrics, ensure_ascii=False))
    LOG.info('Terminé. Résultats : %s', args.output_dir)


if __name__ == '__main__':
    main()
