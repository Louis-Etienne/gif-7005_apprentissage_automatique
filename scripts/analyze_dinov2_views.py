#!/usr/bin/env python3
"""Analyse exploratoire sur validation uniquement : vues et agrégation par sein.
Aucune lecture du test, aucun réentraînement. Les résultats agrégés sont exploratoires.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def summarize(df, threshold, name):
    y = df['cancer'].to_numpy(dtype=int)
    scores = df['probability_cancer'].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    return {
        'groupe': str(name), 'n': len(df), 'positifs': int(y.sum()),
        'prevalence': float(y.mean()),
        'auprc': float(average_precision_score(y, scores)) if y.sum() else None,
        'auroc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        'precision': float(tp / (tp + fp)) if tp + fp else 0.,
        'sensibilite': float(tp / (tp + fn)) if tp + fn else None,
        'specificite': float(tn / (tn + fp)) if tn + fp else None,
        'vrais_negatifs': int(tn), 'faux_positifs': int(fp),
        'faux_negatifs': int(fn), 'vrais_positifs': int(tp),
        'seuil': float(threshold),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, default=ROOT/'results/dinov2/validation_predictions.csv')
    p.add_argument('--metadata', type=Path, default=ROOT/'data/processed/splits/validation.csv')
    p.add_argument('--selection', type=Path, default=ROOT/'results/dinov2/selection.json')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/dinov2/views_analysis')
    args = p.parse_args()
    for f in (args.predictions, args.metadata, args.selection):
        if not f.is_file():
            raise FileNotFoundError(f)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(args.predictions, dtype={'patient_id': str, 'image_id': str})
    meta = pd.read_csv(args.metadata, dtype={'patient_id': str, 'image_id': str})
    required_pred = {'patient_id', 'image_id', 'probability_cancer'}
    if not required_pred.issubset(pred.columns):
        raise ValueError(f'Colonnes manquantes dans les prédictions : {required_pred-set(pred.columns)}')
    required_meta = {'patient_id', 'image_id', 'cancer', 'laterality', 'view'}
    if not required_meta.issubset(meta.columns):
        raise ValueError(f'Colonnes manquantes dans validation.csv : {required_meta-set(meta.columns)}. Vérifier le CSV original RSNA.')
    if pred.duplicated(['patient_id','image_id']).any() or meta.duplicated(['patient_id','image_id']).any():
        raise ValueError('Identifiants patient/image dupliqués')
    df = pred[['patient_id','image_id','probability_cancer']].merge(
        meta[['patient_id','image_id','cancer','laterality','view']],
        on=['patient_id','image_id'], how='outer', validate='one_to_one', indicator=True)
    if not (df['_merge'] == 'both').all():
        raise ValueError(f'Prédictions et validation.csv non alignés : {df._merge.value_counts().to_dict()}')
    df = df.drop(columns='_merge')
    if df[['cancer','probability_cancer','laterality','view']].isna().any().any():
        raise ValueError('Métadonnées ou prédictions manquantes')
    if not df['cancer'].isin([0,1]).all():
        raise ValueError('La colonne cancer doit être binaire')
    if not np.isfinite(df['probability_cancer']).all() or not df['probability_cancer'].between(0,1).all():
        raise ValueError('Scores non valides')
    df['laterality'] = df['laterality'].astype(str).str.upper().str.strip()
    df['view'] = df['view'].astype(str).str.upper().str.strip()
    df['vue_complete'] = df['laterality'] + '-' + df['view']
    threshold = float(json.loads(args.selection.read_text(encoding='utf-8'))['threshold'])

    print('\n=== DISTRIBUTION DES VUES ===', flush=True)
    distribution = df.groupby(['laterality','view'], dropna=False).agg(
        images=('cancer','size'), cancers=('cancer','sum'), patients=('patient_id','nunique')).reset_index()
    distribution['taux_cancer'] = distribution['cancers'] / distribution['images']
    distribution.to_csv(args.output_dir/'distribution_vues.csv', index=False)
    print(distribution.to_string(index=False), flush=True)

    rows = [summarize(df, threshold, 'toutes_images')]
    for view, sub in df.groupby('view'):
        rows.append(summarize(sub, threshold, f'vue_{view}'))
    for view, sub in df.groupby('vue_complete'):
        rows.append(summarize(sub, threshold, f'vue_{view}'))
    view_metrics = pd.DataFrame(rows)
    view_metrics.to_csv(args.output_dir/'metriques_par_vue.csv', index=False)
    print('\n=== METRIQUES PAR VUE (SEUIL ORIGINAL) ===', flush=True)
    print(view_metrics.to_string(index=False), flush=True)

    # RSNA: étiquette cancer au niveau du sein. Vérifier sa cohérence entre vues.
    group_cols = ['patient_id','laterality']
    consistency = df.groupby(group_cols)['cancer'].nunique()
    if (consistency > 1).any():
        raise ValueError(f'Étiquettes cancer contradictoires pour {(consistency > 1).sum()} seins')
    breast = df.groupby(group_cols).agg(
        cancer=('cancer','first'), n_images=('image_id','size'),
        n_vues=('view','nunique'),
        score_max=('probability_cancer','max'),
        score_moyen=('probability_cancer','mean'),
    ).reset_index()
    breast.to_csv(args.output_dir/'predictions_par_sein.csv', index=False)
    print('\n=== NOMBRE D’IMAGES PAR SEIN ===', flush=True)
    print(breast['n_images'].value_counts().sort_index().to_string(), flush=True)
    print('Seins avec plusieurs vues distinctes :', int((breast.n_vues >= 2).sum()), '/', len(breast), flush=True)

    # Même seuil image : comparaison exploratoire, PAS seuil optimisé pour les seins.
    breast_rows = []
    for aggregation in ('max','moyen'):
        tmp = breast[['cancer',f'score_{aggregation}']].rename(columns={f'score_{aggregation}':'probability_cancer'})
        breast_rows.append(summarize(tmp, threshold, f'sein_{aggregation}'))
    breast_metrics = pd.DataFrame(breast_rows)
    breast_metrics.to_csv(args.output_dir/'metriques_par_sein.csv', index=False)
    print('\n=== AGRÉGATION PAR SEIN (EXPLORATOIRE) ===', flush=True)
    print(breast_metrics.to_string(index=False), flush=True)

    plot_df = distribution.copy()
    plot_df['vue_complete'] = plot_df['laterality'] + '-' + plot_df['view']
    fig, ax = plt.subplots(figsize=(9,4.5))
    ax.bar(plot_df['vue_complete'], plot_df['images'])
    ax.set(ylabel="Nombre d'images", xlabel='Vue', title='Distribution des vues — validation')
    ax.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    fig.savefig(args.output_dir/'distribution_vues.png', dpi=160)
    plt.close(fig)
    print(f'\nFichiers enregistrés dans : {args.output_dir}', flush=True)
    print('ATTENTION : analyse exploratoire sur validation ; aucun résultat test calculé.', flush=True)


if __name__ == '__main__':
    main()
