
# GIF-7005 — Classification de mammographies avec des modèles de fondation

## Présentation

Projet de fin de session du cours GIF-7005,
Introduction à l'apprentissage automatique,
Université Laval — Automne 2026.

L'objectif est d'étudier l'influence du domaine de
préentraînement sur les performances de classification
en mammographie.

## Question de recherche

Le préentraînement spécialisé en mammographie
améliore-t-il la classification bénin/malin par rapport
à un préentraînement généraliste ?

## Modèles envisagés

### Modèles spécialisés
- Mammo-FM
- Mammo-CLIP

### Modèles généralistes
- DINOv2
- DINOv3 (facultatif)

## Jeux de données envisagés

- RSNA Breast Cancer Detection
- VinDr-Mammo

Le choix du jeu de données et la définition exacte
des classes restent à confirmer.

## Méthodologie envisagée

1. Étude de la littérature scientifique.
2. Exploration et préparation des données.
3. Extraction des représentations avec les modèles.
4. Entraînement de classifieurs.
5. Évaluation comparative des performances.
6. Analyse et interprétation des résultats.

## Structure du dépôt

- `data/` : données locales, exclues de Git
- `notebooks/` : notebooks Jupyter
- `src/` : code Python réutilisable
- `configs/` : configuration des expériences
- `results/` : résultats et graphiques
- `docs/` : documentation scientifique
- `affiche/` : livrable final

## Équipe

Projet réalisé par une équipe de quatre étudiants.

## Statut

Phase de familiarisation avec la littérature,
les jeux de données et les modèles.
