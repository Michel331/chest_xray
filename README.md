---
title: Pneumonie Demo
emoji: 🐨
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
---

# PneumoDetect AI

> Détection assistée de pneumonie sur radiographie thoracique — projet de fin de cycle.

**Démo en ligne** → [huggingface.co/spaces/MichelG29/pneumonie-demo](https://huggingface.co/spaces/MichelG29/pneumonie-demo)

---

## Contexte

Projet académique de fin de cycle. L'objectif est de concevoir un outil d'aide à la décision radiologique capable de détecter une pneumonie sur une radiographie thoracique, tout en quantifiant son incertitude pour rester utilisable dans un cadre clinique réaliste.

Le modèle est entraîné sur le jeu de données public **Chest X-Ray Images (Pneumonia)** de Kaggle (Paul Mooney), contenant environ 5 800 radiographies pédiatriques étiquetées `NORMAL` ou `PNEUMONIE`.

## Fonctionnalités

- **Classification binaire** `NORMAL` / `PNEUMONIE` via DenseNet121 fine-tuné
- **Carte d'attention Grad-CAM** pour visualiser les zones influentes de la décision
- **Incertitude épistémique** par MC Dropout (30 passes, méthode Gal & Ghahramani 2016)
- **Flag clinique automatique** : suggestion de second avis ou de croisement avec le tableau clinique selon le niveau d'incertitude
- **Interface web** progressive : prédiction rapide (~200 ms) affichée immédiatement, incertitude calculée en arrière-plan

## Architecture

```
Upload image
    │
    ▼
Preprocessing  (niveaux de gris → crop 10% → resize 224×224 → CLAHE → normalisation DenseNet)
    │
    ▼
DenseNet121 fine-tuné
    │
    ├── Prédiction déterministe (training=False)  ──►  Classe + probabilités
    ├── Grad-CAM                                  ──►  Heatmap superposée
    └── MC Dropout × 30 (training=True)           ──►  Incertitude + flag clinique
```

**Stack technique**
- Backend : FastAPI + TensorFlow / Keras 3
- Frontend : HTML + Tailwind CSS (CDN) + Cropper.js
- Déploiement : Docker → Hugging Face Spaces

## Installation locale

```bash
git clone https://github.com/Michel331/chest_xray.git
cd chest_xray/demo_pneumonie
pip install -r backend/requirements.txt
```

Placer le fichier de poids (`modele_pneumonie_densenet_balanced.keras`) à la racine du projet, ou définir la variable d'environnement `MODEL_PATH`.

Alternative : le modèle peut être téléchargé automatiquement depuis Hugging Face Hub en définissant `HF_REPO_ID` (et `HF_TOKEN` si le repo est privé).

## Lancement

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Puis ouvrir [http://localhost:8000](http://localhost:8000).

## API

| Méthode | Endpoint       | Description                                            |
|---------|----------------|--------------------------------------------------------|
| `GET`   | `/model/info`  | Métadonnées du modèle (architecture, seuils…)          |
| `POST`  | `/analyze`     | Prédiction + Grad-CAM (rapide, déterministe)           |
| `POST`  | `/uncertainty` | Incertitude MC Dropout + flag clinique (~5 s sur CPU)  |

## Structure du projet

```
chest_xray/
├── demo_pneumonie/
│   ├── backend/
│   │   ├── app.py              ← API FastAPI
│   │   ├── model_utils.py      ← Chargement modèle, Grad-CAM, MC Dropout
│   │   ├── preprocessing.py    ← Pipeline d'entrée
│   │   └── requirements.txt
│   ├── frontend/
│   │   └── index.html          ← Interface utilisateur
│   ├── Dockerfile
│   └── CAHIER_DES_CHARGES.md
├── code_pneumonie_fine_tuning.ipynb   ← Notebook d'entraînement
└── README.md
```

## Entraînement du modèle

Le notebook [`code_pneumonie_fine_tuning.ipynb`](./code_pneumonie_fine_tuning.ipynb) détaille la procédure complète : préparation du dataset, rééquilibrage des classes, fine-tuning de DenseNet121 pré-entraîné sur ImageNet, et évaluation sur le test set.

## Avertissement

Cet outil est un **prototype académique d'aide à la décision radiologique**. Il ne remplace en aucun cas l'avis d'un médecin ou d'un radiologue qualifié et n'est pas un dispositif médical certifié.

## Auteurs

Projet réalisé par cinq étudiants dans le cadre d'un projet de fin de cycle :

- Leila Toundji
- Bienvenu Arthur Elvis Houin — [@Elvis-bah](https://github.com/Elvis-bah)
- Ange Parfait Amonkou — [@angeparfait18](https://github.com/angeparfait18)
- Menelick Willia Alou — [@Will3007](https://github.com/Will3007)
- Michel Siba Gulavogui — [@Michel331](https://github.com/Michel331)

## Licence

Usage académique uniquement.
