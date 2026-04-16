# PneumoDetect AI — Démo soutenance

Application web de détection de pneumonie sur radiographies thoraciques.  
Modèle : DenseNet121 · Grad-CAM · MC Dropout

## Lancement local (3 commandes)

```bash
cd demo_pneumonie
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Ouvrir : http://localhost:8000

## Structure

```
demo_pneumonie/
├── backend/
│   ├── app.py              # API FastAPI
│   ├── model_utils.py      # Grad-CAM, MC Dropout, incertitude
│   ├── preprocessing.py    # Source unique de vérité du preprocessing
│   └── requirements.txt
├── frontend/
│   └── index.html          # UI complète (Tailwind CDN)
├── data/
│   └── demo_cases/         # 4 images de démo pré-sélectionnées
└── modele_pneumonie_densenet.keras
```

## Variables d'environnement

| Variable      | Description                                      | Défaut                          |
|---------------|--------------------------------------------------|---------------------------------|
| `MODEL_PATH`  | Chemin local vers le fichier `.keras`            | auto-détecté                    |
| `HF_REPO_ID`  | Repo Hugging Face Hub (`user/nom-repo`)          | non défini → cherche en local   |

## Déploiement Hugging Face Spaces

1. Créer un repo modèle sur HF Hub et y déposer `modele_pneumonie_densenet.keras`
2. Créer un Space Docker, y copier ce répertoire
3. Définir `HF_REPO_ID=user/nom-repo` dans les secrets du Space
4. Port : **7860** (configuré dans le Dockerfile)

## Cas de démo

Placer 4 images dans `data/demo_cases/` :
1. `pneumonie_evidente.jpg`  — confiance > 90%, incertitude faible
2. `normal_net.jpg`          — confiance élevée, Grad-CAM diffuse
3. `cas_ambigu.jpg`          — incertitude moyenne/haute, flag clinique
4. `cas_difficile.jpg`       — montre que le modèle sait qu'il ne sait pas
