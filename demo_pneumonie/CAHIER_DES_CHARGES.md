# Cahier des charges — Démo soutenance : détection de pneumonie

## Contexte

Application web d'aide à la décision radiologique pour la détection de pneumonie sur radiographies thoraciques. Démo live devant un jury académique.

Le modèle existe déjà (DenseNet121, entraîné). Le code Grad-CAM existe déjà. Le travail consiste à déployer le tout dans une app web fonctionnelle.

## Ce que le modèle produit

À partir d'une seule image (radio thoracique JPEG/PNG), on extrait **3 résultats** :

1. **Classification** : NORMAL ou PNEUMONIA, avec probabilité softmax (0–100%)
2. **Heatmap Grad-CAM** : image montrant les zones qui ont influencé la décision du modèle
3. **Score d'incertitude MC Dropout** : le modèle est-il sûr de lui ? (à implémenter)

Plus une **règle métier** simple :
- Incertitude haute → afficher "Second avis radiologique recommandé"
- Incertitude moyenne + confiance < 75% → afficher "À croiser avec le tableau clinique"

---

## Spécifications du modèle (NE PAS MODIFIER)

- **Architecture** : DenseNet121 pré-entraîné (ImageNet) + tête custom (GAP → Dropout 0.4 → Dense 128 relu → Dropout 0.3 → Dense 2 softmax)
- **Fichier** : `modele_pneumonie_densenet.keras`
- **Input** : tenseur (1, 224, 224, 3) float32
- **Output** : tenseur (1, 2) softmax — index 0 = NORMAL, index 1 = PNEUMONIA
- **Classes** : `{'NORMAL': 0, 'PNEUMONIA': 1}`

### Preprocessing EXACT (respecter à la lettre)

Le preprocessing est critique. Une erreur ici = modèle inutilisable.

```
1. Charger l'image
2. Convertir en grayscale
3. Crop centré 10% (supprimer 10% de chaque bord)
4. Resize 224×224 (interpolation INTER_AREA)
5. CLAHE : skimage.exposure.equalize_adapthist(gray, clip_limit=0.02)
6. Convertir en uint8 : (résultat_clahe * 255).astype(uint8)
7. Dupliquer en 3 canaux : np.stack([gray_u8]*3, axis=-1)
8. Appliquer densenet_preprocess : tensorflow.keras.applications.densenet.preprocess_input()
9. Ajouter dimension batch : (1, 224, 224, 3)
```

La fonction `preprocess_xray` du notebook (`code_pneumonie_softmax_gan.ipynb`, cellule 2) fait référence.

### Règle importante : source unique de vérité

Le preprocessing **doit être dans un module Python unique** (ex: `backend/preprocessing.py`) importé aussi bien par le backend que par tout script de test. Pas de copier-coller du code depuis le notebook — importer directement la fonction.

Justification : le CLAHE utilise `skimage` qui n'est pas du TensorFlow natif, donc il n'est pas intégrable proprement comme couche du modèle. La fonction Python reste séparée, mais elle doit être **la seule version existante** pour éviter toute divergence entre entraînement et inférence (training/serving skew — cause n°1 des bugs ML en production).

### Test de cohérence (recommandé)

Ajouter un test simple qui compare le tenseur produit par la fonction de preprocessing sur une image de référence, avec un tenseur attendu sauvegardé. Ça permet de détecter immédiatement toute régression du preprocessing.

### Grad-CAM (EXISTE DÉJÀ)

Le code complet est dans le notebook, cellule 15. Fonctions :
- `make_gradcam_heatmap(img_input, model, pred_index)` → heatmap numpy 2D
- `display_gradcam(img_path, model, class_names)` → affichage matplotlib

Le code gère automatiquement la détection de la dernière couche conv dans DenseNet121 (récursif, cherche dans le sous-modèle). Ne pas réécrire, adapter pour renvoyer les données au lieu d'afficher.

### MC Dropout (À IMPLÉMENTER)

Le modèle a 2 couches Dropout (0.4 et 0.3). Pour le MC Dropout :
- Faire N inférences (N=30 par défaut) en forçant `training=True` sur les Dropout
- Calculer : moyenne des probabilités, écart-type par classe, entropie prédictive
- Classifier le niveau d'incertitude : low / medium / high (seuils à calibrer sur le test set)

Référence théorique : Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016.

---

## Architecture de l'application

```
radio.jpg  →  [FastAPI backend]  →  JSON + image
                    ↑
              [modèle .keras]
                    ↓
             [page web frontend]  →  affichage visuel
```

### Backend : FastAPI

Un seul fichier `app.py` suffit (+ un `model_utils.py` pour la logique modèle).

**Endpoints :**

| Méthode | Route         | Entrée                  | Sortie                                                         |
|---------|---------------|-------------------------|----------------------------------------------------------------|
| GET     | `/`           | —                       | Info API + disclaimer                                          |
| POST    | `/analyze`    | image (multipart)       | JSON : classe, confiance, heatmap base64, incertitude, flag    |

Un seul endpoint POST principal (`/analyze`). Pas besoin de séparer predict/gradcam/uncertainty — pour la démo, on veut tout d'un coup.

On peut ajouter `GET /model/info` pour afficher les métadonnées dans le frontend.

**Contraintes :**
- Charger le modèle une seule fois au démarrage (pas à chaque requête)
- CORS activé (frontend servi en statique)
- Le disclaimer médical doit être inclus dans chaque réponse JSON
- Servir le frontend en fichiers statiques depuis FastAPI

**Lancement** : `uvicorn app:app --port 8000`

### Frontend : HTML + Tailwind + JS (un seul fichier)

Pas de React, pas de npm, pas de build. Un `index.html` unique servi par FastAPI.

**Layout** (3 colonnes sur desktop) :

| Colonne gauche          | Colonne centre            | Colonne droite               |
|------------------------|---------------------------|------------------------------|
| Upload (drag & drop)    | Heatmap Grad-CAM          | Résultat + incertitude       |
| Aperçu image            | Image originale / overlay | Barres de probabilité        |
| Bouton "Analyser"       |                           | Badge incertitude (couleur)  |
| Slider N tirages MC     |                           | Flag clinique si applicable  |

**Design soigné** :
- Fond sombre (slate/dark), cartes blanches avec ombre
- Badge d'incertitude coloré : vert (low), orange (medium), rouge (high)
- Barres de progression pour les probabilités NORMAL/PNEUMONIA
- Disclaimer médical visible en permanence en haut de page (bandeau ambre)
- Header avec nom de l'app + statut API (point vert = connecté)
- Section "Contexte du dataset" en bas (stats train/test pour le storytelling)

**Responsive** : pas prioritaire, ça tourne sur un laptop devant le jury.

---

## Panneau "Contexte du dataset" (en bas du frontend)

Afficher les stats du dataset pour le storytelling soutenance :

| Split      | NORMAL | PNEUMONIA | Total |
|------------|--------|-----------|-------|
| Train      | 1 341  | 3 875     | 5 216 |
| Test       | 234    | 390       | 624   |
| Validation | 8      | 8         | 16    |

Mettre en évidence :
- Déséquilibre de classes (~1:2.9) — mention rouge
- Validation set trop petit (16 images) — mention rouge

---

## Cas de démo à préparer

Sélectionner 4 images du test set et les mettre dans `data/demo_cases/` :

1. **Pneumonie évidente** → confiance > 90%, incertitude low, Grad-CAM concentrée
2. **Normal net** → confiance élevée, incertitude low, Grad-CAM diffuse
3. **Cas ambigu** → confiance moyenne, incertitude medium/high, flag clinique affiché
4. **Cas difficile** → pour montrer que le modèle sait quand il ne sait pas

Les tester tous avant la soutenance. Avoir des screenshots de secours en cas de panne.

---

## Arborescence cible

```
demo_pneumonie/
├── backend/
│   ├── app.py                  # API FastAPI
│   ├── model_utils.py          # preprocessing, predict, gradcam, mc_dropout
│   └── requirements.txt        # dépendances Python
├── frontend/
│   └── index.html              # UI complète (HTML + Tailwind CDN + JS)
├── data/
│   └── demo_cases/             # 4 images de démo pré-sélectionnées
├── modele_pneumonie_densenet.keras   # le modèle (à placer ici)
└── README.md                   # instructions de lancement (3 commandes max)
```

---

## Dépendances Python

```
fastapi
uvicorn[standard]
python-multipart
tensorflow
numpy
pillow
opencv-python-headless
scikit-image
matplotlib
```

---

## Déploiement : Hugging Face (gratuit, un seul service)

### Stockage du modèle : Hugging Face Hub

Le fichier `modele_pneumonie_densenet.keras` (~15 MB) est hébergé sur un repo Hugging Face Hub.

- Créer un repo modèle sur https://huggingface.co (ex: `michel/pneumonie-densenet`)
- Y push le fichier `.keras`
- L'app le télécharge au démarrage avec la lib `huggingface_hub` :
  ```python
  from huggingface_hub import hf_hub_download
  model_path = hf_hub_download(repo_id="michel/pneumonie-densenet", filename="modele_pneumonie_densenet.keras")
  model = keras.models.load_model(model_path)
  ```
- Le fichier est mis en cache localement après le premier téléchargement — les redémarrages suivants sont instantanés

### Hébergement de l'app : Hugging Face Spaces (Docker)

L'app (FastAPI + frontend) tourne dans un Space Docker.

**Structure du Space :**
```
├── Dockerfile
├── backend/
│   ├── app.py
│   └── model_utils.py
├── frontend/
│   └── index.html
└── requirements.txt
```

**Dockerfile :**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Note : Hugging Face Spaces impose le port **7860**. Le Dockerfile doit exposer ce port.

**Dépendances supplémentaires** (ajouter à requirements.txt) :
```
huggingface_hub
```

**URL finale** : `https://<username>-<space-name>.hf.space`

### Contraintes Hugging Face Spaces (tier gratuit)

- 16 GB de RAM — largement suffisant pour TensorFlow + DenseNet121
- CPU uniquement (pas de GPU) — suffisant pour de l'inférence image par image
- Le Space se met en veille après 48h d'inactivité — premier accès après veille = ~30s de cold start (téléchargement modèle + chargement TF)
- Pour éviter le cold start pendant la soutenance : ouvrir l'URL 2 minutes avant

### Backup local

Garder la possibilité de lancer en local en cas de problème réseau :
```bash
cd demo_pneumonie
pip install -r requirements.txt
uvicorn backend.app:app --port 8000
```

Le `model_utils.py` doit gérer les deux cas :
- Variable d'env `MODEL_PATH` définie → charger depuis ce chemin local
- Sinon → télécharger depuis Hugging Face Hub

---

## Ce qui est hors périmètre

- Pas de Claude SDK
- Pas de base de données
- Pas d'authentification
- Pas de modification du modèle ou de réentraînement
- Pas de CI/CD
