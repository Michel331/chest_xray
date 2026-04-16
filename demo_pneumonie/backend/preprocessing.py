"""
Source unique de vérité pour le preprocessing des radios thoraciques.
À importer dans app.py et dans tout script de test — ne jamais dupliquer cette logique.

Pipeline (identique à l'entraînement, cellule 2 du notebook) :
1. Charger l'image (bytes → PIL → numpy grayscale)
2. Crop centré 10% (supprime marqueurs R/L et artefacts de bord)
3. Resize 224×224 (INTER_AREA)
4. CLAHE  skimage.exposure.equalize_adapthist(clip_limit=0.02)
5. Convertir en uint8
6. Dupliquer en 3 canaux
7. densenet preprocess_input (normalisation ImageNet)
8. Ajouter dimension batch → (1, 224, 224, 3)
"""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from skimage import exposure
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

CROP_RATIO = 0.10  # supprime 10% de chaque bord


def crop_center(img: np.ndarray, ratio: float = CROP_RATIO) -> np.ndarray:
    """Crop centré : supprime `ratio` des 4 bords."""
    h, w = img.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    return img[ch : h - ch, cw : w - cw]


def preprocess_xray_for_inference(image_bytes: bytes):
    """
    Pipeline complet depuis les bytes bruts de l'image jusqu'au tenseur modèle.

    Paramètres
    ----------
    image_bytes : bytes
        Contenu brut du fichier image (JPEG ou PNG).

    Retourne
    --------
    tensor : np.ndarray, shape (1, 224, 224, 3), float32
        Prêt pour model.predict().
    orig_gray : np.ndarray, shape (224, 224), uint8
        Image cropée + resizée AVANT CLAHE — utilisée pour l'overlay Grad-CAM.
    """
    # 1. Charger en niveaux de gris
    pil_img = Image.open(BytesIO(image_bytes)).convert("L")
    img_array = np.array(pil_img)

    # 2. Crop centré
    img_cropped = crop_center(img_array, CROP_RATIO)

    # 3. Resize 224×224
    img_resized = cv2.resize(img_cropped, (224, 224), interpolation=cv2.INTER_AREA)

    # 4. Conserver pour visualisation (avant CLAHE)
    orig_gray = img_resized.copy()

    # 5. CLAHE
    gray_f = exposure.equalize_adapthist(img_resized, clip_limit=0.02).astype(np.float32)
    gray_u8 = (gray_f * 255.0).astype(np.uint8)

    # 6. Dupliquer en 3 canaux + preprocess DenseNet
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    rgb = densenet_preprocess(rgb)

    # 7. Dimension batch
    tensor = np.expand_dims(rgb, axis=0)

    return tensor, orig_gray
