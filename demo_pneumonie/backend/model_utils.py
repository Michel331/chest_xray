"""
Utilitaires modèle : chargement, Grad-CAM, MC Dropout, incertitude.
"""

import base64
import os
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIE"}

# ─── Seuils d'incertitude (std de la classe prédite sur N passes MC) ──────────
# Calibrés sur le test set (Kaggle Chest X-Ray)
UNCERTAINTY_LOW = 0.05     # std < 5%  → fiable
UNCERTAINTY_HIGH = 0.12    # std > 12% → à vérifier

DISCLAIMER = (
    "Outil d'aide à la décision radiologique uniquement. "
    "Ne remplace pas l'avis d'un médecin ou d'un radiologue qualifié."
)


# ─── Chargement du modèle ─────────────────────────────────────────────────────

def load_model() -> tf.keras.Model:
    """
    Charge le modèle .keras.

    Ordre de recherche :
    1. Variable d'env MODEL_PATH (chemin local absolu)
    2. modele_pneumonie_densenet.keras  (répertoire racine du projet)
    3. modele_pneumonie_final.keras     (nom alternatif)
    4. Téléchargement depuis Hugging Face Hub (variable HF_REPO_ID requise)
    """
    # 1. Chemin explicite
    explicit = os.environ.get("MODEL_PATH")
    if explicit and os.path.isfile(explicit):
        print(f"[model] Chargement depuis MODEL_PATH : {explicit}")
        return tf.keras.models.load_model(explicit)

    # 2-3. Fichiers locaux (cherche dans demo_pneumonie/ puis dans chest_xray/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = os.path.dirname(base_dir)
    candidates = [
        os.path.join(base_dir,   "modele_pneumonie_densenet_balanced.keras"),
        os.path.join(parent_dir, "modele_pneumonie_densenet_balanced.keras"),
        os.path.join(base_dir,   "best_model_densenet_balanced.keras"),
        os.path.join(parent_dir, "best_model_densenet_balanced.keras"),
        os.path.join(base_dir,   "modele_pneumonie_densenet.keras"),
        os.path.join(base_dir,   "modele_pneumonie_final.keras"),
        os.path.join(base_dir,   "best_model_densenet.keras"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            print(f"[model] Chargement local : {path}")
            return tf.keras.models.load_model(path, compile=False)

    # 4. Hugging Face Hub
    repo_id = os.environ.get("HF_REPO_ID")
    if repo_id:
        print(f"[model] Téléchargement depuis HF Hub : {repo_id}")
        from huggingface_hub import hf_hub_download
        hf_path = hf_hub_download(
            repo_id=repo_id,
            filename="modele_pneumonie_densenet.keras",
        )
        return tf.keras.models.load_model(hf_path, compile=False)

    raise FileNotFoundError(
        "Modèle introuvable. Définissez MODEL_PATH ou HF_REPO_ID, "
        "ou placez modele_pneumonie_densenet.keras à la racine du projet."
    )


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

def build_gradcam_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Construit un modèle fonctionnel qui retourne simultanément :
      - la sortie de la dernière Conv2D de DenseNet121
      - les prédictions finales (softmax)

    Cela permet un GradientTape correct (un seul forward pass).
    """
    # Trouver le sous-modèle DenseNet121
    densenet = next(
        (l for l in model.layers if isinstance(l, tf.keras.Model)), None
    )

    if densenet is not None:
        # Dernière Conv2D dans DenseNet
        last_conv_name = [
            l.name for l in densenet.layers
            if isinstance(l, tf.keras.layers.Conv2D)
        ][-1]
        last_conv_layer = densenet.get_layer(last_conv_name)

        # Sous-modèle : densenet.input → [last_conv.output, densenet.output]
        dn_grad = tf.keras.Model(
            inputs=densenet.input,
            outputs=[last_conv_layer.output, densenet.output],
        )

        # Couches de la tête (après DenseNet dans le modèle principal)
        head_layers = []
        after = False
        for layer in model.layers:
            if layer is densenet:
                after = True
                continue
            if after:
                head_layers.append(layer)

        # Modèle combiné : input → [conv_out, preds_finales]
        inp = tf.keras.Input(shape=(224, 224, 3))
        conv_out, dn_out = dn_grad(inp)
        x = dn_out
        for layer in head_layers:
            x = layer(x)

        return tf.keras.Model(inputs=inp, outputs=[conv_out, x])

    # Fallback : modèle sans sous-modèle imbriqué
    last_conv_name = [
        l.name for l in model.layers
        if isinstance(l, tf.keras.layers.Conv2D)
    ][-1]
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )


def make_gradcam_heatmap(
    img_tensor: np.ndarray,
    grad_model: tf.keras.Model,
    pred_index: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Calcule la heatmap Grad-CAM.

    Retourne
    --------
    heatmap : np.ndarray, shape (H, W), float32, valeurs dans [0, 1]
    pred_index : int  (classe utilisée pour les gradients)
    """
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Grad-CAM : gradients None — vérifier l'architecture du modèle."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), pred_index


def create_gradcam_overlay(orig_gray: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Superpose la heatmap Grad-CAM (colormap JET) sur l'image originale.

    Paramètres
    ----------
    orig_gray : np.ndarray (224, 224) uint8
    heatmap   : np.ndarray (H, W) float32 valeurs [0, 1]

    Retourne
    --------
    np.ndarray (224, 224, 3) uint8  image RGB avec overlay
    """
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colormap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    img_rgb = np.stack([orig_gray, orig_gray, orig_gray], axis=-1).astype(np.float32)

    superimposed = np.clip(colormap_rgb * 0.4 + img_rgb, 0, 255).astype(np.uint8)
    return superimposed


# ─── MC Dropout ───────────────────────────────────────────────────────────────

def mc_predict(
    model: tf.keras.Model,
    img_tensor: np.ndarray,
    n_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    N passes forward avec Dropout actif (training=True).

    Retourne
    --------
    mean_pred : np.ndarray (1, num_classes)
    std_pred  : np.ndarray (1, num_classes)
    """
    preds = np.array([
        model(img_tensor, training=True).numpy()
        for _ in range(n_iter)
    ])
    return preds.mean(axis=0), preds.std(axis=0)


# ─── Incertitude & flag clinique ──────────────────────────────────────────────

def classify_uncertainty(std: float) -> str:
    """Retourne 'low', 'medium' ou 'high' selon l'écart-type MC."""
    if std < UNCERTAINTY_LOW:
        return "low"
    if std < UNCERTAINTY_HIGH:
        return "medium"
    return "high"


def get_clinical_flag(uncertainty_level: str, confidence: float) -> str | None:
    """
    Règles métier du cahier des charges :
    - Incertitude haute → second avis
    - Incertitude moyenne + confiance < 75% → croiser avec tableau clinique
    """
    if uncertainty_level == "high":
        return "second_opinion"
    if uncertainty_level == "medium" and confidence < 0.75:
        return "cross_clinical"
    return None


# ─── Encodage image → base64 ──────────────────────────────────────────────────

def array_to_base64(arr: np.ndarray) -> str:
    """Convertit un numpy array RGB (H, W, 3) uint8 en PNG base64."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
