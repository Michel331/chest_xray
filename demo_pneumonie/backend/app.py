"""
API FastAPI — Détection de pneumonie sur radio thoracique.

Endpoints :
  GET  /             → sert frontend/index.html
  GET  /model/info   → métadonnées du modèle
  POST /analyze      → analyse complète (classification + Grad-CAM + MC Dropout)

Lancement :
  uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

sys.path.insert(0, BASE_DIR)

from backend.model_utils import (
    CLASS_NAMES,
    DISCLAIMER,
    array_to_base64,
    build_gradcam_model,
    classify_uncertainty,
    create_gradcam_overlay,
    get_clinical_flag,
    load_model,
    make_gradcam_heatmap,
    mc_predict,
)
from backend.preprocessing import preprocess_xray_for_inference

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PneumoDetect AI",
    description="API de détection de pneumonie par radiographie thoracique (DenseNet121).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── État global (chargé une seule fois au démarrage) ─────────────────────────
_model = None
_grad_model = None


@app.on_event("startup")
async def startup():
    global _model, _grad_model
    print("[startup] Chargement du modèle…")
    _model = load_model()
    _grad_model = build_gradcam_model(_model)
    print("[startup] Modèle prêt.")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
    return {
        "architecture": "DenseNet121",
        "input_shape": [224, 224, 3],
        "classes": CLASS_NAMES,
        "preprocessing": "grayscale → crop 10% → resize 224 → CLAHE → DenseNet preprocess",
        "uncertainty_method": "MC Dropout (Gal & Ghahramani, ICML 2016)",
        "disclaimer": DISCLAIMER,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    n_mc: int = Form(default=30),
):
    """
    Analyse une radio thoracique.

    Paramètres
    ----------
    file : image JPEG ou PNG
    n_mc : nombre de passes MC Dropout (10–50, défaut 30)

    Retourne
    --------
    JSON avec classe, confiances, incertitude, heatmap base64, flag clinique.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    # Validation basique
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Format non supporté (JPEG/PNG uniquement).")
    n_mc = max(10, min(n_mc, 50))

    # 1. Lire les bytes
    image_bytes = await file.read()

    # 2. Preprocessing
    try:
        tensor, orig_gray = preprocess_xray_for_inference(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur preprocessing : {e}")

    # 3. MC Dropout — prédiction + incertitude
    mean_pred, std_pred = mc_predict(_model, tensor, n_iter=n_mc)
    pred_class_idx = int(np.argmax(mean_pred[0]))
    class_name = CLASS_NAMES[pred_class_idx]
    confidence = float(mean_pred[0][pred_class_idx])
    uncertainty_value = float(std_pred[0][pred_class_idx])

    uncertainty_level = classify_uncertainty(uncertainty_value)
    clinical_flag = get_clinical_flag(uncertainty_level, confidence)

    # 4. Grad-CAM
    try:
        heatmap, _ = make_gradcam_heatmap(tensor, _grad_model, pred_index=pred_class_idx)
        overlay = create_gradcam_overlay(orig_gray, heatmap)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Grad-CAM : {e}")

    # 5. Encoder les images en base64
    orig_rgb = np.stack([orig_gray, orig_gray, orig_gray], axis=-1)
    original_b64 = array_to_base64(orig_rgb)
    gradcam_b64 = array_to_base64(overlay)

    return {
        "prediction": {
            "class": class_name,
            "class_index": pred_class_idx,
            "probabilities": {
                "NORMAL": round(float(mean_pred[0][0]), 4),
                "PNEUMONIE": round(float(mean_pred[0][1]), 4),
            },
        },
        "uncertainty": {
            "value": round(uncertainty_value, 4),
            "level": uncertainty_level,
            "n_samples": n_mc,
        },
        "images": {
            "original": f"data:image/png;base64,{original_b64}",
            "gradcam": f"data:image/png;base64,{gradcam_b64}",
        },
        "clinical_flag": clinical_flag,
        "disclaimer": DISCLAIMER,
    }


# ─── Servir le frontend ───────────────────────────────────────────────────────
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
