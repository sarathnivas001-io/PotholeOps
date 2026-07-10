"""
src/api/main.py

FastAPI service exposing:
  GET  /health         -> liveness check
  POST /predict         -> upload a pothole image, get severity prediction
  GET  /model-info       -> which model version is currently serving

Run: uvicorn src.api.main:app --reload --port 8000
"""
import io
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms

from src.api.model_loader import load_production_model

app = FastAPI(title="PotholeOps Inference API", version="1.0")

MODEL = None
CLASS_NAMES = None
LOG_PATH = Path("logs/predictions.jsonl")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.on_event("startup")
def startup():
    global MODEL, CLASS_NAMES
    MODEL, CLASS_NAMES = load_production_model()
    LOG_PATH.parent.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/model-info")
def model_info():
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    tensor = TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()

    pred_idx = int(torch.tensor(probs).argmax())
    result = {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "class_probabilities": {
            CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)
        },
    }

    # Log every prediction for drift monitoring (src/monitoring/drift_report.py reads this)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "timestamp": time.time(),
            "filename": file.filename,
            **result,
        }) + "\n")

    return result
