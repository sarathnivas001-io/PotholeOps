"""
src/api/main.py

FastAPI service exposing:
  GET  /                 -> serves the modern web UI dashboard
  GET  /health           -> liveness check
  POST /predict          -> upload a pothole image, get severity prediction
  GET  /model-info         -> which model version is currently serving
  GET  /api/logs         -> fetch recent prediction logs
  POST /api/trigger-drift -> trigger fresh Evidently AI drift monitoring run

Run: uvicorn src.api.main:app --reload --port 8000
"""
import io
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from src.api.model_loader import load_production_model

app = FastAPI(title="PotholeOps Inference & Monitoring API", version="2.0")

# Enable CORS for web UI accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL = None
CLASS_NAMES = None
LOG_PATH = BASE_DIR / "logs" / "predictions.jsonl"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "dashboard" / "static"

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Ensure directories exist
REPORTS_DIR.mkdir(exist_ok=True)
LOG_PATH.parent.mkdir(exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def startup():
    global MODEL, CLASS_NAMES
    MODEL, CLASS_NAMES = load_production_model()


# Mount static assets & reports with absolute paths
app.mount("/dashboard/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


@app.get("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "PotholeOps API is online. Dashboard index.html not found.", "path": str(index_file)}


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


@app.get("/api/logs")
def get_logs(limit: int = 50):
    """Retrieve recent prediction log records."""
    if not LOG_PATH.exists():
        return []
    records = []
    with open(LOG_PATH) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records[-limit:][::-1]


def run_drift_script():
    """Background runner for drift detection."""
    try:
        subprocess.run(["python", "src/monitoring/drift_report.py"], check=True)
    except Exception as e:
        print(f"Drift evaluation background run failed: {e}")


@app.post("/api/trigger-drift")
async def trigger_drift(background_tasks: BackgroundTasks):
    """Trigger background execution of Evidently AI drift monitoring script."""
    background_tasks.add_task(run_drift_script)
    return {"status": "started", "message": "Drift evaluation script launched in background."}
