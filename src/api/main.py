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
from pydantic import BaseModel
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
INCOMING_DIR = BASE_DIR / "data" / "incoming"
INCOMING_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
REVIEWS_LOG_PATH = BASE_DIR / "logs" / "reviews.jsonl"

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
app.mount("/incoming-images", StaticFiles(directory=str(INCOMING_DIR)), name="incoming")


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

    image_bytes = await file.read()

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # confirms it's a genuine, non-corrupted image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # re-open after verify()
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

    # Save the image with a safe, server-generated filename (never trust client filenames for paths)
    import uuid
    safe_filename = f"{uuid.uuid4().hex}.jpg"
    saved_path = INCOMING_DIR / safe_filename
    image.save(saved_path, format="JPEG")

    # Log every prediction for drift monitoring AND for later human review
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "timestamp": time.time(),
            "filename": file.filename,       # original name, metadata only
            "saved_as": safe_filename,          # actual server-side file
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


def _get_reviewed_filenames() -> set:
    if not REVIEWS_LOG_PATH.exists():
        return set()
    reviewed = set()
    with open(REVIEWS_LOG_PATH) as f:
        for line in f:
            if line.strip():
                try:
                    reviewed.add(json.loads(line)["saved_as"])
                except Exception:
                    pass
    return reviewed


@app.get("/api/review-queue")
def review_queue(limit: int = 20):
    """Recent predictions that haven't been reviewed/labeled yet."""
    if not LOG_PATH.exists():
        return []
    reviewed = _get_reviewed_filenames()
    pending = []
    with open(LOG_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            saved_as = record.get("saved_as")
            if saved_as and saved_as not in reviewed and (INCOMING_DIR / saved_as).exists():
                pending.append({
                    "saved_as": saved_as,
                    "original_filename": record.get("filename"),
                    "prediction": record.get("prediction"),
                    "confidence": record.get("confidence"),
                    "timestamp": record.get("timestamp"),
                    "image_url": f"/incoming-images/{saved_as}",
                })
    return pending[-limit:][::-1]


class ReviewSubmission(BaseModel):
    saved_as: str
    confirmed_label: str  # "low", "medium", or "high"


@app.post("/api/review")
def submit_review(submission: ReviewSubmission):
    if submission.confirmed_label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Label must be one of {CLASS_NAMES}")

    source_path = INCOMING_DIR / submission.saved_as
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Image not found in incoming queue.")

    # Move the image into the correct class folder under data/raw/
    dest_dir = DATA_RAW_DIR / submission.confirmed_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / submission.saved_as
    source_path.rename(dest_path)

    # Record that this image has been reviewed (so it drops out of the queue)
    with open(REVIEWS_LOG_PATH, "a") as f:
        f.write(json.dumps({
            "saved_as": submission.saved_as,
            "confirmed_label": submission.confirmed_label,
            "reviewed_at": time.time(),
        }) + "\n")

    return {
        "status": "ok",
        "message": f"Image moved to data/raw/{submission.confirmed_label}/. "
                   f"Run `dvc add data/raw` and commit to include it in the next training run."
    }


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
