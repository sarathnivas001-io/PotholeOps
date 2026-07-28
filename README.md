# PotholeOps

A portfolio MLOps project for classifying road pothole severity (**Low / Medium / High**) from images, built end-to-end: data versioning, experiment tracking, model registry, serving, monitoring, dashboarding, CI/CD, and containerization.

This project was built as a hands-on learning exercise in MLOps pipeline mechanics — every stage was debugged and validated manually, not just scaffolded.

---

## Tech Stack

- **PyTorch** (ResNet18 backbone, transfer learning)
- **DVC** (data + pipeline versioning) with **Backblaze B2** (S3-compatible) remote storage
- **MLflow** (experiment tracking + Model Registry)
- **FastAPI** (model serving + custom HTML/CSS/JS dashboard, served from the same app)
- **Evidently AI** (prediction drift monitoring)
- **GitHub Actions** (CI/CD — automated retraining on data/code changes)
- **Docker / docker-compose** (containerized deployment)
- **pytest** (API test suite)

---

## Dataset

- Source: [Kaggle — Annotated Potholes Dataset](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset) (Pascal VOC format)
- 665 images total, manually organized into `data/raw/{low,medium,high}/` based on severity labels extracted from XML annotations (`minor_pothole`, `medium_pothole`, `major_pothole`), using a **"most severe label wins"** strategy for images with multiple annotated potholes
- Class distribution: **359 low / 224 medium / 82 high** — a real, disclosed class imbalance (see [Known Limitations](#known-limitations))

---

## Pipeline Overview

```
data/raw/{low,medium,high}/
        │
        ▼  (dvc repro — "prepare" stage)
data/processed/{train,val,test}/
        │
        ▼  (dvc repro — "train" stage)
ResNet18 fine-tuning, class-weighted loss, MLflow tracking
        │
        ▼
models/latest_model.pt  +  MLflow Model Registry
        │
        ▼
FastAPI /predict  →  logs/predictions.jsonl  →  Evidently drift report
```

Pipeline stages and hyperparameters are defined in `dvc.yaml` / `params.yaml`. Data and model artifacts are version-controlled with DVC and stored remotely on Backblaze B2 — Git only tracks code and small pointer files.

---

## Project Structure

```
├── .github/workflows/     # CI/CD: automated retraining pipeline
├── dashboard/static/       # Custom HTML/CSS/JS dashboard (served by FastAPI)
├── data/
│   ├── raw/                 # DVC-tracked source images (low/medium/high)
│   └── processed/          # DVC-tracked train/val/test splits
├── models/                 # DVC-tracked trained model checkpoint
├── metrics/                 # Training metrics (dvc metrics show/diff)
├── reports/                 # Evidently drift reports (generated)
├── src/
│   ├── api/                  # FastAPI inference service + model loader
│   ├── data/                 # Data preparation stage
│   ├── monitoring/         # Evidently AI drift monitoring script
│   └── training/           # Model training stage
├── tests/                    # pytest suite for the API
├── dvc.yaml / params.yaml  # Pipeline definition + hyperparameters
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

---

## Setup (Windows / PowerShell)

```powershell
git clone https://github.com/sarathnivas001-io/PotholeOps.git
cd PotholeOps

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Pull DVC-tracked data + model from Backblaze B2
dvc pull
```

> **Note:** MLflow 3.x deprecates the plain filesystem tracking backend. This project sets `MLFLOW_ALLOW_FILE_STORE=true` to opt out of that warning (added automatically via PowerShell profile during development — see `$PROFILE`).

---

## Running the Pipeline

Run the full `prepare → train` pipeline (skips stages whose inputs haven't changed):

```powershell
dvc repro
```

Force a full re-run (e.g. to compare training runs):

```powershell
dvc repro --force train
```

View metrics:

```powershell
dvc metrics show
```

---

## Running the Service Locally

```powershell
uvicorn src.api.main:app --reload --port 8000
```

Open **http://localhost:8000** for the dashboard (live inference + drift monitoring tabs), or call the API directly:

```powershell
curl.exe -X POST http://localhost:8000/predict -F "file=@data\raw\medium\potholes0.png"
```

**Endpoints:**
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the dashboard UI |
| `/health` | GET | Liveness check |
| `/predict` | POST | Upload an image, get a severity prediction |
| `/model-info` | GET | Current serving model's class names |
| `/api/logs` | GET | Recent prediction logs |
| `/api/trigger-drift` | POST | Trigger a background Evidently drift run |

---

## Drift Monitoring

```powershell
python src\monitoring\drift_report.py
```

Compares the live prediction distribution (`logs/predictions.jsonl`) against the validation set's baseline distribution, using Evidently AI. Outputs `reports/drift_report.html` and `reports/drift_summary.json`.

---

## Running with Docker

```powershell
docker compose build
docker compose up
```

Serves the same FastAPI app (API + dashboard) at **http://localhost:8000**, fully containerized — validated with real predictions inside the container.

---

## Tests

```powershell
python -m pytest tests\ -v
```

---

## CI/CD

`.github/workflows/` defines a **"Retrain and Deploy"** GitHub Actions workflow that:
1. Pulls DVC-tracked data from Backblaze B2
2. Reproduces the `prepare → train` pipeline
3. Shows metrics
4. Pushes updated artifacts back to B2

Triggers automatically on changes to `data/raw.dvc`, `src/**`, or `params.yaml`, or can be run manually from the Actions tab.

---

## Known Limitations

- **Class imbalance:** the `high` severity class has only 82 source images (vs. 359 for `low`), and validation F1 for `high` is noticeably lower and more variable than `low`/`medium` (typically ~0.5–0.67 vs. ~0.8+ for `low`). Class weighting was added to the training loss to help; oversampling was tested but reverted after evidence showed it destabilized the majority class without a clear benefit. More real `high`-severity training data is the most reliable fix, and wasn't available for this project.
- **No live public deployment** — the project runs locally and in CI/Docker, but isn't hosted on a public URL.
- The MLflow model registry has never automatically promoted a model to "Production" stage, since no run has yet cleared the `promote_threshold_f1: 0.85` bar set in `params.yaml` — an intentionally strict threshold given the dataset size.

---

## Learnings

A detailed account of real issues hit and resolved during this build — DVC/Git tracking conflicts, Google Drive OAuth and quota dead-ends, migrating DVC remotes to Backblaze B2, MLflow API version migrations, PowerShell environment quirks, and Docker packaging — is available in project notes and commit history.