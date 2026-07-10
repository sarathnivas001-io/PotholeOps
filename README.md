# PotholeOps
<<<<<<< HEAD

End-to-end MLOps pipeline for automated road pothole severity classification
(Low / Medium / High), with continuous retraining, model versioning, and
drift monitoring — built entirely with free, open-source tools.

## Architecture

| Stage | Component | Description |
|---|---|---|
| 1. Data | DVC + GitHub | Dataset versioning and pipeline stage management |
| 2. Training | PyTorch + Colab | CNN model training with augmentation and validation |
| 3. Tracking | MLflow | Logs hyperparameters, metrics, and model artifacts per run |
| 4. Deployment | FastAPI + Docker | REST API serving predictions from a containerised model |
| 5. CI/CD | GitHub Actions | Auto-retrains and redeploys when new data is pushed |
| 6. Monitoring | Evidently AI + Streamlit | Tracks prediction drift and shows a live dashboard |

## Project layout

```
PotholeOps/
├── data/
│   ├── raw/                # original images (DVC-tracked, not in git)
│   └── processed/          # train/val/test splits after preprocessing
├── src/
│   ├── data/
│   │   └── prepare.py      # splits + preprocessing, DVC stage
│   ├── training/
│   │   └── train.py        # CNN training + MLflow logging, DVC stage
│   ├── api/
│   │   ├── main.py         # FastAPI inference service
│   │   └── model_loader.py # loads latest "Production" model from MLflow registry
│   └── monitoring/
│       └── drift_report.py # Evidently AI drift report generation
├── dashboard/
│   └── app.py               # Streamlit dashboard (inference + drift view)
├── models/                  # local MLflow artifact store (gitignored)
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── retrain_and_deploy.yml
├── dvc.yaml                  # DVC pipeline: prepare -> train
├── params.yaml                # hyperparameters (tracked by DVC)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

## Quickstart

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Initialize DVC (once)
git init
dvc init
dvc remote add -d storage <your-storag.\venv\Scripts\activate.bate-path>   # local folder or free-tier remote (e.g. GDrive)

# 3. Add your dataset
#    Put labelled images under data/raw/{low,medium,high}/*.jpg
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track raw dataset with DVC"

# 4. Run the full pipeline (prepare -> train), tracked by DVC + logged to MLflow
dvc repro

# 5. Inspect experiments
mlflow ui   # http://localhost:5000

# 6. Serve the model
uvicorn src.api.main:app --reload --port 8000

# 7. Run the dashboard
streamlit run dashboard/app.py
```

## CI/CD

Pushing new images to `data/raw/` (tracked via DVC) triggers
`.github/workflows/retrain_and_deploy.yml`, which reproduces the DVC
pipeline, retrains the model, registers it in the MLflow Model Registry if
it beats the current production model, and rebuilds/redeploys the Docker
image.

## Status

This is a **working scaffold**, not a finished product — the CNN
architecture, dataset paths, and thresholds are starting points you should
tune once you have real data. See inline `TODO`s in each file.
=======
MLOps Project
>>>>>>> 833bcd293de3516ebcdaac9d0a764b2a12df6ade
