"""
src/api/model_loader.py

Loads the current "Production" model from the MLflow Model Registry.
Falls back to models/latest_model.pt if the registry has no Production
version yet (e.g. first run, before any promotion).
"""
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import yaml

MODELS_DIR = Path("models")


def _load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["registry"]


def load_production_model():
    """
    Returns (model, class_names). Tries MLflow registry first, then
    falls back to the local checkpoint saved by the training stage.
    """
    reg_params = _load_params()
    mlflow.set_tracking_uri("file:./mlruns")

    try:
        model_uri = f"models:/{reg_params['model_name']}/Production"
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()
        # class names aren't stored in the MLflow pytorch flavor by default,
        # so we keep them alongside the local checkpoint as source of truth.
        checkpoint = torch.load(MODELS_DIR / "latest_model.pt", map_location="cpu")
        return model, checkpoint["classes"]
    except Exception as e:
        print(f"[WARN] Could not load Production model from MLflow registry ({e}). "
              f"Falling back to models/latest_model.pt")
        checkpoint_path = MODELS_DIR / "latest_model.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(
                "No model available: MLflow registry has no Production version and "
                "models/latest_model.pt does not exist. Run `dvc repro` to train one."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        from torchvision import models as tv_models
        import torch.nn as nn

        model = tv_models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint["classes"]
