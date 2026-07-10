"""
tests/test_api.py

Basic smoke tests for the FastAPI service. Requires a trained model to exist
(models/latest_model.pt), so run `dvc repro` first.

Run: pytest tests/
"""
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "classes" in response.json()


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
