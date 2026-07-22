"""
src/training/train.py

DVC stage: train
Trains a CNN (transfer-learned backbone) on data/processed/{train,val},
logs the run to MLflow, evaluates on val, and:
  - always saves models/latest_model.pt (DVC-tracked artifact for the pipeline)
  - registers the model in the MLflow Model Registry
  - promotes it to "Production" stage if val F1 >= registry.promote_threshold_f1
    AND it beats the current Production model's F1

Also writes metrics/train_metrics.json for `dvc metrics show/diff`.
"""
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def build_dataloaders(img_size: int, batch_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(PROCESSED_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(PROCESSED_DIR / "val", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.classes


def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model.to(DEVICE)


def evaluate(model, loader) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }


def main():
    params = load_params()
    train_p, reg_p = params["train"], params["registry"]

    if not (PROCESSED_DIR / "train").exists():
        raise SystemExit(
            "data/processed/train not found. Run `dvc repro prepare` first "
            "(and make sure data/raw/{low,medium,high}/ has images)."
        )

    train_loader, val_loader, class_names = build_dataloaders(
        img_size=params["prepare"]["img_size"], batch_size=train_p["batch_size"]
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("potholeops-severity-classification")

    with mlflow.start_run() as run:
        mlflow.log_params(train_p)

        model = build_model(train_p["backbone"], train_p["num_classes"])
        optimizer = torch.optim.Adam(model.parameters(), lr=train_p["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        best_f1 = -1.0
        patience_left = train_p["early_stopping_patience"]
        best_state = None

        for epoch in range(train_p["epochs"]):
            model.train()
            running_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            val_metrics = evaluate(model, val_loader)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }, step=epoch)
            print(f"epoch {epoch+1}/{train_p['epochs']} "
                  f"loss={train_loss:.4f} val_f1={val_metrics['f1_macro']:.4f}")

            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = train_p["early_stopping_patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(best_state)

        # Save artifact for DVC + local serving fallback
        MODELS_DIR.mkdir(exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "classes": class_names,
             "backbone": train_p["backbone"]},
            MODELS_DIR / "latest_model.pt",
        )

        # Log + register with MLflow
        input_example = torch.randn(1, 3, 224, 224)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            serialization_format="pickle"
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, reg_p["model_name"])

        client = MlflowClient()
        promote = False
        if best_f1 >= reg_p["promote_threshold_f1"]:
            current_prod = client.get_latest_versions(reg_p["model_name"], stages=["Production"])
            if not current_prod:
                promote = True
            else:
                prod_run = client.get_run(current_prod[0].run_id)
                prod_f1 = prod_run.data.metrics.get("val_f1_macro", 0.0)
                promote = best_f1 > prod_f1

        if promote:
            client.transition_model_version_stage(
                name=reg_p["model_name"],
                version=registered.version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"Promoted version {registered.version} to Production (val_f1={best_f1:.4f}).")
        else:
            print(f"Not promoted (val_f1={best_f1:.4f} vs threshold "
                  f"{reg_p['promote_threshold_f1']}). Registered as version "
                  f"{registered.version} without stage change.")

        METRICS_DIR.mkdir(exist_ok=True)
        with open(METRICS_DIR / "train_metrics.json", "w") as f:
            json.dump({"best_val_f1_macro": best_f1, "promoted": promote}, f, indent=2)


if __name__ == "__main__":
    main()
