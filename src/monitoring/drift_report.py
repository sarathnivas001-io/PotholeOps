"""
src/monitoring/drift_report.py

Compares recent prediction logs (logs/predictions.jsonl, written by the API)
against a baseline distribution (the val set's true label distribution,
data/processed/val/) and generates an Evidently AI drift report as HTML.

This is a *prediction distribution* drift check: it flags when the mix of
predicted severities shifts substantially from what the model saw at
training/validation time — a proxy for real-world drift when true labels
for new field images aren't available yet.

Run: python src/monitoring/drift_report.py
Output: reports/drift_report.html
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

LOG_PATH = Path("logs/predictions.jsonl")
VAL_DIR = Path("data/processed/val")
REPORTS_DIR = Path("reports")
CLASSES = ["low", "medium", "high"]


def baseline_distribution() -> pd.DataFrame:
    """Reference distribution: proportion of each class in the val set."""
    counts = {cls: len(list((VAL_DIR / cls).glob("*"))) for cls in CLASSES if (VAL_DIR / cls).exists()}
    total = sum(counts.values()) or 1
    rows = []
    for cls, n in counts.items():
        rows.extend([{"prediction": cls}] * n)
    if not rows:
        raise SystemExit("No validation data found under data/processed/val/. "
                          "Run `dvc repro` first.")
    return pd.DataFrame(rows)


def current_distribution() -> pd.DataFrame:
    """Current distribution: predictions logged by the API since last check."""
    if not LOG_PATH.exists():
        raise SystemExit(f"{LOG_PATH} not found. Serve some predictions via the API first.")
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            record = json.loads(line)
            rows.append({"prediction": record["prediction"]})
    if not rows:
        raise SystemExit("No predictions logged yet.")
    return pd.DataFrame(rows)


def main():
    reference = baseline_distribution()
    current = current_distribution()

    ref_counts = Counter(reference["prediction"])
    cur_counts = Counter(current["prediction"])
    print(f"Baseline distribution (val set): {dict(ref_counts)}")
    print(f"Current distribution ({len(current)} predictions): {dict(cur_counts)}")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "drift_report.html"
    report.save_html(str(out_path))
    print(f"Drift report written to {out_path}")

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"].get("dataset_drift", False)
    print(f"Dataset drift detected: {drift_detected}")

    # Machine-readable summary for the dashboard / CI
    with open(REPORTS_DIR / "drift_summary.json", "w") as f:
        json.dump({"drift_detected": drift_detected,
                    "reference_counts": dict(ref_counts),
                    "current_counts": dict(cur_counts)}, f, indent=2)


if __name__ == "__main__":
    main()
