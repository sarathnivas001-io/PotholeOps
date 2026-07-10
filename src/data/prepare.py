"""
src/data/prepare.py

DVC stage: prepare
Splits data/raw/{low,medium,high}/*.jpg into
data/processed/{train,val,test}/{low,medium,high}/*.jpg

Expected input layout:
    data/raw/
        low/    *.jpg
        medium/ *.jpg
        high/   *.jpg
"""
import random
import shutil
from pathlib import Path

import yaml

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CLASSES = ["low", "medium", "high"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["prepare"]


def split_files(files: list[Path], val_split: float, test_split: float, seed: int):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    val_files = files[:n_val]
    test_files = files[n_val:n_val + n_test]
    train_files = files[n_val + n_test:]
    return train_files, val_files, test_files


def main():
    params = load_params()

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (PROCESSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {}
    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            print(f"[WARN] {cls_dir} does not exist yet — skipping. "
                  f"Add labelled images under data/raw/{cls}/ before running the pipeline.")
            summary[cls] = 0
            continue

        files = [p for p in cls_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
        train_files, val_files, test_files = split_files(
            files, params["val_split"], params["test_split"], params["seed"]
        )

        for split_name, split_files_list in [
            ("train", train_files), ("val", val_files), ("test", test_files)
        ]:
            for src in split_files_list:
                dst = PROCESSED_DIR / split_name / cls / src.name
                shutil.copy2(src, dst)

        summary[cls] = len(files)
        print(f"{cls}: {len(files)} images -> "
              f"train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    total = sum(summary.values())
    if total == 0:
        print("[WARN] No images found under data/raw/. "
              "Populate data/raw/{low,medium,high}/ with labelled pothole images.")
    else:
        print(f"Done. {total} images processed across {len(CLASSES)} classes.")


if __name__ == "__main__":
    main()
