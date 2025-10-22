"""Training utilities for the lightweight vision classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .features_vision import VisionConfig, VisionFeatureExtractor, compute_image_signals

CLASSES: Tuple[str, ...] = (
    "eczema",
    "impetigo",
    "cellulitis",
    "tinea",
    "urticaria",
    "acne",
    "zoster",
)

CACHE_NAME = "train_cache.pkl"


def build_dataset(root: str, cache_dir: str = "artifacts", device: str = "cpu") -> pd.DataFrame:
    """Walk through ``root`` and cache embeddings and signals."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    extractor = VisionFeatureExtractor(VisionConfig(device=device))
    records: List[dict] = []

    for label in CLASSES:
        class_dir = root_path / label
        if not class_dir.exists():
            continue
        for image_path in class_dir.glob("*"):
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            embedding = extractor.embed_array(img)
            signals = compute_image_signals(img)
            record = {
                "path": str(image_path),
                "label": label,
                "embedding": embedding,
            }
            record.update({f"sig_{k}": v for k, v in signals.items()})
            records.append(record)

    if not records:
        raise RuntimeError("No images were processed. Check dataset paths.")

    df = pd.DataFrame(records)
    joblib.dump(df, cache_path / CACHE_NAME)
    return df


def load_cached(cache_dir: str = "artifacts") -> pd.DataFrame:
    cache_path = Path(cache_dir) / CACHE_NAME
    if not cache_path.exists():
        raise FileNotFoundError("Cached features not found. Run build_dataset first.")
    return joblib.load(cache_path)


def vectorize(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    embeddings = np.stack(df["embedding"].to_numpy())
    signal_cols = [c for c in df.columns if c.startswith("sig_")]
    signals = df[signal_cols].to_numpy(dtype=float) if signal_cols else np.empty((len(df), 0))
    X = np.hstack([embeddings, signals]) if signals.size else embeddings

    labels = df["label"].astype("category")
    y = labels.cat.codes.to_numpy()
    label_map = dict(enumerate(labels.cat.categories))
    return X, y, signal_cols, label_map


def train_model(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> dict:
    X, y, signal_cols, label_map = vectorize(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        multi_class="multinomial",
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)

    report = classification_report(y_test, y_pred, target_names=list(label_map.values()), output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    artifact = {
        "scaler": scaler,
        "clf": clf,
        "label_map": label_map,
        "signal_columns": signal_cols,
        "metrics": {
            "classification_report": report,
            "confusion_matrix": matrix,
        },
    }
    return artifact


def save_model(artifact: dict, out_dir: str = "artifacts", name: str = "vision_clf.joblib") -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / name
    joblib.dump(artifact, full_path)
    return full_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hybrid vision classifier")
    parser.add_argument("data_root", help="Path to the labelled training data")
    parser.add_argument("--cache-dir", default="artifacts", help="Directory to store cached features")
    parser.add_argument("--device", default="cpu", help="Torch device for embedding extraction")
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip feature cache rebuild and use existing cache",
    )
    args = parser.parse_args()

    if args.skip_cache:
        df = load_cached(args.cache_dir)
    else:
        df = build_dataset(args.data_root, cache_dir=args.cache_dir, device=args.device)

    artifact = train_model(df)
    model_path = save_model(artifact, out_dir=args.cache_dir)
    print(f"Model saved to {model_path}")
    print("Classification report:")
    print(pd.DataFrame(artifact["metrics"]["classification_report"]).T)
    print("Confusion matrix:")
    print(artifact["metrics"]["confusion_matrix"])


if __name__ == "__main__":
    main()
