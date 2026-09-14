"""Evaluate the frozen SVM baseline on the official ASVspoof 2019 LA Eval set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "LA"
MODEL_PATH = PROJECT_ROOT / "models" / "svm" / "svm_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "svm" / "scaler.pkl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "svm_eval_results.json"


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Calculate EER from spoof scores using the same threshold convention as training."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    if labels.shape[0] != scores.shape[0] or np.unique(labels).size != 2:
        raise ValueError("EER requires equal-length scores and binary labels.")
    thresholds = np.unique(scores)
    gaps = []
    values = []
    for threshold in thresholds:
        predicted = (scores >= threshold).astype(int)
        tp = np.sum((predicted == 1) & (labels == 1))
        fp = np.sum((predicted == 1) & (labels == 0))
        tn = np.sum((predicted == 0) & (labels == 0))
        fn = np.sum((predicted == 0) & (labels == 1))
        far = fp / (fp + tn) if fp + tn else 0.0
        frr = fn / (fn + tp) if fn + tp else 0.0
        gaps.append(abs(far - frr))
        values.append((far + frr) / 2.0)
    return float(values[int(np.argmin(gaps))])


def _metric(path: Path, *keys: str) -> float | None:
    """Read a metric from a JSON object using alternative nested keys."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    current: Any = data.get("metrics", data) if keys and keys[0] == "metrics" else data
    for key in keys:
        if key == "metrics" and current is data.get("metrics", data):
            continue
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def print_comparison(svm_metrics: dict[str, float]) -> None:
    """Print the same-Eval comparison with existing CNN and AASIST results."""
    cnn_path = PROJECT_ROOT / "results" / "cnn_final_eval_results.json"
    aasist_path = PROJECT_ROOT / "results" / "aasist" / "aasist_eval_results.json"
    rows = [
        ("CNN", _metric(cnn_path, "metrics", "accuracy"), _metric(cnn_path, "metrics", "f1_score"),
         _metric(cnn_path, "threshold_independent_metrics", "eer")),
        ("SVM", svm_metrics["accuracy"], svm_metrics["f1_score"], svm_metrics["eer"]),
        ("AASIST", _metric(aasist_path, "metrics", "accuracy"), _metric(aasist_path, "metrics", "f1"),
         _metric(aasist_path, "metrics", "eer")),
    ]
    print("\nOfficial ASVspoof 2019 LA Eval comparison")
    print(f"{'Model':<10}{'Accuracy':>12}{'F1':>12}{'EER':>12}")
    print("-" * 46)
    for model, accuracy, f1, eer in rows:
        print(f"{model:<10}{accuracy if accuracy is not None else float('nan'):>12.4f}"
              f"{f1 if f1 is not None else float('nan'):>12.4f}"
              f"{eer if eer is not None else float('nan'):>12.4f}")


def main() -> int:
    """Load Eval features, score the frozen SVM, and save metrics."""
    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise FileNotFoundError(f"SVM model and scaler are required: {MODEL_PATH}, {SCALER_PATH}")
    print("[INFO] Loading official Eval MFCC-delta features...")
    features, labels, _ = load_dataset(DATASET_ROOT, protocol_type="eval", feature_type="mfcc_delta")
    if features.size == 0:
        raise ValueError("No Eval features were loaded.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(scaled_features)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "eer": calculate_eer(labels, probabilities),
    }
    results = {
        "model": "SVM",
        "split": "official_eval",
        "total_samples": int(labels.size),
        "bonafide_samples": int(np.sum(labels == 0)),
        "spoof_samples": int(np.sum(labels == 1)),
        "metrics": metrics,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print_comparison(metrics)
    print(f"\nSaved Eval results to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
