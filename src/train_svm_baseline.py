"""Train and evaluate the SVM baseline for ASVspoof 2019 LA audio.

This script loads the official train/dev split, trains an RBF SVM on the
MFCC-delta features, evaluates the resulting model on the development set,
and saves the model, scaler, results CSV/JSON, and a confusion matrix plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from data_loader import prepare_dataset_svm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "LA"
MODELS_DIR = PROJECT_ROOT / "models" / "svm"
RESULTS_DIR = PROJECT_ROOT / "results"


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute the equal error rate (EER) from binary labels and scores.

    Higher scores are interpreted as the positive class, i.e. spoof.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    if y_true.shape[0] != y_scores.shape[0]:
        raise ValueError("y_true and y_scores must have the same length.")
    if np.unique(y_true).size != 2:
        raise ValueError("EER requires binary labels represented as 0/1 values.")

    thresholds = np.unique(y_scores)
    min_gap = float("inf")
    eer_value = 0.0

    for threshold in thresholds:
        predictions = (y_scores >= threshold).astype(int)

        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        tn = np.sum((predictions == 0) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))

        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        gap = abs(far - frr)

        if gap < min_gap:
            min_gap = gap
            eer_value = (far + frr) / 2.0

    return float(eer_value)


def save_results_json(results_path: Path, metrics: dict[str, float | int | list[int]]) -> None:
    """Save evaluation metrics to a JSON file."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_confusion_matrix_plot(confusion_path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save the confusion matrix as a PNG plot."""
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["bonafide", "spoof"], yticklabels=["bonafide", "spoof"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("SVM Confusion Matrix (Dev Set)")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=300)
    plt.close()


def main() -> None:
    """Train the SVM baseline, evaluate it, and save model artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Preparing SVM dataset...")
    X_train, y_train, X_dev, y_dev, scaler = prepare_dataset_svm(
        dataset_root=DATASET_ROOT,
        use_delta=True,
        test_mode=False,
    )

    print("[INFO] Training SVM classifier with RBF kernel...")
    svm_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42,
        verbose=True,
    )
    svm_model.fit(X_train, y_train)

    print("[INFO] Evaluating on development set...")
    y_pred = svm_model.predict(X_dev)
    y_scores = svm_model.predict_proba(X_dev)[:, 1]

    accuracy = accuracy_score(y_dev, y_pred)
    precision = precision_score(y_dev, y_pred, zero_division=0)
    recall = recall_score(y_dev, y_pred, zero_division=0)
    f1 = f1_score(y_dev, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_dev, y_scores)
    eer = calculate_eer(y_dev, y_scores)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "eer": float(eer),
        "confusion_matrix": confusion_matrix(y_dev, y_pred, labels=[0, 1]).tolist(),
    }

    model_path = MODELS_DIR / "svm_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    results_json_path = RESULTS_DIR / "svm_results.json"
    results_csv_path = RESULTS_DIR / "svm_results.csv"
    confusion_plot_path = RESULTS_DIR / "svm_confusion_matrix.png"

    dump(svm_model, model_path)
    dump(scaler, scaler_path)
    save_results_json(results_json_path, metrics)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_csv_path, index=False)
    save_confusion_matrix_plot(confusion_plot_path, y_dev, y_pred)

    print("\nSVM baseline evaluation summary")
    print("-" * 90)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"EER:       {eer:.4f}")
    print("-" * 90)

    print("\nSenior EchoShield reported baseline results:")
    print("  SVM: Accuracy=93.4%, Precision=0.93, Recall=0.92, F1=0.92")
    print("  CNN: Accuracy=96.8%, Precision=0.97, Recall=0.96, F1=0.96")
    print("\nComparison note: this script saves our reproduced SVM baseline outputs separately from the senior reported baseline.")
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved JSON metrics to: {results_json_path}")
    print(f"Saved CSV metrics to: {results_csv_path}")
    print(f"Saved confusion matrix plot to: {confusion_plot_path}")
    print("\nNext steps: reproduce the CNN baseline, compare against AASIST, then extend with metadata and forensic evidence analysis.")


if __name__ == "__main__":
    main()
