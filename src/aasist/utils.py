from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.clip(logits, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-logits))


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    if len(np.unique(y_true)) != 2:
        return float("nan")
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    except ValueError:
        return float("nan")
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    fnr = 1.0 - tpr
    diffs = np.abs(fpr - fnr)
    if diffs.size == 0:
        return float("nan")
    idx = int(np.argmin(diffs))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def select_threshold_from_scores(y_true: np.ndarray, y_scores: np.ndarray, method: str = "eer") -> tuple[float, dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    if method.lower() == "eer":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1.0 - tpr
        diff = np.abs(fpr - fnr)
        idx = int(np.argmin(diff))
        threshold = float(thresholds[idx])
    elif method.lower() == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        j = tpr - fpr
        idx = int(np.argmax(j))
        threshold = float(thresholds[idx])
    elif method.lower() == "f1":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        best = -1.0
        best_threshold = float(thresholds[0])
        for thr, tp, fp in zip(thresholds, tpr, fpr):
            pred = (y_scores >= thr).astype(np.int64)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best:
                best = f1
                best_threshold = float(thr)
        threshold = best_threshold
    else:
        raise ValueError(f"Unsupported threshold method: {method}")

    preds = (y_scores >= threshold).astype(np.int64)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "eer": float(calculate_eer(y_true, y_scores)),
        "specificity": float(np.sum((y_true == 0) & (preds == 0)) / max(np.sum(y_true == 0), 1)),
        "fpr": float(np.sum((y_true == 0) & (preds == 1)) / max(np.sum(y_true == 0), 1)),
        "tpr": float(np.sum((y_true == 1) & (preds == 1)) / max(np.sum(y_true == 1), 1)),
    }
    return float(threshold), metrics


def compute_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    y_pred = (y_scores >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_neg = tn + fp
    total_pos = tp + fn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = (tn / total_neg) if total_neg else 0.0
    fpr = (fp / total_neg) if total_neg else 0.0
    tpr = (tp / total_pos) if total_pos else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["Bonafide", "Spoof"])
    plt.yticks([0, 1], ["Bonafide", "Spoof"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > cm.max() / 2 else "black"
            plt.text(j, i, str(value), ha="center", va="center", color=color)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
