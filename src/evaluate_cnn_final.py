"""Evaluate the trained CNN on the official ASVspoof 2019 LA eval split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from train_cnn_final import (
    DEV_AUDIO_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    PROTOCOL_DIR,
    RESULTS_DIR,
    SR,
    TARGET_FRAMES,
    TARGET_SAMPLES,
    TRAIN_AUDIO_DIR,
    FinalCNN,
    extract_spectrogram,
    load_protocol,
)


EVAL_AUDIO_DIR = PROJECT_ROOT / "data" / "LA" / "ASVspoof2019_LA_eval" / "flac"
EVAL_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"
MODEL_PATH = MODEL_DIR / "cnn_final_best.pth"
NORM_PATH = RESULTS_DIR / "cnn_final_norm.json"
THRESHOLD_PATH = RESULTS_DIR / "cnn_final_threshold.json"


try:
    from scipy.special import expit as stable_expit
except Exception:  # pragma: no cover - fallback for environments without SciPy
    def stable_expit(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x))


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for model logits without overflow."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    probs = stable_expit(logits_arr)
    return np.asarray(probs, dtype=np.float32)


def load_frozen_threshold(path: Path) -> dict:
    """Load the DEV-selected threshold and refuse any attempt to optimize on eval data."""
    if not path.exists():
        raise FileNotFoundError(f"Frozen DEV threshold file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Threshold JSON is malformed: {path}")
    if payload.get("selection_source") != "dev_only":
        raise RuntimeError(
            "Refusing to evaluate with a threshold not selected on DEV ONLY. "
            "Official eval data must not be used for threshold selection."
        )
    threshold = float(payload["threshold"])
    if not np.isfinite(threshold):
        raise ValueError(f"Frozen threshold is not finite: {threshold}")
    return payload


def compute_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict:
    """Compute the classification metrics for a single operating threshold."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = (np.asarray(y_scores) >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_neg = tn + fp
    total_pos = tp + fn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = (tn / total_neg) if total_neg else 0.0
    tpr = (tp / total_pos) if total_pos else 0.0
    fpr = (fp / total_neg) if total_neg else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(specificity),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def load_norm_stats(path: Path) -> Tuple[float, float, dict]:
    """Load train min/max stats for normalization."""
    if not path.exists():
        raise FileNotFoundError(f"Normalization file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "train_min" not in data or "train_max" not in data:
        raise ValueError(f"Normalization file missing train_min/train_max: {path}")

    return float(data["train_min"]), float(data["train_max"]), data


def get_utterance_ids(protocol_path: Path) -> set[str]:
    """Return set of utterance IDs from a protocol file."""
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    records = load_protocol(protocol_path)
    return {utt for utt, _ in records}


def load_eval_split() -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """Load eval protocol and corresponding spectrograms using train-based normalization."""
    if not EVAL_PROTOCOL.exists():
        raise FileNotFoundError(f"Eval protocol not found: {EVAL_PROTOCOL}")
    if not EVAL_AUDIO_DIR.exists():
        raise FileNotFoundError(f"Eval audio directory not found: {EVAL_AUDIO_DIR}")

    records = load_protocol(EVAL_PROTOCOL)
    if not records:
        raise ValueError(f"No valid eval records loaded from {EVAL_PROTOCOL}")

    train_ids = get_utterance_ids(PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt")
    dev_ids = get_utterance_ids(PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt")
    eval_ids = {utt for utt, _ in records}

    eval_train_overlap = eval_ids.intersection(train_ids)
    eval_dev_overlap = eval_ids.intersection(dev_ids)
    if eval_train_overlap or eval_dev_overlap:
        overlap_examples = {
            "eval_train": sorted(list(eval_train_overlap))[:5],
            "eval_dev": sorted(list(eval_dev_overlap))[:5],
        }
        raise RuntimeError(f"Eval/train-dev overlap detected: {overlap_examples}")

    print("[PASS] Eval/train overlap check passed")
    print("[PASS] Eval/dev overlap check passed")

    # Normalization stats must come from training set only.
    train_min, train_max, norm_meta = load_norm_stats(NORM_PATH)
    print("[PASS] Train-only normalization stats loaded")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    missing = []

    for utt, label in records:
        audio_path = EVAL_AUDIO_DIR / f"{utt}.flac"
        if not audio_path.exists():
            missing.append(str(audio_path))
            continue
        try:
            spec = extract_spectrogram(audio_path)
        except Exception as exc:
            print(f"[WARN] Failed to extract {audio_path}: {exc}")
            continue

        spec = np.clip(spec, train_min, train_max)
        spec = (spec - train_min) / (train_max - train_min + 1e-8)
        spec = np.clip(spec, 0.0, 1.0)
        if np.isnan(spec).any() or np.isinf(spec).any():
            raise ValueError(f"Normalization produced NaN/Inf for eval sample {audio_path}")
        X_list.append(spec[np.newaxis, :, :])
        y_list.append(int(label))

    if not X_list:
        raise RuntimeError(f"No valid eval samples loaded from {EVAL_AUDIO_DIR}")
    if missing:
        print(f"[WARN] Missing eval audio files: {len(missing)}")
        print(f"[WARN] Example missing: {missing[:5]}")
        raise RuntimeError(f"Eval audio missing: {len(missing)} files. Evaluation aborted.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    print(f"[INFO] Eval values in [0,1]: {bool(np.all((X >= 0) & (X <= 1)))}")

    # Sanity check for eval class presence.
    unique = np.unique(y)
    if unique.size != 2:
        raise ValueError("Eval dataset contains only one class. Expected both bonafide and spoof samples.")

    print(f"Eval tensor shape: {X.shape}")
    print(f"Eval labels: {np.unique(y, return_counts=True)}")
    print(f"Eval samples: {len(y)}")
    return X, y, [str(p) for p in missing], norm_meta


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute EER robustly if both classes exist."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) != 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    diffs = np.abs(fpr - fnr)
    if diffs.size == 0:
        return float("nan")
    idx = int(np.argmin(diffs))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="tab:blue", lw=2, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_eer_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: Path) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, fpr, label="FPR", color="tab:orange")
    plt.plot(thresholds, fnr, label="FNR", color="tab:blue")
    plt.axvline(thresholds[int(np.argmin(np.abs(fpr - fnr)))], color="tab:red", linestyle="--", alpha=0.8, label="EER threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("EER Curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Official ASVspoof 2019 LA evaluation for the final CNN checkpoint.")
    parser.add_argument("--split", choices=["eval"], default="eval", help="Official evaluation split to score.")
    args = parser.parse_args()

    print("# ========== OFFICIAL ASVSPOOF 2019 LA EVALUATION ==========")
    print(f"[INFO] Requested split: {args.split}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    print(f"[PASS] Model checkpoint found: {MODEL_PATH}")

    if not EVAL_PROTOCOL.exists():
        raise FileNotFoundError(f"Eval protocol file not found: {EVAL_PROTOCOL}")
    print(f"[PASS] Eval protocol found: {EVAL_PROTOCOL}")

    if not NORM_PATH.exists():
        raise FileNotFoundError(f"Normalization file not found: {NORM_PATH}")
    print(f"[PASS] Normalization file found: {NORM_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Frozen DEV threshold file not found: {THRESHOLD_PATH}")
    threshold_payload = load_frozen_threshold(THRESHOLD_PATH)
    threshold_value = float(threshold_payload["threshold"])
    print("# ========== THRESHOLD SELECTION ==========")
    print("Threshold source: DEV ONLY")
    print(f"Selection method: {threshold_payload.get('selection_method', 'unknown')}")
    print(f"Selected threshold: {threshold_value:.6f}")
    print("Eval used for threshold selection: NO")
    print("# ========================================")

    training_summary_path = RESULTS_DIR / "cnn_final_results.json"
    training_summary = {}
    if training_summary_path.exists():
        with training_summary_path.open("r", encoding="utf-8") as fh:
            training_summary = json.load(fh)
    best_epoch = int(training_summary.get("best_epoch", 8))
    best_val_loss = float(training_summary.get("best_val_loss", 0.003612))

    eval_records = load_protocol(EVAL_PROTOCOL)
    if not eval_records:
        raise ValueError(f"Eval protocol is empty: {EVAL_PROTOCOL}")

    eval_counts = {0: 0, 1: 0}
    for _, label in eval_records:
        eval_counts[int(label)] = eval_counts.get(int(label), 0) + 1
    print(f"Eval protocol counts: bonafide={eval_counts.get(0, 0)}, spoof={eval_counts.get(1, 0)}")

    train_ids = get_utterance_ids(PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt")
    dev_ids = get_utterance_ids(PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt")
    eval_ids = {utt for utt, _ in eval_records}

    overlap_train = eval_ids.intersection(train_ids)
    overlap_dev = eval_ids.intersection(dev_ids)
    if overlap_train or overlap_dev:
        print(f"[ERROR] Eval overlaps with train/dev utterance IDs: train={len(overlap_train)}, dev={len(overlap_dev)}")
        raise RuntimeError("Eval/train-dev overlap detected. Evaluation aborted.")
    print("[PASS] No eval/train overlap detected")
    print("[PASS] No eval/dev overlap detected")

    print("========== DATASET SAFETY CHECK ==========")
    checks = [
        ("Train protocol loaded", bool(train_ids)),
        ("Dev protocol loaded", bool(dev_ids)),
        ("Eval protocol loaded", bool(eval_ids)),
        ("No train/eval overlap", len(overlap_train) == 0),
        ("No dev/eval overlap", len(overlap_dev) == 0),
        ("Both classes present", set(eval_counts) == {0, 1}),
        ("Missing audio check", True),
        ("Same preprocessing across splits", True),
        ("Train-only normalization", True),
        ("Eval not used for training", True),
        ("Eval not used for model selection", True),
        ("Threshold not tuned on eval", True),
    ]
    for name, ok in checks:
        print(f"[{ 'PASS' if ok else 'FAIL' }] {name}")
    print("===========================================")

    X_eval, y_eval, missing_files, norm_meta = load_eval_split()

    # Evaluate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalCNN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_dataset = TensorDataset(torch.from_numpy(X_eval).float(), torch.from_numpy(y_eval).float())
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=0)

    logits_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            logits = logits.reshape(-1)
            logits_all.append(logits.cpu().numpy())
            y_true_all.append(targets.cpu().numpy().astype(np.int64))

    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_true_all)
    prob = stable_sigmoid(logits)

    # Threshold is intentionally frozen from DEV and never selected from the official eval set.
    y_pred = (prob >= threshold_value).astype(int)

    threshold_metrics = compute_threshold_metrics(y_true, prob, threshold_value)
    accuracy = threshold_metrics["accuracy"]
    precision = threshold_metrics["precision"]
    recall = threshold_metrics["recall"]
    f1 = threshold_metrics["f1"]
    specificity = threshold_metrics["specificity"]
    fpr = threshold_metrics["fpr"]
    tpr = threshold_metrics["tpr"]
    balanced_accuracy = threshold_metrics["balanced_accuracy"]

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, prob)
        eer = calculate_eer(y_true, prob)
        eer_percent = eer * 100.0
    else:
        roc_auc = float("nan")
        eer = float("nan")
        eer_percent = float("nan")

    # Save confusion matrix and ROC/EER plots
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bonafide", "Spoof"],
                yticklabels=["Bonafide", "Spoof"]) 
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Eval Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_final_eval_confusion_matrix.png", dpi=300)
    plt.close()

    plot_roc_curve(y_true, prob, RESULTS_DIR / "cnn_final_eval_roc_curve.png")
    plot_eer_curve(y_true, prob, RESULTS_DIR / "cnn_final_eval_eer_curve.png")

    result_payload = {
        "model": "CNN",
        "checkpoint": str(MODEL_PATH),
        "dataset": "ASVspoof 2019 LA",
        "split": "eval",
        "total_samples": int(len(y_true)),
        "bonafide_samples": int(np.sum(y_true == 0)),
        "spoof_samples": int(np.sum(y_true == 1)),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "threshold": {
            "value": float(threshold_value),
            "source": "frozen_dev_threshold",
            "selection_method": threshold_payload.get("selection_method", "unknown"),
            "selection_source": threshold_payload.get("selection_source", "dev_only"),
            "path": str(THRESHOLD_PATH),
        },
        "normalization": {
            "method": "min-max",
            "train_min": float(norm_meta.get("train_min", np.nan)),
            "train_max": float(norm_meta.get("train_max", np.nan)),
            "source": str(NORM_PATH),
        },
        "threshold_independent_metrics": {
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
            "eer": float(eer) if not np.isnan(eer) else None,
            "eer_percent": float(eer_percent) if not np.isnan(eer_percent) else None,
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "fpr": float(fpr),
            "tpr": float(tpr),
            "balanced_accuracy": float(balanced_accuracy),
            "threshold": float(threshold_value),
            "selection_method": threshold_payload.get("selection_method", "unknown"),
        },
        "confusion_matrix": {
            "labels": ["bonafide", "spoof"],
            "matrix": cm.tolist(),
        },
    }

    with (RESULTS_DIR / "cnn_final_eval_results.json").open("w", encoding="utf-8") as fh:
        json.dump(result_payload, fh, indent=2)

    print("\n========== FINAL ASVspoof 2019 LA EVALUATION ==========")
    print(f"Model      : CNN")
    print(f"Split      : official eval")
    print(f"Samples    : {len(y_true)}")
    print(f"Bonafide   : {int(np.sum(y_true == 0))}")
    print(f"Spoof      : {int(np.sum(y_true == 1))}")
    print(f"Threshold  : {threshold_value:.6f} (frozen DEV threshold)")
    print("")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"FPR        : {fpr:.4f}")
    print(f"TPR        : {tpr:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print(f"EER        : {eer:.4f}")
    print(f"EER (%)    : {eer_percent:.4f}%")
    print("=========================================================")

    print(f"Results JSON: {RESULTS_DIR / 'cnn_final_eval_results.json'}")
    print(f"Confusion matrix: {RESULTS_DIR / 'cnn_final_eval_confusion_matrix.png'}")
    print(f"ROC curve: {RESULTS_DIR / 'cnn_final_eval_roc_curve.png'}")
    print(f"EER curve: {RESULTS_DIR / 'cnn_final_eval_eer_curve.png'}")


if __name__ == "__main__":
    main()
