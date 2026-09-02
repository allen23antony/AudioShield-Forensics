from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aasist.config import ATTACK_IDS, RESULTS_DIR
from aasist.utils import calculate_eer, compute_threshold_metrics

CACHE_PATH = RESULTS_DIR / "aasist_eval_scores.npz"
THRESHOLD_PATH = RESULTS_DIR / "aasist_threshold.json"


def load_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Eval cache not found at {CACHE_PATH}. Run evaluate.py first.")
    payload = np.load(CACHE_PATH)
    utterance_ids = np.asarray(payload["utterance_ids"], dtype=str)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    attack_ids = np.asarray(payload["attack_ids"], dtype=str)
    scores = np.asarray(payload["scores"], dtype=np.float64)
    return utterance_ids, labels, attack_ids, scores


def load_threshold() -> float:
    with THRESHOLD_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("selection_source") != "dev_only":
        raise ValueError("Threshold selection source is not dev_only; aborting attack analysis.")
    return float(payload["threshold"])


def summarize_attack(labels: np.ndarray, scores: np.ndarray, attack_ids: np.ndarray, attack: str, threshold: float) -> dict[str, object]:
    spoof_mask = (attack_ids == attack) & (labels == 1)
    bonafide_mask = labels == 0
    combined_mask = bonafide_mask | spoof_mask
    y_true = labels[combined_mask]
    y_scores = scores[combined_mask]
    y_pred = (y_scores >= threshold).astype(np.int64)
    metrics = compute_threshold_metrics(y_true, y_scores, threshold)
    roc_auc = None
    eer = None
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(y_true, y_scores))
        eer = float(calculate_eer(y_true, y_scores))
    return {
        "attack": attack,
        "total_samples": int(len(y_true)),
        "bonafide_samples": int((y_true == 0).sum()),
        "spoof_samples": int((y_true == 1).sum()),
        "tp": int(metrics["tp"]),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "specificity": float(metrics["specificity"]),
        "fpr": float(metrics["fpr"]),
        "tpr": float(metrics["tpr"]),
        "roc_auc": roc_auc,
        "eer": eer,
    }


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attack",
        "total_samples",
        "bonafide_samples",
        "spoof_samples",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "fpr",
        "tpr",
        "roc_auc",
        "eer",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def make_plot(values: list[float], labels: list[str], title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="steelblue")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    utterance_ids, labels, attack_ids, scores = load_cache()
    threshold = load_threshold()
    rows = [summarize_attack(labels, scores, attack_ids, attack, threshold) for attack in ATTACK_IDS]
    output = {
        "threshold": float(threshold),
        "selection_source": "dev_only",
        "attack_analysis": rows,
    }
    out_json = RESULTS_DIR / "aasist_attack_analysis.json"
    out_csv = RESULTS_DIR / "aasist_attack_analysis.csv"
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    save_csv(rows, out_csv)

    f1_values = [float(row["f1"]) for row in rows]
    recall_values = [float(row["recall"]) for row in rows]
    eer_values = [float(row["eer"]) if row["eer"] is not None else 0.0 for row in rows]
    attack_names = [row["attack"] for row in rows]
    make_plot(f1_values, attack_names, "F1 by Attack Family (AASIST)", "F1", RESULTS_DIR / "aasist_attack_f1.png")
    make_plot(recall_values, attack_names, "Recall by Attack Family (AASIST)", "Recall", RESULTS_DIR / "aasist_attack_recall.png")
    make_plot(eer_values, attack_names, "EER by Attack Family (AASIST)", "EER", RESULTS_DIR / "aasist_attack_eer.png")

    print("AASIST attack analysis complete.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
