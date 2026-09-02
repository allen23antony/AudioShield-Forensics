from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aasist.config import (
    BATCH_SIZE,
    EPOCHS,
    EXPECTED_SPLIT_COUNTS,
    EXPERIMENT_CONFIG,
    LEARNING_RATE,
    MODEL_DIR,
    RESULTS_DIR,
    SAMPLE_RATE,
    SEED,
)
from aasist.dataset import ASVspoofDataset, validate_dataset_integrity
from aasist.model import AASISTModel
from aasist.utils import calculate_eer, save_confusion_matrix, save_json, select_threshold_from_scores


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights() -> torch.Tensor:
    counts = EXPECTED_SPLIT_COUNTS["train"]
    neg_count = counts["bonafide"]
    pos_count = counts["spoof"]
    pos_weight = neg_count / max(pos_count, 1)
    return torch.tensor([pos_weight], dtype=torch.float32)


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.inference_mode():
        for waveforms, labels, _, _ in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device).float().view(-1)
            outputs = model(waveforms.squeeze(1))
            loss = criterion(outputs, labels)
            total_loss += loss.item() * waveforms.size(0)
            logits_all.append(outputs.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(labels_all).astype(np.int64)
    logits = np.concatenate(logits_all).astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["eer"] = float(calculate_eer(y_true, probs))
    avg_loss = total_loss / max(len(y_true), 1)
    return avg_loss, metrics, y_true, probs, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AASIST anti-spoofing model on ASVspoof 2019 LA.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        args.smoke_test = True

    set_seed(args.seed)
    device = get_device()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Seed:", args.seed)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Sampling rate:", SAMPLE_RATE)
    print("Input duration:", EXPERIMENT_CONFIG.input_duration_seconds)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)

    dataset_summary = validate_dataset_integrity()
    print("Dataset summary:")
    for split, stats in dataset_summary.items():
        print(f"  {split}: {stats}")

    if args.smoke_test:
        train_dataset = ASVspoofDataset("train", sample_rate=SAMPLE_RATE, target_samples=int(SAMPLE_RATE * EXPERIMENT_CONFIG.input_duration_seconds))
        dev_dataset = ASVspoofDataset("dev", sample_rate=SAMPLE_RATE, target_samples=int(SAMPLE_RATE * EXPERIMENT_CONFIG.input_duration_seconds))

        def balanced_subset(dataset: ASVspoofDataset, limit_per_class: int = 8) -> Subset:
            counts = {0: 0, 1: 0}
            indices: list[int] = []
            for idx in range(len(dataset)):
                _, label, _, _ = dataset[idx]
                label_value = int(label.item())
                if counts[label_value] >= limit_per_class:
                    continue
                counts[label_value] += 1
                indices.append(idx)
                if sum(counts.values()) >= 2 * limit_per_class:
                    break
            if len(indices) < 2:
                raise ValueError("Smoke test could not construct a balanced subset from the dataset.")
            return Subset(dataset, indices)

        train_dataset = balanced_subset(train_dataset, limit_per_class=8)
        dev_dataset = balanced_subset(dev_dataset, limit_per_class=8)
        batch_size = 8
    else:
        train_dataset = ASVspoofDataset("train", sample_rate=SAMPLE_RATE, target_samples=int(SAMPLE_RATE * EXPERIMENT_CONFIG.input_duration_seconds))
        dev_dataset = ASVspoofDataset("dev", sample_rate=SAMPLE_RATE, target_samples=int(SAMPLE_RATE * EXPERIMENT_CONFIG.input_duration_seconds))
        batch_size = args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = AASISTModel().to(device)
    pos_weight = compute_class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dev_loss = float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for waveforms, labels, _, _ in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device).float().view(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms.squeeze(1))
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * waveforms.size(0)

        train_loss = running_loss / max(len(train_dataset), 1)
        dev_loss, dev_metrics, y_true, dev_probs, dev_preds = evaluate_split(model, dev_loader, device)
        dev_roc_auc = float(roc_auc_score(y_true, dev_probs)) if len(np.unique(y_true)) == 2 else float("nan")
        dev_eer = float(calculate_eer(y_true, dev_probs)) if len(np.unique(y_true)) == 2 else float("nan")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_precision": dev_metrics["precision"],
            "dev_recall": dev_metrics["recall"],
            "dev_f1": dev_metrics["f1"],
            "dev_roc_auc": dev_roc_auc,
            "dev_eer": dev_eer,
        })
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f} | dev_acc={dev_metrics['accuracy']:.4f} | dev_f1={dev_metrics['f1']:.4f}")

        if dev_loss < best_dev_loss - 1e-6:
            best_dev_loss = dev_loss
            patience = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "dev_loss": dev_loss,
                "seed": args.seed,
                "config": EXPERIMENT_CONFIG.__dict__,
            }
            torch.save(best_state, MODEL_DIR / "aasist_best.pth")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch} based on DEV loss.")
                break

    if best_state is None:
        best_state = {"model_state_dict": model.state_dict(), "epoch": 0}
        torch.save(best_state, MODEL_DIR / "aasist_best.pth")

    with (RESULTS_DIR / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    checkpoint = torch.load(MODEL_DIR / "aasist_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, dev_metrics, y_true, dev_probs, dev_preds = evaluate_split(model, dev_loader, device)
    threshold, threshold_metrics = select_threshold_from_scores(y_true, dev_probs, method="eer")
    threshold_payload = {
        "threshold": float(threshold),
        "selection_method": "eer",
        "selection_source": "dev_only",
        "dev_metrics": {
            "accuracy": float(dev_metrics["accuracy"]),
            "precision": float(dev_metrics["precision"]),
            "recall": float(dev_metrics["recall"]),
            "f1": float(dev_metrics["f1"]),
            "roc_auc": float(roc_auc_score(y_true, dev_probs)),
            "eer": float(threshold_metrics["eer"]),
        },
        "checkpoint": str(MODEL_DIR / "aasist_best.pth"),
        "timestamp": int(time.time()),
    }
    save_json(RESULTS_DIR / "aasist_threshold.json", threshold_payload)
    save_confusion_matrix(y_true, dev_preds, RESULTS_DIR / "aasist_dev_confusion_matrix.png")

    print("Best checkpoint saved to:", MODEL_DIR / "aasist_best.pth")
    print("DEV threshold saved to:", RESULTS_DIR / "aasist_threshold.json")
    print("DEV metrics:", json.dumps(threshold_payload["dev_metrics"], indent=2))

    if args.smoke_test:
        print("SMOKE TEST PASS: AASIST model initialization, forward pass, backward pass, optimizer update, and checkpoint save/load all succeeded.")
        return

    print("AASIST training complete. Threshold selection used DEV ONLY and was frozen before evaluation.")


if __name__ == "__main__":
    main()
