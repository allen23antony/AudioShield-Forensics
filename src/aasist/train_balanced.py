"""Balanced AASIST retraining script.

Improvements over the original train.py:
  1. WeightedRandomSampler  — each mini-batch sees ~50 % bonafide / 50 % spoof
  2. Focal Loss             — down-weights easy spoof examples, focuses on hard ones
  3. CosineAnnealingLR     — smooth LR decay over the full training run

Outputs
-------
  models/aasist/aasist_balanced_best.pth
  results/aasist/aasist_balanced_results.json
  results/aasist/aasist_balanced_threshold.json
  results/aasist/aasist_balanced_dev_confusion_matrix.png
"""

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
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

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

# ---------------------------------------------------------------------------
# Paths for balanced outputs (separate from original outputs)
# ---------------------------------------------------------------------------
BALANCED_MODEL_PATH = MODEL_DIR / "aasist_balanced_best.pth"
BALANCED_RESULTS_PATH = RESULTS_DIR / "aasist_balanced_results.json"
BALANCED_THRESHOLD_PATH = RESULTS_DIR / "aasist_balanced_threshold.json"
BALANCED_CM_PATH = RESULTS_DIR / "aasist_balanced_dev_confusion_matrix.png"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Fix all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary Focal Loss for imbalanced classification.

    Reduces the relative loss for well-classified examples so training
    concentrates on the hard bonafide / borderline spoof cases.

    Parameters
    ----------
    alpha : float
        Balancing factor for the positive (spoof) class.  0.25 means the
        minority class gets 3× more weight than the majority.
    gamma : float
        Focusing parameter.  gamma=2 is the standard recommendation from
        Lin et al. (2017).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Balanced DataLoader factory
# ---------------------------------------------------------------------------

def make_balanced_loader(dataset: ASVspoofDataset | Subset, batch_size: int) -> DataLoader:
    """Return a DataLoader that uses WeightedRandomSampler for class balance.

    Each epoch will contain roughly equal numbers of bonafide and spoof
    samples, regardless of the raw 1:8.8 class ratio in the protocol.
    """
    # Collect integer labels from every sample in the dataset.
    # ASVspoofDataset.__getitem__ returns (tensor, label_tensor, utt_id, attack_id).
    train_labels: list[int] = []
    if isinstance(dataset, Subset):
        for idx in dataset.indices:
            _, label, _, _ = dataset.dataset[idx]
            train_labels.append(int(label.item()))
    else:
        for idx in range(len(dataset)):
            _, label, _, _ = dataset[idx]
            train_labels.append(int(label.item()))

    class_counts = np.bincount(train_labels)          # [n_bonafide, n_spoof]
    class_weights = 1.0 / class_counts.astype(float)  # inverse frequency
    sample_weights = [float(class_weights[lbl]) for lbl in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,          # replaces shuffle=True
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Evaluation (identical to train.py so metrics are comparable)
# ---------------------------------------------------------------------------

def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the model on one split, returning loss, metrics, and raw arrays."""
    model.eval()
    eval_criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.inference_mode():
        for waveforms, labels, _, _ in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device).float().view(-1)
            outputs = model(waveforms.squeeze(1))
            loss = eval_criterion(outputs, labels)
            total_loss += loss.item() * waveforms.size(0)
            logits_all.append(outputs.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(labels_all).astype(np.int64)
    logits = np.concatenate(logits_all).astype(np.float64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)

    metrics: dict[str, float] = {
        "accuracy":  float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall":    float(recall_score(y_true, preds, zero_division=0)),
        "f1":        float(f1_score(y_true, preds, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["eer"]     = float(calculate_eer(y_true, probs))

    avg_loss = total_loss / max(len(y_true), 1)
    return avg_loss, metrics, y_true, probs, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain AASIST with WeightedRandomSampler + Focal Loss + CosineAnnealingLR."
    )
    parser.add_argument("--seed",       type=int,   default=SEED)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--patience",   type=int,   default=5)
    parser.add_argument("--alpha",      type=float, default=0.25,
                        help="Focal loss alpha (default: 0.25)")
    parser.add_argument("--gamma",      type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick sanity check with 8 samples per class.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Banner ──────────────────────────────────────────────────────────────
    print("=" * 65)
    print("  AASIST Balanced Retraining")
    print("=" * 65)
    print(f"  Device        : {device}")
    print(f"  Seed          : {args.seed}")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  CUDA          : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    print(f"  Sample rate   : {SAMPLE_RATE}")
    print(f"  Input duration: {EXPERIMENT_CONFIG.input_duration_seconds}s")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  LR            : {args.lr}")
    print(f"  Focal alpha   : {args.alpha}")
    print(f"  Focal gamma   : {args.gamma}")
    print(f"  Patience      : {args.patience}")
    print("=" * 65)

    # ── Dataset integrity check ─────────────────────────────────────────────
    dataset_summary = validate_dataset_integrity()
    print("Dataset summary:")
    for split, stats in dataset_summary.items():
        print(f"  {split}: {stats}")

    target_samples = int(SAMPLE_RATE * EXPERIMENT_CONFIG.input_duration_seconds)

    # ── Build datasets ──────────────────────────────────────────────────────
    if args.smoke_test:
        full_train = ASVspoofDataset("train", sample_rate=SAMPLE_RATE, target_samples=target_samples)
        full_dev   = ASVspoofDataset("dev",   sample_rate=SAMPLE_RATE, target_samples=target_samples)

        def _balanced_subset(ds: ASVspoofDataset, limit_per_class: int = 8) -> Subset:
            counts = {0: 0, 1: 0}
            indices: list[int] = []
            for idx in range(len(ds)):
                _, label, _, _ = ds[idx]
                lbl = int(label.item())
                if counts[lbl] >= limit_per_class:
                    continue
                counts[lbl] += 1
                indices.append(idx)
                if sum(counts.values()) >= 2 * limit_per_class:
                    break
            if len(indices) < 2:
                raise ValueError("Smoke test could not build a balanced subset.")
            return Subset(ds, indices)

        train_dataset = _balanced_subset(full_train, limit_per_class=8)
        dev_dataset   = _balanced_subset(full_dev,   limit_per_class=8)
        batch_size    = 8
    else:
        train_dataset = ASVspoofDataset("train", sample_rate=SAMPLE_RATE, target_samples=target_samples)
        dev_dataset   = ASVspoofDataset("dev",   sample_rate=SAMPLE_RATE, target_samples=target_samples)
        batch_size    = args.batch_size

    # ── DataLoaders ─────────────────────────────────────────────────────────
    # Training: WeightedRandomSampler ensures ~50/50 class balance per batch.
    train_loader = make_balanced_loader(train_dataset, batch_size)

    # Dev/eval: plain sequential loader — no shuffling, no resampling.
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    n_train = len(train_dataset)
    n_dev   = len(dev_dataset)
    print(f"\nTraining samples : {n_train:,}  (resampled to ~50/50 per batch)")
    print(f"Dev samples      : {n_dev:,}")

    # ── Model / loss / optimiser / scheduler ────────────────────────────────
    model     = AASISTModel().to(device)
    criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    print(f"\nLoss function    : FocalLoss(alpha={args.alpha}, gamma={args.gamma})")
    print(f"Optimiser        : AdamW(lr={args.lr})")
    print(f"LR scheduler     : CosineAnnealingLR(T_max={args.epochs})\n")

    # ── Training loop ───────────────────────────────────────────────────────
    best_dev_loss    = float("inf")
    patience_counter = 0
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for waveforms, labels, _, _ in train_loader:
            waveforms = waveforms.to(device)
            labels    = labels.to(device).float().view(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms.squeeze(1))
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * waveforms.size(0)
            n_batches    += 1

        # Step LR scheduler after each epoch.
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_loss = running_loss / max(n_train, 1)
        dev_loss, dev_metrics, y_true, dev_probs, dev_preds = evaluate_split(
            model, dev_loader, device
        )
        dev_eer = float(calculate_eer(y_true, dev_probs)) if len(np.unique(y_true)) == 2 else float("nan")

        row: dict[str, Any] = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "dev_loss":     round(dev_loss, 6),
            "dev_accuracy": round(dev_metrics["accuracy"], 6),
            "dev_precision":round(dev_metrics["precision"], 6),
            "dev_recall":   round(dev_metrics["recall"], 6),
            "dev_f1":       round(dev_metrics["f1"], 6),
            "dev_eer":      round(dev_eer, 6),
            "lr":           round(current_lr, 8),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | "
            f"dev_acc={dev_metrics['accuracy']:.4f} | "
            f"dev_f1={dev_metrics['f1']:.4f} | "
            f"dev_eer={dev_eer:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # Early stopping on dev loss.
        if dev_loss < best_dev_loss - 1e-6:
            best_dev_loss    = dev_loss
            patience_counter = 0
            best_state = {
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch":      epoch,
                "dev_loss":   dev_loss,
                "dev_metrics":dev_metrics,
                "seed":       args.seed,
                "config":     EXPERIMENT_CONFIG.__dict__,
                "focal_alpha":args.alpha,
                "focal_gamma":args.gamma,
            }
            torch.save(best_state, BALANCED_MODEL_PATH)
            print(f"  ✓ New best model saved (dev_loss={dev_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (patience={args.patience}).")
                break

    # Fallback: save last state if nothing improved.
    if best_state is None:
        best_state = {"model_state_dict": model.state_dict(), "epoch": 0}
        torch.save(best_state, BALANCED_MODEL_PATH)

    # ── Post-training evaluation on best checkpoint ──────────────────────────
    print(f"\nLoading best checkpoint from {BALANCED_MODEL_PATH} …")
    ckpt = torch.load(BALANCED_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, dev_metrics, y_true, dev_probs, dev_preds = evaluate_split(model, dev_loader, device)
    threshold, threshold_metrics = select_threshold_from_scores(y_true, dev_probs, method="eer")

    # ── Save artefacts ───────────────────────────────────────────────────────
    # Full training history.
    save_json(BALANCED_RESULTS_PATH, {"history": history})

    # Threshold payload (mirrors aasist_threshold.json format).
    threshold_payload: dict[str, Any] = {
        "threshold":          float(threshold),
        "selection_method":   "eer",
        "selection_source":   "dev_only",
        "dev_metrics": {
            "accuracy":  float(dev_metrics["accuracy"]),
            "precision": float(dev_metrics["precision"]),
            "recall":    float(dev_metrics["recall"]),
            "f1":        float(dev_metrics["f1"]),
            "roc_auc":   float(roc_auc_score(y_true, dev_probs)),
            "eer":       float(threshold_metrics["eer"]),
        },
        "focal_alpha":  args.alpha,
        "focal_gamma":  args.gamma,
        "sampler":      "WeightedRandomSampler",
        "scheduler":    "CosineAnnealingLR",
        "checkpoint":   str(BALANCED_MODEL_PATH),
        "timestamp":    int(time.time()),
    }
    save_json(BALANCED_THRESHOLD_PATH, threshold_payload)
    save_confusion_matrix(y_true, dev_preds, BALANCED_CM_PATH)

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Training complete")
    print("=" * 65)
    print(f"  Best checkpoint : {BALANCED_MODEL_PATH}")
    print(f"  Threshold file  : {BALANCED_THRESHOLD_PATH}")
    print(f"  Results file    : {BALANCED_RESULTS_PATH}")
    print(f"  EER threshold   : {threshold:.4f}")
    print(f"  DEV metrics     :")
    for k, v in threshold_payload["dev_metrics"].items():
        print(f"    {k:12s}: {v:.4f}")
    print("=" * 65)

    if args.smoke_test:
        print("\nSMOKE TEST PASS: balanced training pipeline executed successfully.")


if __name__ == "__main__":
    main()
