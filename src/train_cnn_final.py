"""Final CNN training with architecture and preprocessing fixes.

This script implements a compact, stable CNN for mel-spectrogram inputs with
critical fixes applied:
 - LeakyReLU activations to avoid dying ReLUs
 - Proper weight initialization (kaiming_uniform for convs, xavier_uniform for linears)
 - Min-max normalization of mel-spectrograms to [0,1]
 - Gradient clipping to avoid exploding gradients
 - Smaller, proven architecture for stable learning

Designed to run a quick test on a limited number of samples (default 1000).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import argparse
import time
from datetime import datetime, timezone
from collections import Counter

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
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


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "LA"
TRAIN_AUDIO_DIR = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO_DIR = DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac"
PROTOCOL_DIR = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols"
MODEL_DIR = PROJECT_ROOT / "models" / "cnn"
RESULTS_DIR = PROJECT_ROOT / "results"

# Audio / spectrogram params
SR = 16000
TARGET_SECONDS = 5.0
TARGET_SAMPLES = int(SR * TARGET_SECONDS)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 157

# Training params for quick test
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 10
MAX_SAMPLES_TEST = 1000  # limit for quick verification


def load_protocol(protocol_path: Path) -> List[Tuple[str, int]]:
    records: List[Tuple[str, int]] = []
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    with protocol_path.open("r", encoding="utf-8") as fh:
        for ln, raw in enumerate(fh, start=1):
            s = raw.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) == 4:
                _, utt, attack, label = parts
            elif len(parts) == 5:
                _, utt, dash, attack, label = parts
            else:
                print(f"[WARN] Skipping malformed protocol line {ln}: {s}")
                continue
            if label not in {"bonafide", "spoof"}:
                print(f"[WARN] Unexpected label '{label}' at line {ln}; skipping")
                continue
            records.append((utt, 0 if label == "bonafide" else 1))
    return records


def preprocess_audio(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if y.shape[0] < TARGET_SAMPLES:
        y = librosa.util.fix_length(y, size=TARGET_SAMPLES)
    else:
        y = y[:TARGET_SAMPLES]
    return y.astype(np.float32)


def extract_spectrogram(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = preprocess_audio(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # pad/truncate frames
    frames = mel_db.shape[1]
    if frames < TARGET_FRAMES:
        pad = TARGET_FRAMES - frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=(mel_db.min(),))
    elif frames > TARGET_FRAMES:
        mel_db = mel_db[:, :TARGET_FRAMES]
    assert mel_db.shape == (N_MELS, TARGET_FRAMES)
    return mel_db.astype(np.float32)


LABEL_MAP = {"bonafide": 0, "spoof": 1}


def _stratified_sample(records: List[Tuple[str, int]], n_per_class: int, seed: int = 42) -> List[Tuple[str,int]]:
    """Return a balanced list with n_per_class samples per class (if available).

    If there are fewer than n_per_class for a class, use all available and
    print a warning. Sampling is reproducible using the provided seed.
    """
    import random
    random.seed(seed)

    by_class = {}
    for utt, label in records:
        by_class.setdefault(label, []).append((utt, label))

    sampled: List[Tuple[str,int]] = []
    for label, items in by_class.items():
        if len(items) < n_per_class:
            print(f"[WARN] Only {len(items)} samples available for label {label}; requested {n_per_class}")
            chosen = items[:]
        else:
            chosen = random.sample(items, n_per_class)
        sampled.extend(chosen)

    # Shuffle combined list deterministically
    random.shuffle(sampled)
    return sampled


def load_dataset(split: str, max_samples: int | None = None, balanced: bool = False, n_per_class: int | None = None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset split. When balanced=True, perform stratified sampling.

    Parameters:
        split: 'train' or 'dev'
        max_samples: legacy cap ignored when balanced=True
        balanced: whether to perform stratified balanced sampling
        n_per_class: number of samples per class when balanced
        seed: RNG seed for reproducibility
    """
    if split not in {"train", "dev"}:
        raise ValueError("split must be 'train' or 'dev'")
    protocol_file = PROTOCOL_DIR / ("ASVspoof2019.LA.cm.train.trn.txt" if split == "train" else "ASVspoof2019.LA.cm.dev.trl.txt")
    records = load_protocol(protocol_file)

    if balanced:
        if n_per_class is None:
            raise ValueError("n_per_class must be provided when balanced=True")
        records = _stratified_sample(records, n_per_class, seed=seed)
    elif max_samples is not None:
        records = records[:max_samples]

    audio_dir = TRAIN_AUDIO_DIR if split == "train" else DEV_AUDIO_DIR
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    total = len(records)
    for idx, (utt, label) in enumerate(records, start=1):
        path = audio_dir / f"{utt}.flac"
        if not path.exists():
            print(f"[WARN] Missing audio {path}")
            continue
        try:
            spec = extract_spectrogram(path)
        except Exception as exc:
            print(f"[WARN] Failed {path}: {exc}")
            continue
        X_list.append(spec)
        y_list.append(label)
        if idx % 1000 == 0:
            print(f"  Loaded {idx}/{total} files for {split}...")
    if not X_list:
        raise RuntimeError(f"No data loaded for split {split}")
    X = np.stack(X_list, axis=0)  # (N, 128, frames)
    X = X[:, np.newaxis, :, :]  # (N, 1, 128, frames)
    y = np.asarray(y_list, dtype=np.int64)
    print(f"[INFO] Loaded {X.shape[0]} samples for split {split}")
    return X, y


class FinalCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Using epsilon and momentum as required
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 19, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            # NOTE: do not apply Sigmoid here. Use BCEWithLogitsLoss for numerical stability
        )
        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


try:
    from scipy.special import expit as sigmoid
except Exception:  # pragma: no cover - fallback when SciPy is unavailable
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-x))


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid. Keeps probability calibration identical without overflow."""
    return np.asarray(sigmoid(np.asarray(logits, dtype=np.float64)), dtype=np.float32)


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute EER robustly. Returns NaN if undefined (e.g., single-class labels)."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    unique = np.unique(y_true)
    if unique.size != 2:
        # EER is undefined when there is not both positive and negative classes
        raise ValueError("EER requires both classes present in y_true")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    # find point where abs(fpr - fnr) is minimized
    diffs = np.abs(fpr - fnr)
    idx = int(np.argmin(diffs))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer)


THRESHOLD_PATH = RESULTS_DIR / "cnn_final_threshold.json"


def compute_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict:
    """Return threshold-dependent metrics for a single operating point."""
    y_pred = (np.asarray(y_scores) >= float(threshold)).astype(int)
    y_true = np.asarray(y_true, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_neg = tn + fp
    total_pos = tp + fn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = (tn / total_neg) if total_neg > 0 else 0.0
    tpr = (tp / total_pos) if total_pos > 0 else 0.0
    fpr = (fp / total_neg) if total_neg > 0 else 0.0
    balanced_accuracy = 0.5 * (tpr + specificity)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "specificity": float(specificity),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def candidate_thresholds(y_scores: np.ndarray) -> np.ndarray:
    """Enumerate candidate thresholds from the DEV score distribution, including 0.5."""
    unique_scores = np.unique(np.asarray(y_scores, dtype=np.float64))
    grid = np.linspace(0.0, 1.0, 1001)
    combined = np.concatenate([np.array([0.5], dtype=np.float64), unique_scores, grid])
    return np.unique(np.clip(combined, 0.0, 1.0))


def select_threshold_from_dev(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, str, dict, dict, dict, dict, dict]:
    """Choose a single operating threshold using only DEV data.

    The criterion is intentionally tied to spoof detection: maximize balanced accuracy
    while requiring a high spoof recall and keeping the bonafide false-positive rate
    under a conservative budget. This avoids optimizing on the official ASVspoof eval
    split, which is held out for the final, unbiased report.
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("Threshold selection requires both bonafide and spoof labels in DEV.")

    candidates = [compute_threshold_metrics(y_true, y_scores, float(th)) for th in candidate_thresholds(y_scores)]
    default_metric = compute_threshold_metrics(y_true, y_scores, 0.5)
    youden_metric = max(candidates, key=lambda m: (m["tpr"] - m["fpr"]))
    f1_metric = max(candidates, key=lambda m: m["f1"])
    balanced_metric = max(candidates, key=lambda m: m["balanced_accuracy"])
    budgeted = [
        m for m in candidates
        if m["fpr"] <= 0.05 and m["tpr"] >= 0.95
    ]
    if budgeted:
        selected = max(budgeted, key=lambda m: (m["balanced_accuracy"], m["f1"], m["tpr"]))
        selection_method = "balanced_accuracy_with_spoof_recall_constraint"
    else:
        selected = max(candidates, key=lambda m: (m["balanced_accuracy"], m["f1"], m["tpr"] - 0.5 * m["fpr"]))
        selection_method = "balanced_accuracy_with_spoof_recall_fallback"

    return (
        float(selected["threshold"]),
        selection_method,
        default_metric,
        youden_metric,
        f1_metric,
        balanced_metric,
        selected,
    )


def save_threshold_summary(threshold: float, selection_method: str, threshold_metrics: dict) -> None:
    """Persist the frozen DEV threshold and the DEV metrics used to select it."""
    payload = {
        "threshold": float(threshold),
        "selection_method": selection_method,
        "selection_source": "dev_only",
        "dev_accuracy": float(threshold_metrics["accuracy"]),
        "dev_precision": float(threshold_metrics["precision"]),
        "dev_recall": float(threshold_metrics["recall"]),
        "dev_f1": float(threshold_metrics["f1"]),
        "dev_balanced_accuracy": float(threshold_metrics["balanced_accuracy"]),
        "dev_specificity": float(threshold_metrics["specificity"]),
        "dev_fpr": float(threshold_metrics["fpr"]),
        "dev_tpr": float(threshold_metrics["tpr"]),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with THRESHOLD_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN final (quick or full ASVspoof modes)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run quick balanced test (default if no flag provided)")
    group.add_argument("--full", action="store_true", help="Run full training on entire protocol (careful: long)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides mode defaults)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (validation loss)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (detailed logits/probs)")
    args = parser.parse_args()

    MODE = "quick" if (args.quick or not args.full) else "full"
    SEED = int(args.seed)

    # set mode defaults
    epochs = args.epochs if args.epochs is not None else (10 if MODE == "quick" else 20)
    batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    lr = args.lr if args.lr is not None else LR
    patience = int(args.patience)
    debug = bool(args.debug)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Protocol files
    train_protocol = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_protocol = PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"

    # Read protocols for counts and leakage check (do not load eval audio)
    train_records_all = load_protocol(train_protocol)
    dev_records_all = load_protocol(dev_protocol)
    eval_records_all = load_protocol(eval_protocol)

    train_ids = {utt for utt, _ in train_records_all}
    dev_ids = {utt for utt, _ in dev_records_all}
    eval_ids = {utt for utt, _ in eval_records_all}

    # Dataset safety checks
    overlap = train_ids.intersection(dev_ids)
    if overlap:
        print(f"[ERROR] Found {len(overlap)} overlapping utterance IDs between train and dev. Examples: {list(overlap)[:5]}")
        raise RuntimeError("Train/dev overlap detected - aborting to prevent leakage")

    print("[PASS] No train/dev utterance overlap detected.")

    # Print dataset summary (counts and percentages)
    def counts_and_pct(records):
        total = len(records)
        c = Counter(label for _, label in records)
        bon = c.get(0, 0)
        spo = c.get(1, 0)
        return total, bon, spo

    t_total, t_bon, t_spo = counts_and_pct(train_records_all)
    d_total, d_bon, d_spo = counts_and_pct(dev_records_all)
    e_total, e_bon, e_spo = counts_and_pct(eval_records_all)

    print("\n========== DATASET SUMMARY ==========")
    print(f"Train total: {t_total}")
    print(f"  bonafide: {t_bon} ({t_bon / t_total * 100:.2f}%)")
    print(f"  spoof:    {t_spo} ({t_spo / t_total * 100:.2f}%)")
    print("")
    print(f"Dev total: {d_total}")
    print(f"  bonafide: {d_bon} ({d_bon / d_total * 100:.2f}%)")
    print(f"  spoof:    {d_spo} ({d_spo / d_total * 100:.2f}%)")
    print("")
    print(f"Eval total: {e_total}  (protocol statistics only — NOT USED for training/validation)")
    print(f"  bonafide: {e_bon} ({e_bon / e_total * 100:.2f}%)")
    print(f"  spoof:    {e_spo} ({e_spo / e_total * 100:.2f}%)")
    print("====================================================\n")

    # Missing audio check (quick existence check using protocol lists)
    def missing_audio(records, audio_dir):
        missing = []
        for utt, _ in records:
            if not (audio_dir / f"{utt}.flac").exists():
                missing.append(utt)
        return missing

    missing_train = missing_audio(train_records_all, TRAIN_AUDIO_DIR)
    missing_dev = missing_audio(dev_records_all, DEV_AUDIO_DIR)

    if missing_train or missing_dev:
        print(f"[ERROR] Missing audio files - train missing: {len(missing_train)}, dev missing: {len(missing_dev)}")
        raise RuntimeError("Missing audio files detected; aborting")

    print("[PASS] Missing audio check passed")

    # Load datasets
    if MODE == "quick":
        print("[QUICK TEST]")
        print("WARNING: Quick-mode metrics are for pipeline validation only and are NOT official ASVspoof benchmark results.")
        desired_train_per_class = 500
        desired_dev_per_class = 250
        X_train, y_train = load_dataset("train", balanced=True, n_per_class=desired_train_per_class, seed=SEED)
        X_dev, y_dev = load_dataset("dev", balanced=True, n_per_class=desired_dev_per_class, seed=SEED)
    else:
        print("[FULL ASVSPOOF 2019 LA TRAINING]")
        print("[INFO] Loading full train/dev datasets (this may take significant RAM and time)")
        X_train, y_train = load_dataset("train", balanced=False)
        X_dev, y_dev = load_dataset("dev", balanced=False)

    # Post-load validations
    train_counts = Counter(y_train.tolist())
    dev_counts = Counter(y_dev.tolist())

    if len(train_counts) < 2:
        raise ValueError("TRAIN dataset contains only one class. Expected both bonafide and spoof samples.")
    if len(dev_counts) < 2:
        raise ValueError("DEV dataset contains only one class. Expected both bonafide and spoof samples.")

    print(f"Train raw shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"Dev raw shape:   {X_dev.shape}, dtype: {X_dev.dtype}")

    # Normalization (train-only) with clipping to preserve the training feature range
    # and prevent eval/dev samples outside the train dynamic range from producing
    # unintended negative values after scaling.
    global_min = float(np.nanmin(X_train))
    global_max = float(np.nanmax(X_train))
    print(f"[INFO] min-max scaling using train min={global_min:.6f}, max={global_max:.6f}")
    eps = 1e-8

    def normalize_feature_array(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, global_min, global_max)
        normed = (clipped - global_min) / (global_max - global_min + eps)
        return np.clip(normed, 0.0, 1.0).astype(np.float32)

    print("\n========== DATA PREPROCESSING VALIDATION ==========")
    X_train_valid = normalize_feature_array(X_train)
    X_dev_valid = normalize_feature_array(X_dev)
    train_range_out = int(np.count_nonzero((X_train_valid < 0.0) | (X_train_valid > 1.0)))
    dev_range_out = int(np.count_nonzero((X_dev_valid < 0.0) | (X_dev_valid > 1.0)))
    train_nan_inf = int(np.count_nonzero(np.isnan(X_train_valid) | np.isinf(X_train_valid)))
    dev_nan_inf = int(np.count_nonzero(np.isnan(X_dev_valid) | np.isinf(X_dev_valid)))
    print(f"[PASS] Train normalized range: [{float(X_train_valid.min()):.6f}, {float(X_train_valid.max()):.6f}]")
    print(f"[PASS] Dev normalized range:   [{float(X_dev_valid.min()):.6f}, {float(X_dev_valid.max()):.6f}]")
    print(f"[PASS] Train out-of-range count: {train_range_out}")
    print(f"[PASS] Dev out-of-range count:   {dev_range_out}")
    print(f"[PASS] Train NaN/Inf count: {train_nan_inf}")
    print(f"[PASS] Dev NaN/Inf count:   {dev_nan_inf}")
    print("[PASS] Clipping enabled: True")
    print("[PASS] Train statistics only used for normalization")
    print("====================================================")

    X_train = X_train_valid
    X_dev = X_dev_valid

    if np.isnan(X_train).any() or np.isinf(X_train).any():
        raise ValueError("Normalization produced NaN or Inf in X_train")
    if np.isnan(X_dev).any() or np.isinf(X_dev).any():
        raise ValueError("Normalization produced NaN or Inf in X_dev")

    # Save normalization stats
    norm_info = {
        "method": "min-max",
        "train_min": global_min,
        "train_max": global_max,
        "clipping_enabled": True,
        "normalization_formula": "clip((X - train_min) / (train_max - train_min + 1e-8), 0, 1)",
        "seed": SEED,
        "mode": MODE,
    }
    norm_file = RESULTS_DIR / "cnn_final_norm.json"
    with norm_file.open("w", encoding="utf-8") as fh:
        json.dump(norm_info, fh, indent=2)

    # Convert to CPU tensors only (do NOT move entire dataset to device)
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).unsqueeze(1).float()
    X_dev_t = torch.from_numpy(X_dev).float()
    y_dev_t = torch.from_numpy(y_dev).unsqueeze(1).float()

    # Diagnostics
    print(f"Train tensor shape (on CPU): {X_train_t.shape}")
    print(f"Dev tensor shape (on CPU):   {X_dev_t.shape}")
    print(f"Train min/max after norm: {float(X_train.min()):.6f}/{float(X_train.max()):.6f}")

    # DataLoaders (use generator for reproducible shuffling)
    g = torch.Generator()
    g.manual_seed(SEED)

    train_ds = TensorDataset(X_train_t, y_train_t)
    dev_ds = TensorDataset(X_dev_t, y_dev_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, criterion, optimizer
    model = FinalCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_path = MODEL_DIR / 'cnn_final_best.pth'
    best_epoch = None

    # Training loop with early stopping
    history = []
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        train_n = 0

        for batch_i, (inputs, targets) in enumerate(train_loader, start=1):
            # Move batch to device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_acc += loss.item() * inputs.size(0)
            train_n += inputs.size(0)

            if debug and batch_i == 1:
                logits_vals = logits.detach().cpu().numpy().reshape(-1)
                probs = torch.sigmoid(torch.from_numpy(logits_vals)).numpy()
                print(f"[DEBUG] After first batch logits (epoch {epoch}): min={logits_vals.min():.6f}, max={logits_vals.max():.6f}, mean={logits_vals.mean():.6f}")
                print(f"[DEBUG] After first batch probs  (epoch {epoch}): min={probs.min():.6f}, max={probs.max():.6f}, mean={probs.mean():.6f}")

        train_loss = train_loss_acc / max(1, train_n)

        # Validation
        model.eval()
        val_loss_acc = 0.0
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in dev_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(inputs)
                val_loss_acc += criterion(logits, targets).item() * inputs.size(0)
                all_logits.append(logits.cpu().numpy().reshape(-1))
                all_targets.append(targets.cpu().numpy().reshape(-1))
        val_loss = val_loss_acc / max(1, len(dev_ds))
        logits_concat = np.concatenate(all_logits)
        probs_concat = stable_sigmoid(logits_concat)
        y_true = np.concatenate(all_targets)
        y_pred = (probs_concat >= 0.5).astype(int)

        val_acc = float(accuracy_score(y_true, y_pred))

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        # update history
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss), "val_accuracy": float(val_acc), "learning_rate": float(lr_now)})

        # model checkpointing + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            best_epoch = epoch
            epochs_no_improve = 0
            print(f"  ✅ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Save training history
    history_file = RESULTS_DIR / "cnn_final_training_history.json"
    with history_file.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    # Final safety: ensure a best model exists
    if best_epoch is None:
        print("Warning: no best epoch saved; aborting final evaluation.")
        return

    # Final evaluation using best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    all_logits = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dev_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs).cpu().numpy().reshape(-1)
            all_logits.append(logits)
            all_targets.append(targets.cpu().numpy().reshape(-1))

    logits_concat = np.concatenate(all_logits)
    probs_concat = stable_sigmoid(logits_concat)
    y_true = np.concatenate(all_targets)

    # Freeze the classification threshold on DEV only before any official eval is scored.
    # This keeps the final evaluation unbiased and prevents tuning against the hidden
    # ASVspoof 2019 LA eval set.
    threshold_value, selection_method, default_metric, youden_metric, f1_metric, balanced_metric, selected_metric = select_threshold_from_dev(y_true, probs_concat)
    selected_threshold_metrics = compute_threshold_metrics(y_true, probs_concat, threshold_value)
    save_threshold_summary(threshold_value, selection_method, selected_threshold_metrics)

    print("\n# ========== THRESHOLD SELECTION ==========")
    print("Threshold source: DEV ONLY")
    print(f"Selection method: {selection_method}")
    print(f"Selected threshold: {threshold_value:.6f}")
    print(f"Default threshold 0.500000 -> accuracy={default_metric['accuracy']:.4f}, precision={default_metric['precision']:.4f}, recall={default_metric['recall']:.4f}, f1={default_metric['f1']:.4f}, balanced_accuracy={default_metric['balanced_accuracy']:.4f}, fpr={default_metric['fpr']:.4f}, tpr={default_metric['tpr']:.4f}")
    print(f"Youden J threshold {youden_metric['threshold']:.6f} -> accuracy={youden_metric['accuracy']:.4f}, precision={youden_metric['precision']:.4f}, recall={youden_metric['recall']:.4f}, f1={youden_metric['f1']:.4f}, balanced_accuracy={youden_metric['balanced_accuracy']:.4f}, fpr={youden_metric['fpr']:.4f}, tpr={youden_metric['tpr']:.4f}")
    print(f"F1-max threshold {f1_metric['threshold']:.6f} -> accuracy={f1_metric['accuracy']:.4f}, precision={f1_metric['precision']:.4f}, recall={f1_metric['recall']:.4f}, f1={f1_metric['f1']:.4f}, balanced_accuracy={f1_metric['balanced_accuracy']:.4f}, fpr={f1_metric['fpr']:.4f}, tpr={f1_metric['tpr']:.4f}")
    print(f"Balanced-accuracy-max threshold {balanced_metric['threshold']:.6f} -> accuracy={balanced_metric['accuracy']:.4f}, precision={balanced_metric['precision']:.4f}, recall={balanced_metric['recall']:.4f}, f1={balanced_metric['f1']:.4f}, balanced_accuracy={balanced_metric['balanced_accuracy']:.4f}, fpr={balanced_metric['fpr']:.4f}, tpr={balanced_metric['tpr']:.4f}")
    print(f"High-recall/FPR-controlled threshold {selected_metric['threshold']:.6f} -> accuracy={selected_metric['accuracy']:.4f}, precision={selected_metric['precision']:.4f}, recall={selected_metric['recall']:.4f}, f1={selected_metric['f1']:.4f}, balanced_accuracy={selected_metric['balanced_accuracy']:.4f}, fpr={selected_metric['fpr']:.4f}, tpr={selected_metric['tpr']:.4f}")
    print("# ===============================================")

    y_pred = (probs_concat >= threshold_value).astype(int)

    # Compute metrics safely
    def safe_metric(func, y_true, y_pred=None, y_scores=None):
        try:
            if y_scores is not None:
                return float(func(y_true, y_scores))
            return float(func(y_true, y_pred))
        except Exception:
            return float('nan')

    accuracy = safe_metric(accuracy_score, y_true=y_true, y_pred=y_pred)
    precision = safe_metric(precision_score, y_true=y_true, y_pred=y_pred)
    recall = safe_metric(recall_score, y_true=y_true, y_pred=y_pred)
    f1 = safe_metric(f1_score, y_true=y_true, y_pred=y_pred)
    balanced_accuracy = safe_metric(balanced_accuracy_score, y_true=y_true, y_pred=y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = (tn / (tn + fp)) if (tn + fp) else 0.0
    fpr = (fp / (tn + fp)) if (tn + fp) else 0.0
    tpr = (tp / (tp + fn)) if (tp + fn) else 0.0
    roc_auc = float('nan')
    eer = float('nan')
    if len(np.unique(y_true)) == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, probs_concat))
        except Exception:
            roc_auc = float('nan')
        try:
            eer = float(calculate_eer(y_true, probs_concat))
        except Exception:
            eer = float('nan')
    else:
        print("[WARN] Only one class present in DEV; ROC-AUC and EER are undefined.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Save final results JSON (structured)
    results = {
        "dataset_mode": MODE,
        "seed": SEED,
        "train_samples": int(len(y_train)),
        "dev_samples": int(len(y_dev)),
        "eval_samples": int(len(eval_records_all)),
        "train_bonafide": int(train_counts.get(0, 0)),
        "train_spoof": int(train_counts.get(1, 0)),
        "dev_bonafide": int(dev_counts.get(0, 0)),
        "dev_spoof": int(dev_counts.get(1, 0)),
        "eval_bonafide": int(e_bon),
        "eval_spoof": int(e_spo),
        "normalization": norm_info,
        "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "patience": patience},
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "threshold_selection": {
            "source": "dev_only",
            "selection_method": selection_method,
            "threshold": float(threshold_value),
            "threshold_file": str(THRESHOLD_PATH),
            "default_threshold": {"value": 0.5, **compute_threshold_metrics(y_true, probs_concat, 0.5)},
            "youden_j": {"threshold": youden_metric["threshold"], **{k: v for k, v in youden_metric.items() if k != "threshold"}},
            "f1_max": {"threshold": f1_metric["threshold"], **{k: v for k, v in f1_metric.items() if k != "threshold"}},
            "balanced_accuracy_max": {"threshold": balanced_metric["threshold"], **{k: v for k, v in balanced_metric.items() if k != "threshold"}},
            "selected": {"threshold": selected_metric["threshold"], **{k: v for k, v in selected_metric.items() if k != "threshold"}},
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "balanced_accuracy": balanced_accuracy,
            "specificity": specificity,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": float(threshold_value),
            "selection_method": selection_method,
        },
        "threshold_independent_metrics": {"roc_auc": roc_auc, "eer": eer}
    }
    with (RESULTS_DIR / "cnn_final_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # Save confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bonafide", "Spoof"], yticklabels=["Bonafide", "Spoof"]) 
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix (DEV) - Best Checkpoint")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_final_confusion_matrix.png")

    # Final safety summary
    print("\n========== DATASET SAFETY CHECK ==========")
    print("[PASS] Train protocol loaded")
    print("[PASS] Dev protocol loaded")
    print("[PASS] No train/dev overlap")
    print("[PASS] Both classes present in train")
    print("[PASS] Both classes present in dev")
    print("[PASS] Missing audio check passed")
    print("[PASS] Eval isolated from training")
    print("[PASS] Train-only normalization")
    print("[PASS] Batch-level device transfer")
    print("===========================================")

    # Print final evaluation block
    print("\n========== FINAL EVALUATION ==========")
    print(f"Accuracy : {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"F1       : {f1}")
    print(f"ROC-AUC  : {roc_auc}")
    print(f"EER      : {eer}")
    print("=======================================")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Loss: {best_val_loss}")

    print(f"Best model path: {best_path}")
    print(f"Results JSON: {RESULTS_DIR / 'cnn_final_results.json'}")
    print(f"Norm JSON: {norm_file}")
    print(f"Confusion matrix image: {RESULTS_DIR / 'cnn_final_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
