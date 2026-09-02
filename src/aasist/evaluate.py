from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aasist.config import (
    ATTACK_IDS,
    DEV_PROTOCOL,
    EVAL_AUDIO_DIR,
    EVAL_PROTOCOL,
    EXPERIMENT_CONFIG,
    MODEL_DIR,
    RESULTS_DIR,
    SAMPLE_RATE,
    SEED,
)
from aasist.dataset import ASVspoofDataset, validate_dataset_integrity
from aasist.model import AASISTModel
from aasist.utils import calculate_eer, compute_threshold_metrics, save_confusion_matrix, save_json, stable_sigmoid


def set_seed(seed: int = SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_threshold(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Threshold file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("selection_source") != "dev_only":
        raise ValueError("Threshold selection source must be dev_only; official Eval must not be used for threshold tuning.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen AASIST model on the official ASVspoof 2019 LA eval set.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint", type=str, default=str(MODEL_DIR / "aasist_best.pth"))
    parser.add_argument("--threshold-file", type=str, default=str(RESULTS_DIR / "aasist_threshold.json"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("========== AASIST OFFICIAL EVALUATION ==========")
    print("TRAIN -> DEV threshold selection -> FROZEN -> OFFICIAL EVAL")
    print("Device:", device)
    print("Seed:", args.seed)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Sampling rate:", SAMPLE_RATE)

    validate_dataset_integrity()
    threshold_payload = load_threshold(Path(args.threshold_file))
    threshold = float(threshold_payload["threshold"])
    print("Frozen DEV threshold:", threshold)

    dataset = ASVspoofDataset("eval", max_samples=128 if args.smoke_test else None, sample_rate=SAMPLE_RATE, target_samples=int(SAMPLE_RATE * EXPERIMENT_CONFIG.input_duration_seconds))
    loader = DataLoader(dataset, batch_size=32 if not args.smoke_test else 8, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = AASISTModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    utterance_ids: list[str] = []
    labels: list[int] = []
    attack_ids: list[str] = []
    scores: list[float] = []

    with torch.inference_mode():
        for waveforms, label_tensor, utt_ids, attack_id_list in loader:
            waveforms = waveforms.to(device)
            logits = model(waveforms.squeeze(1))
            probs = stable_sigmoid(logits.detach().cpu().numpy())
            for i, utt in enumerate(utt_ids):
                utterance_ids.append(utt)
                labels.append(int(label_tensor[i].item()))
                attack_ids.append(str(attack_id_list[i]))
                scores.append(float(probs[i]))

    utterance_ids_arr = np.asarray(utterance_ids, dtype=str)
    labels_arr = np.asarray(labels, dtype=np.int64)
    attack_ids_arr = np.asarray(attack_ids, dtype=str)
    scores_arr = np.asarray(scores, dtype=np.float32)

    if len(utterance_ids_arr) != 71237 and not args.smoke_test:
        raise ValueError(f"Unexpected eval cache size: {len(utterance_ids_arr)}; expected 71237.")

    cache_path = RESULTS_DIR / "aasist_eval_scores.npz"
    np.savez(cache_path, utterance_ids=utterance_ids_arr, labels=labels_arr, attack_ids=attack_ids_arr, scores=scores_arr)

    y_true = labels_arr
    y_scores = scores_arr
    y_pred = (y_scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "total_samples": int(len(y_true)),
        "bonafide_samples": int((y_true == 0).sum()),
        "spoof_samples": int((y_true == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float((tn / max((y_true == 0).sum(), 1))),
        "fpr": float((fp / max((y_true == 0).sum(), 1))),
        "tpr": float((tp / max((y_true == 1).sum(), 1))),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "eer": float(calculate_eer(y_true, y_scores)),
        "threshold": float(threshold),
    }
    save_json(RESULTS_DIR / "aasist_eval_results.json", metrics)
    save_confusion_matrix(y_true, y_pred, RESULTS_DIR / "aasist_confusion_matrix.png")

    print("Official evaluation summary:")
    print(json.dumps(metrics, indent=2))
    print(f"Cache saved to: {cache_path}")
    print("Eval used for threshold selection: NO")

    if args.smoke_test:
        print("SMOKE TEST PASS: evaluation loop and cache generation succeeded.")


if __name__ == "__main__":
    main()
