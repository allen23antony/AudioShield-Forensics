from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import (
    ATTACK_IDS,
    DATA_ROOT,
    DEV_AUDIO_DIR,
    DEV_PROTOCOL,
    EVAL_AUDIO_DIR,
    EVAL_PROTOCOL,
    EXPECTED_SPLIT_COUNTS,
    LABEL_MAP,
    SAMPLE_RATE,
    TRAIN_AUDIO_DIR,
    TRAIN_PROTOCOL,
)


def parse_protocol_records(protocol_path: Path) -> list[dict[str, object]]:
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    rows: list[dict[str, object]] = []
    with protocol_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 4:
                speaker_id, utterance_id, attack_id, label = parts
            elif len(parts) == 5:
                speaker_id, utterance_id, _, attack_id, label = parts
            else:
                raise ValueError(f"Malformed protocol row at line {line_number}: {line}")

            if label not in LABEL_MAP:
                raise ValueError(f"Unknown label '{label}' at line {line_number}: {protocol_path}")

            rows.append(
                {
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "attack_id": None if attack_id == "-" else attack_id,
                    "label": label,
                    "label_index": LABEL_MAP[label],
                }
            )
    return rows


def verify_split_counts(split_name: str, protocol_path: Path, audio_dir: Path) -> dict[str, int]:
    rows = parse_protocol_records(protocol_path)
    counts = {"total": len(rows), "bonafide": 0, "spoof": 0}
    for row in rows:
        label = str(row["label"])
        if label == "bonafide":
            counts["bonafide"] += 1
        elif label == "spoof":
            counts["spoof"] += 1

    expected = EXPECTED_SPLIT_COUNTS[split_name]
    if counts["total"] != expected["total"] or counts["bonafide"] != expected["bonafide"] or counts["spoof"] != expected["spoof"]:
        raise ValueError(f"Split count mismatch for {split_name}: got {counts}, expected {expected}")

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    for row in rows:
        utt_id = str(row["utterance_id"])
        path = audio_dir / f"{utt_id}.flac"
        if not path.exists():
            raise FileNotFoundError(f"Missing audio file for {split_name}: {path}")

    return counts


def validate_dataset_integrity() -> dict[str, object]:
    summary: dict[str, object] = {}
    split_specs = {
        "train": (TRAIN_PROTOCOL, TRAIN_AUDIO_DIR),
        "dev": (DEV_PROTOCOL, DEV_AUDIO_DIR),
        "eval": (EVAL_PROTOCOL, EVAL_AUDIO_DIR),
    }

    train_rows = parse_protocol_records(TRAIN_PROTOCOL)
    dev_rows = parse_protocol_records(DEV_PROTOCOL)
    eval_rows = parse_protocol_records(EVAL_PROTOCOL)

    train_ids = {str(r["utterance_id"]) for r in train_rows}
    dev_ids = {str(r["utterance_id"]) for r in dev_rows}
    eval_ids = {str(r["utterance_id"]) for r in eval_rows}

    if train_ids & dev_ids:
        raise ValueError("Train/dev overlap detected; leakage is not allowed.")
    if train_ids & eval_ids:
        raise ValueError("Train/eval overlap detected; leakage is not allowed.")
    if dev_ids & eval_ids:
        raise ValueError("Dev/eval overlap detected; leakage is not allowed.")

    for split_name, (protocol_path, audio_dir) in split_specs.items():
        split_summary = verify_split_counts(split_name, protocol_path, audio_dir)
        summary[split_name] = split_summary

    for split_name, rows in {"train": train_rows, "dev": dev_rows, "eval": eval_rows}.items():
        ids = [str(r["utterance_id"]) for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate utterance IDs detected in {split_name} split.")

    return summary


def load_waveform(path: Path, sample_rate: int = SAMPLE_RATE, target_samples: int | None = None, normalize: bool = True) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.size == 0:
        raise ValueError(f"Audio is empty: {path}")
    wav = np.asarray(audio, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != sample_rate:
        # Use a deterministic library-based resampler; librosa is already available in the project.
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate, res_type="soxr_hq")
    if target_samples is None:
        target_samples = int(sample_rate * 4.0)
    if len(wav) < target_samples:
        pad_len = target_samples - len(wav)
        wav = np.pad(wav, (0, pad_len), mode="constant", constant_values=0.0)
    else:
        wav = wav[:target_samples]
    if normalize:
        peak = np.max(np.abs(wav))
        if peak > 1e-12:
            wav = wav / peak
    wav = np.clip(wav, -1.0, 1.0)
    return wav.astype(np.float32)


class ASVspoofDataset(Dataset):
    def __init__(self, split: str, max_samples: int | None = None, sample_rate: int = SAMPLE_RATE, target_samples: int | None = None, deterministic: bool = True):
        if split not in {"train", "dev", "eval"}:
            raise ValueError(f"Unsupported split '{split}'")
        split_root = {"train": TRAIN_AUDIO_DIR, "dev": DEV_AUDIO_DIR, "eval": EVAL_AUDIO_DIR}[split]
        protocol_path = {"train": TRAIN_PROTOCOL, "dev": DEV_PROTOCOL, "eval": EVAL_PROTOCOL}[split]

        rows = parse_protocol_records(protocol_path)
        if max_samples is not None:
            rows = rows[:max_samples]

        self.rows = rows
        self.audio_dir = split_root
        self.sample_rate = sample_rate
        self.target_samples = target_samples or int(sample_rate * 4.0)
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str | None]:
        row = self.rows[idx]
        utterance_id = str(row["utterance_id"])
        attack_id = str(row["attack_id"]) if row["attack_id"] is not None else "-"
        label_index = int(row["label_index"])
        wav_path = self.audio_dir / f"{utterance_id}.flac"
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing audio file: {wav_path}")

        waveform = load_waveform(wav_path, sample_rate=self.sample_rate, target_samples=self.target_samples, normalize=True)
        tensor = torch.from_numpy(waveform).float()
        tensor = tensor.unsqueeze(0)
        label = torch.tensor(label_index, dtype=torch.float32)
        return tensor, label, utterance_id, attack_id


def build_split_dataset(split: str, max_samples: int | None = None) -> ASVspoofDataset:
    return ASVspoofDataset(split=split, max_samples=max_samples, sample_rate=SAMPLE_RATE, target_samples=int(SAMPLE_RATE * 4.0), deterministic=True)
