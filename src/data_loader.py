"""Dataset loading utilities for ASVspoof 2019 LA experiments.

This module reads the official protocol files, loads audio examples, applies the
project preprocessing step, and extracts the configured feature type for later
model training pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from audio_preprocessing import preprocess_audio
from feature_extraction import (
    extract_mfcc,
    extract_mfcc_with_deltas,
    extract_mel_spectrogram,
)


LABEL_MAP = {"bonafide": 0, "spoof": 1}


def load_protocol(protocol_path: Path) -> pd.DataFrame:
    """Load an ASVspoof CM protocol file into a DataFrame.

    The ASVspoof 2019 LA protocol is observed in two common forms:
    - 4 columns: speaker_id utterance_id attack_type label
    - 5 columns: speaker_id utterance_id - attack_type label

    The returned DataFrame always uses the same column names to simplify dataset
    loading.
    """
    protocol_path = Path(protocol_path)

    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    records: list[dict[str, str]] = []

    with protocol_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            columns = line.split()
            if len(columns) == 4:
                speaker_id, utterance_id, attack_type, label = columns
            elif len(columns) == 5:
                speaker_id, utterance_id, _, attack_type, label = columns
            else:
                raise ValueError(
                    f"Unsupported protocol format on line {line_number} of {protocol_path}: "
                    f"expected 4 or 5 columns, got {len(columns)} => '{line}'"
                )

            if label not in LABEL_MAP:
                raise ValueError(
                    f"Unexpected label '{label}' on line {line_number}: {protocol_path}"
                )

            records.append(
                {
                    "speaker_id": speaker_id,
                    "utterance_id": utterance_id,
                    "file_id": utterance_id,
                    "attack_type": attack_type,
                    "label": label,
                }
            )

    dataframe = pd.DataFrame(
        records,
        columns=["speaker_id", "utterance_id", "file_id", "attack_type", "label"],
    )
    return dataframe


def _extract_feature_vector(audio: np.ndarray, sr: int, feature_type: str) -> np.ndarray:
    """Return the feature vector for the selected feature type."""
    feature_type = feature_type.lower()

    if feature_type == "mfcc":
        return extract_mfcc(audio, sr)
    if feature_type == "mfcc_delta":
        return extract_mfcc_with_deltas(audio, sr)
    if feature_type == "mel_spectrogram":
        return extract_mel_spectrogram(audio, sr)

    raise ValueError(f"Unsupported feature_type '{feature_type}'. Use 'mfcc', 'mfcc_delta', or 'mel_spectrogram'.")


def load_dataset(
    dataset_root: Path,
    protocol_type: str,
    feature_type: str,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess one ASVspoof split into feature vectors and labels.

    Parameters
    ----------
    dataset_root:
        Root folder that contains the LA dataset hierarchy.
    protocol_type:
        One of: 'train', 'dev', or 'eval'.
    feature_type:
        One of: 'mfcc', 'mfcc_delta', or 'mel_spectrogram'.
    max_samples:
        Optional cap for the number of samples to load.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        Feature matrix, integer labels, and file IDs.
    """
    dataset_root = Path(dataset_root)
    valid_types = {"train", "dev", "eval"}
    if protocol_type not in valid_types:
        raise ValueError(f"protocol_type must be one of {sorted(valid_types)}, received '{protocol_type}'.")

    protocol_path = (
        dataset_root
        / "ASVspoof2019_LA_cm_protocols"
        / {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
        }[protocol_type]
    )
    audio_dir = dataset_root / f"ASVspoof2019_LA_{protocol_type}" / "flac"

    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file missing: {protocol_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory missing: {audio_dir}")

    protocol_df = load_protocol(protocol_path)
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"max_samples must be positive or None; received {max_samples}.")

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    file_ids: list[str] = []

    total_rows = len(protocol_df)
    processed_count = 0

    for index, row in protocol_df.iterrows():
        if max_samples is not None and processed_count >= max_samples:
            break

        utterance_id = str(row["utterance_id"])
        audio_path = audio_dir / f"{utterance_id}.flac"

        if not audio_path.exists():
            print(f"[WARNING] Missing audio file for {protocol_type}: {audio_path}")
            continue

        try:
            audio, sample_rate = preprocess_audio(audio_path)
            feature_vector = _extract_feature_vector(audio, sample_rate, feature_type)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"[WARNING] Could not load audio for {protocol_type}: {audio_path}")
            print(f"          Reason: {exc}")
            continue

        feature_rows.append(np.asarray(feature_vector, dtype=np.float32))
        labels.append(LABEL_MAP[str(row["label"])])
        file_ids.append(utterance_id)
        processed_count += 1

        if processed_count % 1000 == 0:
            print(f"[{protocol_type}] Processed {processed_count}/{total_rows} files...")

    if not feature_rows:
        empty_array = np.empty((0,), dtype=np.float32)
        return empty_array, np.asarray([], dtype=np.int64), []

    feature_matrix = np.vstack(feature_rows)
    label_array = np.asarray(labels, dtype=np.int64)
    return feature_matrix, label_array, file_ids


def prepare_dataset_svm(
    dataset_root: Path,
    use_delta: bool = True,
    test_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Prepare the train and dev sets for the SVM baseline pipeline.

    The features are standardized using the training-set scaler before both
    train and dev data are returned.
    """
    dataset_root = Path(dataset_root)
    feature_type = "mfcc_delta" if use_delta else "mfcc"

    train_max = 1000 if test_mode else None
    dev_max = 1000 if test_mode else None

    print(f"[INFO] Loading training set for SVM baseline using feature type: {feature_type}")
    X_train, y_train, _ = load_dataset(dataset_root, "train", feature_type, max_samples=train_max)

    print(f"[INFO] Loading development set for SVM baseline using feature type: {feature_type}")
    X_dev, y_dev, _ = load_dataset(dataset_root, "dev", feature_type, max_samples=dev_max)

    if X_train.size == 0 or X_dev.size == 0:
        raise ValueError("Training or development dataset is empty after preprocessing. Check dataset paths and protocol files.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    return X_train_scaled, y_train, X_dev_scaled, y_dev, scaler
