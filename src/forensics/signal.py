"""Signal-level feature extraction and anomaly detection."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np


def _prepare_audio(audio: np.ndarray, sr: int, minimum_seconds: float = 0.1) -> np.ndarray:
    """Validate mono audio and pad short inputs for stable librosa features."""
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive; received {sr}.")
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Audio must be one-dimensional; received shape {values.shape}.")
    if values.size == 0:
        raise ValueError("Audio cannot be empty.")
    minimum_samples = max(32, int(round(sr * minimum_seconds)))
    if values.size < minimum_samples:
        values = np.pad(values, (0, minimum_samples - values.size))
    return values


def _describe(values: np.ndarray) -> dict[str, float]:
    """Return robust descriptive statistics for a feature sequence."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def extract_spectral_features(audio: np.ndarray, sr: int) -> dict[str, dict[str, float]]:
    """Extract summary statistics for common spectral descriptors."""
    values = _prepare_audio(audio, sr)
    n_fft = min(2048, max(32, values.size))
    hop_length = max(1, n_fft // 4)
    features: dict[str, np.ndarray] = {
        "spectral_centroid": librosa.feature.spectral_centroid(y=values, sr=sr, n_fft=n_fft, hop_length=hop_length)[0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=values, sr=sr, n_fft=n_fft, hop_length=hop_length)[0],
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=values, sr=sr, n_fft=n_fft, hop_length=hop_length)[0],
        "spectral_flatness": librosa.feature.spectral_flatness(y=values, n_fft=n_fft, hop_length=hop_length)[0],
        "spectral_contrast": librosa.feature.spectral_contrast(
            y=values,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_bands=4,
        ).mean(axis=0),
    }
    return {name: _describe(sequence) for name, sequence in features.items()}


def extract_temporal_features(audio: np.ndarray, sr: int) -> dict[str, dict[str, float]]:
    """Extract zero-crossing, RMS, and fundamental-frequency statistics."""
    values = _prepare_audio(audio, sr)
    n_fft = min(2048, max(32, values.size))
    hop_length = max(1, n_fft // 4)
    fmin = max(50.0, librosa.note_to_hz("C2"))
    fmax = min(sr / 2.0, librosa.note_to_hz("C7"))
    pitch, _, _ = librosa.pyin(
        values,
        fmin=fmin,
        fmax=max(fmin + 1.0, fmax),
        sr=sr,
        frame_length=n_fft if n_fft % 2 == 0 else n_fft + 1,
        hop_length=hop_length,
    )
    return {
        "zero_crossing_rate": _describe(librosa.feature.zero_crossing_rate(values, hop_length=hop_length)[0]),
        "rms_energy": _describe(librosa.feature.rms(y=values, frame_length=n_fft, hop_length=hop_length)[0]),
        "pitch": _describe(pitch),
    }


def extract_signal_features(audio: np.ndarray, sr: int) -> dict[str, dict[str, dict[str, float]]]:
    """Combine spectral and temporal signal descriptors."""
    return {
        "spectral": extract_spectral_features(audio, sr),
        "temporal": extract_temporal_features(audio, sr),
    }


def detect_signal_anomalies(features: dict[str, Any]) -> dict[str, Any]:
    """Compare feature means with broad speech ranges and report indicators."""
    expected_ranges: dict[str, tuple[float, float]] = {
        "spectral_centroid": (300.0, 5000.0),
        "spectral_bandwidth": (300.0, 5000.0),
        "spectral_rolloff": (500.0, 7000.0),
        "spectral_flatness": (0.001, 0.6),
        "spectral_contrast": (5.0, 50.0),
        "zero_crossing_rate": (0.005, 0.35),
        "rms_energy": (1e-5, 1.0),
        "pitch": (60.0, 500.0),
    }
    anomalies: list[dict[str, Any]] = []
    for group in ("spectral", "temporal"):
        for name, summary in features.get(group, {}).items():
            if name not in expected_ranges:
                continue
            value = float(summary.get("mean", np.nan))
            lower, upper = expected_ranges[name]
            is_anomaly = not np.isfinite(value) or value < lower or value > upper
            if is_anomaly:
                severity = "high" if not np.isfinite(value) else "medium"
                anomalies.append(
                    {
                        "feature_name": name,
                        "value": None if not np.isfinite(value) else value,
                        "expected_range": [lower, upper],
                        "is_anomaly": True,
                        "severity": severity,
                    }
                )
    highest = {"low": 0, "medium": 1, "high": 2}
    risk_score = max((item["severity"] for item in anomalies), key=lambda level: highest[level], default="low")
    summary = (
        "No broad signal-level anomalies identified."
        if not anomalies
        else f"{len(anomalies)} signal-level indicator(s) fall outside broad speech ranges; requires further investigation."
    )
    return {"anomalies": anomalies, "risk_score": risk_score, "summary": summary}

