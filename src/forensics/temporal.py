"""Temporal consistency analysis for speech signals."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from .signal import _prepare_audio


def extract_pitch_contour(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract an F0 contour in Hz, retaining NaN for unvoiced frames."""
    values = _prepare_audio(audio, sr)
    frame_length = min(2048, max(32, values.size))
    if frame_length % 2:
        frame_length += 1
    fmin = max(50.0, librosa.note_to_hz("C2"))
    fmax = min(sr / 2.0, librosa.note_to_hz("C7"))
    pitch, _, _ = librosa.pyin(
        values,
        fmin=fmin,
        fmax=max(fmin + 1.0, fmax),
        sr=sr,
        frame_length=frame_length,
        hop_length=max(1, frame_length // 4),
    )
    return np.asarray(pitch, dtype=np.float32)


def analyze_pitch_stability(pitch_contour: np.ndarray) -> dict[str, Any]:
    """Measure voiced pitch continuity and identify unusually large jumps."""
    pitch = np.asarray(pitch_contour, dtype=np.float64)
    valid = pitch[np.isfinite(pitch) & (pitch > 0)]
    if valid.size < 2:
        return {"stability_score": 0.0, "anomalies": ["Insufficient voiced pitch for stability analysis."], "description": "Pitch stability cannot be determined reliably."}
    jumps = np.abs(np.diff(valid))
    threshold = max(80.0, float(np.median(valid) * 0.75))
    jump_count = int(np.count_nonzero(jumps > threshold))
    stability = max(0.0, 1.0 - jump_count / max(1, jumps.size))
    anomalies = [f"{jump_count} unusually large pitch jump(s) detected."] if jump_count else []
    return {
        "stability_score": float(stability),
        "anomalies": anomalies,
        "description": "Pitch contour is broadly continuous." if not anomalies else "Pitch contour contains abrupt changes; requires further investigation.",
    }


def extract_energy_profile(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Extract RMS energy over time and flag extreme frame-to-frame changes."""
    values = _prepare_audio(audio, sr)
    frame_length = min(2048, max(32, values.size))
    hop_length = max(1, frame_length // 4)
    energy = librosa.feature.rms(y=values, frame_length=frame_length, hop_length=hop_length)[0]
    differences = np.abs(np.diff(energy))
    threshold = max(0.1, float(np.mean(energy) + 3 * np.std(energy)))
    count = int(np.count_nonzero(differences > threshold))
    return {
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "anomalies": [] if count == 0 else [f"{count} abrupt energy transition(s) detected."],
    }


def analyze_temporal_consistency(audio: np.ndarray, sr: int) -> dict[str, Any]:
    """Combine pitch and energy temporal indicators."""
    pitch_analysis = analyze_pitch_stability(extract_pitch_contour(audio, sr))
    energy_analysis = extract_energy_profile(audio, sr)
    anomalies = pitch_analysis["anomalies"] + energy_analysis["anomalies"]
    consistency = max(0.0, pitch_analysis["stability_score"] - 0.1 * len(energy_analysis["anomalies"]))
    return {
        "consistency_score": float(consistency),
        "anomalies": anomalies,
        "summary": "Temporal patterns are broadly consistent." if not anomalies else "Temporal indicators require further investigation.",
        "pitch_analysis": pitch_analysis,
        "energy_analysis": energy_analysis,
    }

