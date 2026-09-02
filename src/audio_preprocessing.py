"""Reusable preprocessing utilities for ASVspoof 2019 LA audio.

This module normalizes input audio to a fixed 16 kHz / 5 second format so that
later experiments can use consistent temporal and sampling settings.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def preprocess_audio(
    audio_path: str | Path,
    target_sr: int = 16000,
    duration: float = 5.0,
) -> tuple[np.ndarray, int]:
    """Load, resample, and normalize an audio file to a fixed length.

    Parameters
    ----------
    audio_path:
        Path to an audio file.
    target_sr:
        Target sample rate in Hz. Default is 16,000 Hz to match the baseline
        preprocessing used in the EchoShield-style comparison workflow.
    duration:
        Desired audio duration in seconds. The audio is truncated or zero-padded
        to match this length exactly.

    Returns
    -------
    tuple[numpy.ndarray, int]
        A mono audio array and the final sample rate.

    Raises
    ------
    FileNotFoundError
        If the input audio file does not exist.
    ValueError
        If the file cannot be read or converted to usable audio data.
    """
    input_path = Path(audio_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    try:
        audio, original_sr = sf.read(input_path, dtype="float32", always_2d=False)
    except Exception as exc:  # pragma: no cover - defensive branch for corrupted files
        raise ValueError(f"Unable to read audio file '{input_path}': {exc}") from exc

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {input_path}")

    audio_array = np.asarray(audio, dtype=np.float32)

    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    if original_sr != target_sr:
        audio_array = librosa.resample(
            y=audio_array,
            orig_sr=original_sr,
            target_sr=target_sr,
            res_type="soxr_hq",
        )

    target_samples = int(round(duration * target_sr))
    processed_audio = librosa.util.fix_length(audio_array, size=target_samples)

    return processed_audio.astype(np.float32), target_sr
