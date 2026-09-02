"""Feature extraction utilities for ASVspoof 2019 LA audio.

This module exposes small, reusable feature extractors that produce compact
representations for SVM baseline experiments.
"""

from __future__ import annotations

import librosa
import numpy as np


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Extract MFCC features and return the mean across time.

    Parameters
    ----------
    audio:
        One-dimensional audio waveform.
    sr:
        Sample rate in Hz.
    n_mfcc:
        Number of MFCC coefficients to retain.

    Returns
    -------
    numpy.ndarray
        A one-dimensional vector with length ``n_mfcc``.
    """
    if audio.ndim != 1:
        raise ValueError(f"MFCC extraction expects a 1D audio array; received shape {audio.shape}.")
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive; received {sr}.")
    if n_mfcc <= 0:
        raise ValueError(f"n_mfcc must be positive; received {n_mfcc}.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
    )
    return np.asarray(mfcc.mean(axis=1), dtype=np.float32)


def extract_mfcc_with_deltas(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCC, delta, and delta-delta coefficients and average over time.

    The final feature vector concatenates the 13 MFCC coefficients, the first
    delta coefficients, and the second delta coefficients. The result is a
    39-dimensional representation.
    """
    if audio.ndim != 1:
        raise ValueError(f"MFCC delta extraction expects a 1D audio array; received shape {audio.shape}.")
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive; received {sr}.")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
    )
    delta_1 = librosa.feature.delta(mfcc, order=1)
    delta_2 = librosa.feature.delta(mfcc, order=2)

    stacked = np.concatenate([mfcc, delta_1, delta_2], axis=0)
    return np.asarray(stacked.mean(axis=1), dtype=np.float32)


def extract_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract a mel-spectrogram in dB scale.

    The output is a 2D array of shape ``(n_mels, time_frames)``.
    """
    if audio.ndim != 1:
        raise ValueError(f"Mel spectrogram extraction expects a 1D audio array; received shape {audio.shape}.")
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive; received {sr}.")

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
    )
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return np.asarray(mel_db, dtype=np.float32)
