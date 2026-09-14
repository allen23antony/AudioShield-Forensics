"""Professional PNG visualizations for signal forensics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np


def _save(figure: plt.Figure, output_path: Path) -> None:
    """Save and close a figure."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_waveform(audio: np.ndarray, sr: int, output_path: Path) -> None:
    """Plot and save the waveform."""
    figure, axis = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, ax=axis, color="#1f4e79")
    axis.set(title="Audio Waveform", xlabel="Time (s)", ylabel="Amplitude")
    _save(figure, output_path)


def plot_spectrogram(audio: np.ndarray, sr: int, output_path: Path) -> None:
    """Plot and save a mel-spectrogram."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=96)
    figure, axis = plt.subplots(figsize=(10, 4))
    image = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, x_axis="time", y_axis="mel", ax=axis, cmap="magma")
    figure.colorbar(image, ax=axis, format="%+2.0f dB")
    axis.set(title="Mel-Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    _save(figure, output_path)


def plot_mfcc(audio: np.ndarray, sr: int, output_path: Path) -> None:
    """Plot and save an MFCC heatmap."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    figure, axis = plt.subplots(figsize=(10, 4))
    image = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=axis, cmap="viridis")
    figure.colorbar(image, ax=axis)
    axis.set(title="MFCC Features", xlabel="Time (s)", ylabel="Coefficient")
    _save(figure, output_path)


def plot_pitch_contour(pitch: np.ndarray, output_path: Path) -> None:
    """Plot and save a fundamental-frequency contour."""
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(np.arange(pitch.size), pitch, color="#8b1e3f", linewidth=1.2)
    axis.set(title="Pitch Contour", xlabel="Frame", ylabel="Frequency (Hz)")
    axis.grid(alpha=0.25)
    _save(figure, output_path)


def generate_forensic_plots(audio: np.ndarray, sr: int, output_dir: Path) -> list[Path]:
    """Generate all signal forensic plots and return their paths."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pitch = librosa.pyin(
        audio,
        fmin=max(50.0, librosa.note_to_hz("C2")),
        fmax=min(sr / 2.0, librosa.note_to_hz("C7")),
        sr=sr,
    )[0]
    paths = [
        directory / "waveform.png",
        directory / "spectrogram.png",
        directory / "mfcc.png",
        directory / "pitch_contour.png",
    ]
    plot_waveform(audio, sr, paths[0])
    plot_spectrogram(audio, sr, paths[1])
    plot_mfcc(audio, sr, paths[2])
    plot_pitch_contour(pitch, paths[3])
    return paths

