"""Segment-level deepfake detection using the existing AASIST model.

Splits an audio file into fixed-size windows and runs the trained AASIST model
on each window.  The model weights, architecture, and threshold are **never
modified** — this module purely wraps the existing inference pipeline.

Typical usage
-------------
>>> from pathlib import Path
>>> from src.forensics.segment import run_segment_analysis
>>> result = run_segment_analysis(Path("audio.wav"))
>>> print(result["overall_verdict"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Windows/server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from ..audio_preprocessing import preprocess_audio
from ..inference.predict import load_aasist_model

# ── Constants ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AASIST_THRESHOLD_PATH = PROJECT_ROOT / "results" / "aasist" / "aasist_balanced_threshold.json"
SEGMENT_PLOT_PATH = PROJECT_ROOT / "results" / "signal_plots" / "segment_timeline.png"

_DEFAULT_WINDOW  = 1.0   # seconds
_DEFAULT_HOP     = 1.0   # seconds
_DEFAULT_SAMPLES = 64_000  # 4 s × 16 000 Hz — the model's native input length
_DEFAULT_SR      = 16_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Internal helpers ───────────────────────────────────────────────────────

def _load_threshold() -> float:
    """Return the EER-calibrated threshold from disk, defaulting to 0.819."""
    if AASIST_THRESHOLD_PATH.is_file():
        try:
            return float(json.loads(AASIST_THRESHOLD_PATH.read_text())
                         .get("threshold", 0.819))
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    return 0.819


def _score_segment(model: torch.nn.Module, audio_segment: np.ndarray) -> float:
    """Run one AASIST forward pass and return P(spoof) in [0, 1].

    The segment is padded or truncated to exactly ``_DEFAULT_SAMPLES`` samples,
    matching the model's expected input length, before inference.
    """
    # Pad / truncate to the model's native 4-second window.
    seg = audio_segment.astype(np.float32)
    if len(seg) < _DEFAULT_SAMPLES:
        seg = np.pad(seg, (0, _DEFAULT_SAMPLES - len(seg)), mode="constant")
    else:
        seg = seg[:_DEFAULT_SAMPLES]

    tensor = torch.from_numpy(seg).float().unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logit = model(tensor).reshape(-1)[0]
        prob  = float(torch.sigmoid(logit).item())
    return prob


# ── Public API ─────────────────────────────────────────────────────────────

def run_segment_analysis(
    audio_path: Path,
    window_seconds: float = _DEFAULT_WINDOW,
    hop_seconds: float    = _DEFAULT_HOP,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run segment-level deepfake detection using the existing AASIST model.

    The audio is split into fixed-size windows.  Each window is scored by the
    trained AASIST graph-attention network.  No model weights are modified.

    Parameters
    ----------
    audio_path:
        Path to the audio file (any format supported by ``preprocess_audio``).
    window_seconds:
        Duration of each analysis window in seconds (default 1.0).
    hop_seconds:
        Step size between consecutive windows in seconds (default 1.0, i.e.
        non-overlapping).
    threshold:
        Spoof-probability decision threshold.  Defaults to the EER-calibrated
        value stored in ``results/aasist/aasist_balanced_threshold.json``
        (≈ 0.819).

    Returns
    -------
    dict
        ``total_duration``, ``window_seconds``, ``hop_seconds``, ``threshold``,
        ``segments`` (full list), ``suspicious_segments``, ``overall_verdict``,
        ``visualization_path``.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if threshold is None:
        threshold = _load_threshold()

    # ── Load + resample audio ─────────────────────────────────────────────
    # Use a long duration so we capture the entire file; preprocess_audio
    # pads/truncates to exactly `duration` seconds, so we peek at the real
    # length first via soundfile / FFmpeg without resampling overhead.
    try:
        import soundfile as sf
        info = sf.info(str(audio_path))
        total_duration = float(info.duration)
    except Exception:
        # Fallback: load at 5 s to get the sample rate, then compute.
        _probe, _sr = preprocess_audio(audio_path, duration=5.0)
        total_duration = 5.0

    if total_duration <= 0:
        total_duration = 5.0

    # Load full audio at target sample rate.
    raw_audio, sr = preprocess_audio(audio_path, duration=total_duration)
    actual_duration = len(raw_audio) / sr

    window_samples = int(window_seconds * sr)
    hop_samples    = int(hop_seconds   * sr)

    if window_samples <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
    if hop_samples <= 0:
        raise ValueError(f"hop_seconds must be > 0, got {hop_seconds}")

    # ── Load model once ───────────────────────────────────────────────────
    model = load_aasist_model()
    model.eval()

    # ── Segment loop ──────────────────────────────────────────────────────
    segments: list[dict[str, Any]] = []
    start_sample = 0

    while start_sample < len(raw_audio):
        end_sample = start_sample + window_samples
        chunk      = raw_audio[start_sample:end_sample]

        # Skip fully silent / empty tails that add no information.
        if len(chunk) == 0:
            break

        start_t = start_sample / sr
        end_t   = min(end_sample, len(raw_audio)) / sr

        prob    = _score_segment(model, chunk)
        verdict = "spoof" if prob > threshold else "bonafide"

        segments.append({
            "start":            round(start_t, 3),
            "end":              round(end_t,   3),
            "spoof_probability": round(prob,   4),
            "verdict":          verdict,
        })

        start_sample += hop_samples

    # ── Aggregate result ──────────────────────────────────────────────────
    if not segments:
        overall_verdict = "unknown"
        suspicious      = []
    else:
        spoof_count  = sum(1 for s in segments if s["verdict"] == "spoof")
        overall_verdict = "spoof" if spoof_count > len(segments) / 2 else "bonafide"
        suspicious = [
            {"start": s["start"], "end": s["end"], "probability": s["spoof_probability"]}
            for s in segments if s["verdict"] == "spoof"
        ]

    # ── Visualization ─────────────────────────────────────────────────────
    viz_path = _generate_timeline(
        segments         = segments,
        threshold        = threshold,
        total_duration   = actual_duration,
        output_path      = SEGMENT_PLOT_PATH,
    )

    return {
        "total_duration":    round(actual_duration, 3),
        "window_seconds":    window_seconds,
        "hop_seconds":       hop_seconds,
        "threshold":         threshold,
        "segments":          segments,
        "overall_verdict":   overall_verdict,
        "suspicious_segments": suspicious,
        "visualization_path":  str(viz_path) if viz_path else None,
    }


# ── Visualization ──────────────────────────────────────────────────────────

def _generate_timeline(
    segments:       list[dict[str, Any]],
    threshold:      float,
    total_duration: float,
    output_path:    Path,
) -> Path | None:
    """Save a bar-chart timeline of spoof probability over time.

    Parameters
    ----------
    segments:
        List of segment dicts as returned by ``run_segment_analysis``.
    threshold:
        Decision threshold line drawn on the plot.
    total_duration:
        Total audio duration for x-axis range.
    output_path:
        File path where the PNG will be saved.

    Returns
    -------
    Path or None
        The saved file path, or ``None`` if saving failed.
    """
    if not segments:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    starts = [s["start"] for s in segments]
    probs  = [s["spoof_probability"] for s in segments]
    colors = ["#EF4444" if p > threshold else "#10B981" for p in probs]
    width  = max(s["end"] - s["start"] for s in segments) * 0.85

    fig, ax = plt.subplots(figsize=(max(8, total_duration * 0.9), 4))
    ax.bar(starts, probs, width=width, color=colors, align="edge",
           edgecolor="white", linewidth=0.5, alpha=0.9)

    # Threshold line
    ax.axhline(y=threshold, color="#F59E0B", linewidth=1.8,
               linestyle="--", label=f"Threshold ({threshold:.3f})")

    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Spoof Probability", fontsize=10)
    ax.set_title("Segment-Level Deepfake Detection", fontsize=12, fontweight="bold")

    spoof_patch    = mpatches.Patch(color="#EF4444", label="SPOOF segment")
    bonafide_patch = mpatches.Patch(color="#10B981", label="BONAFIDE segment")
    ax.legend(handles=[spoof_patch, bonafide_patch,
                        plt.Line2D([0], [0], color="#F59E0B",
                                   linewidth=1.8, linestyle="--",
                                   label=f"Threshold ({threshold:.3f})")],
              loc="upper right", fontsize=8)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    try:
        fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
        return output_path
    except OSError:
        return None
    finally:
        plt.close(fig)
