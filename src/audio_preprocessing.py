"""Reusable preprocessing utilities for ASVspoof 2019 LA audio.

This module normalizes input audio to a fixed 16 kHz / mono format at a
caller-specified duration.  Supported backends, tried in order:

1. **soundfile** (default) — lossless formats: WAV, FLAC, AIFF, AU, …
2. **FFmpeg subprocess** — compressed formats: M4A, AAC, MP3, OGG, OPUS,
   WMA, MP4, and anything else FFmpeg can decode.  No extra Python packages
   are required; only the ``ffmpeg`` binary must be on PATH.

The FFmpeg path is used automatically whenever soundfile raises an error
(e.g. "Format not recognised").
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Formats that soundfile cannot read and that should go straight to FFmpeg.
# (librosa.load falls back to audioread for these, but audioread is not
# installed in this environment — we use the FFmpeg subprocess directly.)
_COMPRESSED_EXTENSIONS: frozenset[str] = frozenset(
    {".m4a", ".aac", ".mp3", ".ogg", ".opus", ".wma", ".mp4", ".webm"}
)


def _load_via_ffmpeg(
    input_path: Path,
    target_sr: int,
) -> tuple[np.ndarray, int]:
    """Decode *any* audio file to float32 PCM via FFmpeg subprocess.

    FFmpeg writes raw 32-bit little-endian float samples to stdout at the
    requested sample rate and in mono.  No intermediate temp file is created.

    Parameters
    ----------
    input_path:
        Absolute path to the audio file.
    target_sr:
        Output sample rate in Hz.

    Returns
    -------
    tuple[numpy.ndarray, int]
        A mono float32 array and the output sample rate.

    Raises
    ------
    FileNotFoundError
        If the ``ffmpeg`` binary is not found on PATH.
    ValueError
        If FFmpeg exits non-zero (unrecognised or corrupt file).
    """
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise FileNotFoundError(
            "ffmpeg binary not found on PATH. "
            "Install FFmpeg or convert the file to WAV first: "
            f'ffmpeg -i "{input_path}" -ar {target_sr} -ac 1 output.wav'
        )

    cmd: list[str] = [
        ffmpeg_exe,
        "-v", "error",          # suppress banner, show only errors
        "-i", str(input_path),
        "-ar", str(target_sr),  # resample to target_sr in one pass
        "-ac", "1",             # downmix to mono
        "-f", "f32le",          # output format: 32-bit float LE
        "-nostdin",             # never wait for stdin
        "pipe:1",               # write PCM to stdout
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"FFmpeg timed out decoding '{input_path}'."
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise ValueError(
            f"FFmpeg failed to decode '{input_path}' (exit {result.returncode}): "
            f"{stderr}. "
            f"Try: ffmpeg -i \"{input_path}\" -ar {target_sr} -ac 1 output.wav"
        )

    if not result.stdout:
        raise ValueError(
            f"FFmpeg produced no audio output for '{input_path}'. "
            "The file may be silent, corrupt, or contain only video."
        )

    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return audio, target_sr


def preprocess_audio(
    audio_path: str | Path,
    target_sr: int = 16000,
    duration: float = 5.0,
) -> tuple[np.ndarray, int]:
    """Load, resample, and pad/truncate audio to a fixed length.

    Parameters
    ----------
    audio_path:
        Path to an audio file.  WAV and FLAC are read directly via
        soundfile; M4A, AAC, MP3, OGG, OPUS, WMA and any other format
        supported by FFmpeg are decoded via a subprocess pipe.
    target_sr:
        Target sample rate in Hz (default 16,000 Hz).
    duration:
        Desired audio duration in seconds.  The output is truncated or
        zero-padded to exactly ``int(round(duration * target_sr))`` samples.

    Returns
    -------
    tuple[numpy.ndarray, int]
        A mono float32 array and the final sample rate.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist, or FFmpeg is required but not found.
    ValueError
        If the file cannot be decoded by any available backend.
    """
    input_path = Path(audio_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    suffix = input_path.suffix.lower()

    # ── Backend selection ──────────────────────────────────────────────
    # 1. Known compressed formats → FFmpeg directly (avoids a misleading
    #    "Format not recognised" from soundfile before we even try FFmpeg).
    # 2. Everything else → soundfile first, FFmpeg on failure.

    if suffix in _COMPRESSED_EXTENSIONS:
        audio_array, original_sr = _load_via_ffmpeg(input_path, target_sr)
        # FFmpeg already resampled to target_sr, so skip the librosa step.
        needs_resample = False
    else:
        try:
            audio_array, original_sr = sf.read(
                input_path, dtype="float32", always_2d=False
            )
            if audio_array.size == 0:
                raise ValueError(f"Audio file is empty: {input_path}")
            needs_resample = original_sr != target_sr
        except Exception as sf_exc:
            # soundfile failed (e.g. an unrecognised container) — fall back
            # to FFmpeg which can handle almost any format.
            try:
                audio_array, original_sr = _load_via_ffmpeg(input_path, target_sr)
                needs_resample = False
            except (FileNotFoundError, ValueError):
                # Re-raise the original soundfile error with full context.
                raise ValueError(
                    f"Unable to read audio file '{input_path}': {sf_exc}. "
                    f"Try converting to WAV: "
                    f'ffmpeg -i "{input_path}" -ar {target_sr} -ac 1 output.wav'
                ) from sf_exc

    # ── Post-load normalisation ────────────────────────────────────────
    audio_array = np.asarray(audio_array, dtype=np.float32)

    # Ensure mono (soundfile path may return stereo).
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    # Resample if soundfile loaded at the wrong rate.
    if needs_resample:
        audio_array = librosa.resample(
            y=audio_array,
            orig_sr=original_sr,
            target_sr=target_sr,
            res_type="soxr_hq",
        )

    # Truncate or zero-pad to the exact required length.
    target_samples = int(round(duration * target_sr))
    audio_array = librosa.util.fix_length(audio_array, size=target_samples)

    return audio_array.astype(np.float32), target_sr

