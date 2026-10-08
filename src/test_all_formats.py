"""Audio format support test for AudioShield Forensics.

Converts the reference FLAC file to WAV, FLAC, MP3, OGG, M4A, and AAC using
FFmpeg, then runs each through the existing ``preprocess_audio()`` function and
checks that the output dimensions and sample rate are correct.

All temporary files are deleted whether the tests pass or fail.

Usage
-----
    python src/test_all_formats.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Allow running as a plain script from the project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.audio_preprocessing import preprocess_audio

# ── Constants ──────────────────────────────────────────────────────────────
SOURCE_FILE     = _ROOT / "data" / "LA" / "ASVspoof2019_LA_eval" / "flac" / "LA_E_5849185.flac"
TEST_DURATION   = 5.0
EXPECTED_SR     = 16_000
EXPECTED_SAMPLES = int(EXPECTED_SR * TEST_DURATION)  # 80 000

# (suffix, extra_ffmpeg_output_args)
# All conversions target 16 kHz mono; extra args control codec selection.
FORMATS: list[tuple[str, list[str]]] = [
    ("wav",  []),
    ("flac", []),
    ("mp3",  ["-q:a", "2"]),
    ("ogg",  ["-c:a", "libvorbis", "-q:a", "4"]),
    ("m4a",  ["-c:a", "aac", "-b:a", "128k"]),
    ("aac",  ["-c:a", "aac", "-b:a", "128k"]),
]


class Result(NamedTuple):
    """Outcome of a single format test."""
    fmt:    str
    passed: bool
    reason: str


# ── Helpers ────────────────────────────────────────────────────────────────

def _ffmpeg_convert(
    src: Path,
    dst: Path,
    extra: list[str],
) -> tuple[bool, str]:
    """Convert *src* to *dst* via FFmpeg at 16 kHz mono.

    Returns
    -------
    tuple[bool, str]
        ``(success, message)``
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found on PATH"

    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src),
           "-ar", str(EXPECTED_SR), "-ac", "1", *extra, str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timed out"
    if r.returncode != 0:
        stderr = r.stderr.decode(errors="replace").strip()
        return False, f"FFmpeg exit {r.returncode}: {stderr[:160]}"
    if not dst.is_file() or dst.stat().st_size == 0:
        return False, "FFmpeg produced an empty output file"
    return True, "converted OK"


def _run_preprocess(path: Path) -> tuple[bool, str]:
    """Call preprocess_audio and validate the result.

    Does NOT modify ``preprocess_audio`` — reports failures as-is.

    Returns
    -------
    tuple[bool, str]
        ``(passed, reason)``
    """
    try:
        audio, sr = preprocess_audio(path, duration=TEST_DURATION)
    except Exception as exc:
        return False, f"preprocess_audio raised: {exc}"

    if sr != EXPECTED_SR:
        return False, f"sample rate {sr} != {EXPECTED_SR}"
    if len(audio) != EXPECTED_SAMPLES:
        return False, f"length {len(audio)} != {EXPECTED_SAMPLES}"
    if not isinstance(audio, np.ndarray):
        return False, f"returned {type(audio).__name__}, expected ndarray"
    if not np.isfinite(audio).all():
        return False, "output contains NaN or Inf"
    if np.all(audio == 0):
        return False, "output is all zeros (likely silent / decode failure)"

    peak = float(np.abs(audio).max())
    return True, f"{len(audio)} samples @ {sr} Hz  peak={peak:.4f}"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the format suite and print a summary."""
    print("=" * 60)
    print("  AudioShield Forensics — Format Support Test")
    print("=" * 60)
    print(f"  Source : {SOURCE_FILE.name}")
    print(f"  Target : {EXPECTED_SR} Hz · {TEST_DURATION} s · {EXPECTED_SAMPLES} samples")
    print()

    if not SOURCE_FILE.is_file():
        print(f"ABORT: source file not found: {SOURCE_FILE}")
        sys.exit(1)
    if not shutil.which("ffmpeg"):
        print("ABORT: ffmpeg not found on PATH")
        sys.exit(1)

    results: list[Result] = []

    with tempfile.TemporaryDirectory(prefix="asf_fmt_test_") as tmpdir:
        tmp = Path(tmpdir)

        for suffix, extra in FORMATS:
            dst = tmp / f"sample.{suffix}"
            print(f"  .{suffix:<5} ", end="", flush=True)

            ok, msg = _ffmpeg_convert(SOURCE_FILE, dst, extra)
            if not ok:
                print(f"CONVERT-FAIL  {msg}")
                results.append(Result(suffix, False, f"Conversion failed: {msg}"))
                continue

            passed, reason = _run_preprocess(dst)
            icon = "PASS" if passed else "FAIL"
            print(f"{icon}  {reason}")
            results.append(Result(suffix, passed, reason))

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    passed_n = sum(r.passed for r in results)
    for r in results:
        icon = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"  {icon}  .{r.fmt:<6}  {r.reason}")
    print()
    print(f"  {passed_n}/{len(results)} formats passed")
    print("=" * 60)

    if passed_n < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
