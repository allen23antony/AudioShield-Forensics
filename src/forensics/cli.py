"""Command-line interface for signal forensics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze_signal, generate_signal_report


def main() -> int:
    """Run signal analysis for a command-line audio path."""
    parser = argparse.ArgumentParser(description="Analyze signal-level audio forensics.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to the audio file.")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")
    try:
        analysis = analyze_signal(args.audio)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    report = generate_signal_report(analysis)
    print(report)
    output_path = Path("results") / "signal_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON report saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

