"""Command-line entry point for metadata forensics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analyzer import analyze_metadata, generate_metadata_report
from .extractor import extract_metadata, get_metadata_summary


def main() -> int:
    """Extract and analyze metadata for an audio file."""
    parser = argparse.ArgumentParser(description="Extract and analyze audio metadata.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to the audio file.")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")

    metadata = extract_metadata(args.audio)
    analysis = analyze_metadata(metadata)
    report = generate_metadata_report(analysis)
    print(report)

    output_path = Path("results") / "metadata_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata": metadata,
        "summary": get_metadata_summary(metadata),
        "analysis": analysis,
        "report": report,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON report saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

