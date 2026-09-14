"""Command-line interface for comprehensive AudioShield forensic reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generator import generate_comprehensive_report


def build_parser() -> argparse.ArgumentParser:
    """Build the report CLI argument parser."""
    parser = argparse.ArgumentParser(description="Generate a comprehensive AudioShield forensic PDF report.")
    parser.add_argument("--audio", required=True, type=Path, help="Audio file to identify in the report.")
    parser.add_argument("--metadata", type=Path, default=Path("results/metadata_report.json"))
    parser.add_argument("--signal", type=Path, default=Path("results/signal_report.json"))
    parser.add_argument("--fusion", type=Path, default=Path("results/fusion_report.json"))
    parser.add_argument("--output", type=Path, default=Path("results/comprehensive_report.pdf"))
    return parser


def main() -> None:
    """Parse arguments and generate the requested report."""
    args = build_parser().parse_args()
    output = generate_comprehensive_report(args.audio, args.metadata, args.signal, args.fusion, args.output)
    print(f"Comprehensive forensic report generated: {output.resolve()}")


if __name__ == "__main__":
    main()
