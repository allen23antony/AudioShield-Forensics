"""Command-line interface for evidence fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .combine import combine_evidence, load_ai_results
from .report import generate_fusion_report, generate_json_report, print_fusion_summary


def _load_json(path: Path | None, default: dict[str, Any]) -> dict[str, Any]:
    """Load an optional JSON object, returning a clear error record if invalid."""
    if path is None:
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {**default, "error": f"Results file not found: {path}"}
    except (OSError, json.JSONDecodeError) as exc:
        return {**default, "error": f"Unable to load results from {path}: {exc}"}
    return value if isinstance(value, dict) else {**default, "error": f"Results must be a JSON object: {path}"}


def main() -> int:
    """Run evidence fusion from optional source result files."""
    parser = argparse.ArgumentParser(description="Fuse AI, metadata, and signal forensic evidence.")
    parser.add_argument("--audio", required=True, type=Path, help="Audio file under examination.")
    parser.add_argument("--ai-results", type=Path, help="AI results JSON.")
    parser.add_argument("--metadata-results", type=Path, help="Metadata report JSON.")
    parser.add_argument("--signal-results", type=Path, help="Signal report JSON.")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")

    ai_results = load_ai_results(args.ai_results)
    metadata_wrapper = _load_json(args.metadata_results or Path("results") / "metadata_report.json", {})
    signal_analysis = _load_json(args.signal_results or Path("results") / "signal_report.json", {})
    metadata_analysis = metadata_wrapper.get("analysis", metadata_wrapper)
    evidence = combine_evidence(ai_results, metadata_analysis, signal_analysis)
    generate_json_report(evidence, Path("results") / "fusion_report.json")
    generate_fusion_report(evidence, Path("results") / "fusion_report.txt")
    print_fusion_summary(evidence)
    print("\nReports saved to results\\fusion_report.json and results\\fusion_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
