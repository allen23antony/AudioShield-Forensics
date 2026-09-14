"""End-to-end AudioShield forensic pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..fusion.combine import combine_evidence
from ..fusion.report import generate_fusion_report, generate_json_report
from ..forensics.analyze import analyze_signal
from ..metadata.analyzer import analyze_metadata
from ..metadata.extractor import extract_metadata, get_metadata_summary
from ..report.generator import generate_comprehensive_report
from .predict import predict_all
from .save import save_predictions


def run_full_pipeline(audio_path: Path) -> dict[str, Any]:
    """Run AI, metadata, signal, fusion, and PDF report generation."""
    audio = Path(audio_path)
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    ai_predictions = predict_all(audio)
    ai_path = results_dir / "ai_predictions.json"
    save_predictions(ai_predictions, ai_path)

    metadata = extract_metadata(audio)
    metadata_analysis = analyze_metadata(metadata)
    metadata_payload = {
        "metadata": metadata,
        "summary": get_metadata_summary(metadata),
        "analysis": metadata_analysis,
    }
    (results_dir / "metadata_report.json").write_text(
        json.dumps(metadata_payload, indent=2, default=str), encoding="utf-8"
    )

    signal_analysis = analyze_signal(audio)
    (results_dir / "signal_report.json").write_text(
        json.dumps(signal_analysis, indent=2, default=str), encoding="utf-8"
    )

    fusion = combine_evidence(ai_predictions, metadata_analysis, signal_analysis)
    generate_json_report(fusion, results_dir / "fusion_report.json")
    generate_fusion_report(fusion, results_dir / "fusion_report.txt")
    report_path = generate_comprehensive_report(
        audio,
        results_dir / "metadata_report.json",
        results_dir / "signal_report.json",
        results_dir / "fusion_report.json",
        results_dir / "comprehensive_report.pdf",
    )
    return {
        "ai": ai_predictions,
        "metadata": metadata_payload,
        "signal": signal_analysis,
        "fusion": fusion,
        "report_path": str(report_path),
    }


def run_pipeline_cli() -> int:
    """Run the complete forensic pipeline from the command line."""
    parser = argparse.ArgumentParser(description="Run the complete AudioShield forensic pipeline.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to the audio file.")
    args = parser.parse_args()
    try:
        result = run_full_pipeline(args.audio)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["fusion"], indent=2, default=str))
    print(f"\nComprehensive report saved to {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_pipeline_cli())
