"""High-level signal forensic analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audio_preprocessing import preprocess_audio
from .signal import detect_signal_anomalies, extract_signal_features
from .temporal import analyze_temporal_consistency
from .visualize import generate_forensic_plots


def analyze_signal(audio_path: Path) -> dict[str, Any]:
    """Load an audio file and run signal, temporal, and visualization analysis."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, sr = preprocess_audio(path)
    features = extract_signal_features(audio, sr)
    anomalies = detect_signal_anomalies(features)
    temporal = analyze_temporal_consistency(audio, sr)
    plot_paths = generate_forensic_plots(audio, sr, Path("results") / "signal_plots")
    risk_levels = {"low": 0, "medium": 1, "high": 2}
    temporal_risk = "medium" if temporal["anomalies"] else "low"
    signal_risk = max((anomalies["risk_score"], temporal_risk), key=lambda level: risk_levels[level])
    return {
        "spectral_features": features["spectral"],
        "temporal_features": features["temporal"],
        "anomalies": anomalies,
        "temporal_analysis": temporal,
        "plots": [str(plot) for plot in plot_paths],
        "signal_risk": signal_risk,
        "summary": f"Signal risk is {signal_risk}. " + anomalies["summary"],
    }


def generate_signal_report(analysis: dict[str, Any]) -> str:
    """Render signal analysis as a human-readable report."""
    lines = [
        "Signal Forensics Report",
        f"Signal risk: {analysis.get('signal_risk', 'low')}",
        f"Summary: {analysis.get('summary', '')}",
        "",
        "Signal anomalies:",
    ]
    findings = analysis.get("anomalies", {}).get("anomalies", [])
    lines.extend(
        f"- [{item['severity']}] {item['feature_name']}: {item['value']} (expected {item['expected_range']})"
        for item in findings
    )
    if not findings:
        lines.append("- None identified.")
    lines.append(f"Temporal analysis: {analysis.get('temporal_analysis', {}).get('summary', '')}")
    lines.append("Signal indicators are not proof of synthetic audio and require further investigation.")
    return "\n".join(lines)

