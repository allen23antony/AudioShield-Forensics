"""Load and combine AI, metadata, and signal evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .risk import calculate_ai_risk, calculate_metadata_risk, calculate_overall_risk, calculate_signal_risk


def load_ai_results(result_path: Path | None = None) -> dict[str, Any]:
    """Load AI detection results, defaulting to the pipeline output location."""
    path = Path(result_path) if result_path is not None else Path("results") / "ai_predictions.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": f"Unable to load AI results from {path}: {exc}"}
    return data if isinstance(data, dict) else {"error": f"AI results must contain a JSON object: {path}"}


def _ai_evidence(ai_results: dict[str, Any], ai_risk: dict[str, Any]) -> dict[str, Any]:
    """Build the stable AI evidence output shape."""
    models = ai_risk.get("models", {})
    return {
        "cnn": models.get("cnn", {"label": "unknown", "confidence": None}),
        "svm": models.get("svm", {"label": "unknown", "confidence": None}),
        "aasist": models.get("aasist", {"label": "unknown", "confidence": None}),
        "consensus": ai_risk.get("consensus", "unknown"),
        "ai_risk": ai_risk.get("risk_level", "medium"),
        "details": ai_risk.get("details"),
        "source_error": ai_results.get("error"),
    }


def combine_evidence(
    ai_results: dict[str, Any],
    metadata_analysis: dict[str, Any],
    signal_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Combine all evidence sources while preserving their independent findings."""
    ai_risk = calculate_ai_risk(ai_results)
    metadata_risk = calculate_metadata_risk(metadata_analysis)
    signal_risk = calculate_signal_risk(signal_analysis)
    overall = calculate_overall_risk(ai_risk, metadata_risk, signal_risk)
    evidence: dict[str, Any] = {
        "ai_evidence": _ai_evidence(ai_results, ai_risk),
        "metadata_evidence": {
            "risk_score": metadata_risk["risk_level"],
            "findings": metadata_risk["findings"],
            "summary": metadata_risk["details"],
        },
        "signal_evidence": {
            "signal_risk": signal_risk["risk_level"],
            "anomalies": signal_risk["anomalies"],
            "summary": signal_risk["details"],
        },
        "fusion": {**overall, "evidence_summary": []},
    }
    evidence["fusion"]["evidence_summary"] = generate_evidence_summary(evidence)
    evidence["fusion"]["recommendation"] = (
        "Likely synthetic indicators require further investigation."
        if overall["assessment"] == "likely_synthetic"
        else "Likely authentic based on the available indicators, subject to further verification."
        if overall["assessment"] == "likely_authentic"
        else "Evidence is mixed or incomplete; requires further investigation."
    )
    return evidence


def get_risk_level(score: float) -> str:
    """Convert a numeric risk score using the requested thresholds."""
    if score < 0.3:
        return "low"
    if score <= 0.7:
        return "medium"
    return "high"


def generate_evidence_summary(evidence: dict[str, Any]) -> list[str]:
    """Generate concise bullet points describing source indicators."""
    ai = evidence.get("ai_evidence", {})
    metadata = evidence.get("metadata_evidence", {})
    signal = evidence.get("signal_evidence", {})
    summary = [
        f"AI consensus: {ai.get('consensus', 'unknown')} (risk: {ai.get('ai_risk', 'medium')}).",
        f"Metadata risk: {metadata.get('risk_score', 'medium')}; {len(metadata.get('findings', []))} finding(s).",
        f"Signal risk: {signal.get('signal_risk', 'medium')}; {len(signal.get('anomalies', []))} anomaly/anomalies.",
    ]
    if ai.get("source_error"):
        summary.append(f"AI evidence unavailable: {ai['source_error']}")
    return summary
