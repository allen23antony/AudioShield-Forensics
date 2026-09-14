"""Combine independent AI and metadata evidence for forensic review."""

from __future__ import annotations

from typing import Any


def combine_evidence(ai_result: dict[str, Any], metadata_analysis: dict[str, Any]) -> dict[str, Any]:
    """Combine AI output and metadata risk while keeping their sources separate."""
    ai_score = ai_result.get("ai_score", ai_result.get("score"))
    metadata_risk = metadata_analysis.get("risk_score", "low")
    ai_label = str(ai_result.get("label", ai_result.get("prediction", ""))).lower()
    if metadata_risk == "high" and ai_label in {"synthetic", "fake", "spoof"}:
        overall_risk = "high"
        recommendation = "Likely synthetic based on converging indicators; requires further investigation."
    elif metadata_risk == "low" and ai_label in {"authentic", "bonafide", "real"}:
        overall_risk = "low"
        recommendation = "Likely authentic based on the available evidence, subject to further verification."
    else:
        overall_risk = "medium"
        recommendation = "Requires further investigation because AI and metadata evidence are limited or mixed."
    return {
        "ai_score": ai_score,
        "metadata_risk": metadata_risk,
        "overall_risk": overall_risk,
        "evidence_summary": {
            "ai": ai_result,
            "metadata_findings": metadata_analysis.get("findings", []),
        },
        "recommendation": recommendation,
    }


def generate_forensic_assessment(evidence: dict[str, Any]) -> str:
    """Render a conservative combined forensic assessment."""
    summary = evidence.get("evidence_summary", {})
    return "\n".join(
        [
            "Audio Forensic Assessment",
            f"AI score: {evidence.get('ai_score')}",
            f"Metadata risk: {evidence.get('metadata_risk', 'low')}",
            f"Overall risk: {evidence.get('overall_risk', 'medium')}",
            f"AI evidence: {summary.get('ai', {})}",
            f"Metadata findings: {summary.get('metadata_findings', [])}",
            f"Recommendation: {evidence.get('recommendation', 'Requires further investigation.')}",
            "This assessment does not establish authenticity from metadata alone.",
        ]
    )

