"""Transparent rule-based risk calculations for evidence fusion."""

from __future__ import annotations

from typing import Any


RISK_LEVELS = {"low": 0, "medium": 1, "high": 2}
SPOOF_LABELS = {"spoof", "synthetic", "fake", "deepfake", "bonafide_false"}
BONAFIDE_LABELS = {"bonafide", "bona fide", "authentic", "real", "genuine", "human"}


def _normalize_risk(value: Any, default: str = "medium") -> str:
    """Normalize a risk value to low, medium, or high."""
    normalized = str(value or "").strip().lower()
    return normalized if normalized in RISK_LEVELS else default


def _model_result(value: Any) -> tuple[str, float | None]:
    """Extract a normalized label and optional confidence from a model result."""
    if not isinstance(value, dict):
        return "", None
    label = str(value.get("label", value.get("prediction", value.get("class", "")))).strip().lower()
    confidence_value = value.get("confidence", value.get("score", value.get("probability")))
    try:
        confidence = float(confidence_value) if confidence_value is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and confidence > 1:
        confidence /= 100.0
    return label, max(0.0, min(1.0, confidence)) if confidence is not None else None


def calculate_ai_risk(ai_results: dict[str, Any]) -> dict[str, Any]:
    """Calculate AI risk from model labels using consensus, not arbitrary weights."""
    model_results: dict[str, dict[str, Any]] = {}
    labels: list[str] = []
    for model_name in ("cnn", "svm", "aasist"):
        label, confidence = _model_result(ai_results.get(model_name))
        if label:
            model_results[model_name] = {"label": label, "confidence": confidence}
            labels.append(label)
    if not labels:
        return {"risk_level": "medium", "confidence": 0.0, "consensus": "unknown", "details": "No AI model results were provided."}
    spoof_count = sum(label in SPOOF_LABELS for label in labels)
    bonafide_count = sum(label in BONAFIDE_LABELS for label in labels)
    if spoof_count > bonafide_count:
        consensus = "spoof"
    elif bonafide_count > spoof_count:
        consensus = "bonafide"
    else:
        consensus = "inconclusive"
    agreement = max(spoof_count, bonafide_count) / len(labels)
    confidence_values = [item["confidence"] for item in model_results.values() if item["confidence"] is not None]
    confidence = agreement if not confidence_values else sum(confidence_values) / len(confidence_values)
    risk_level = "high" if consensus == "spoof" and agreement >= 2 / 3 else "low" if consensus == "bonafide" and agreement >= 2 / 3 else "medium"
    return {
        "risk_level": risk_level,
        "confidence": float(confidence),
        "consensus": consensus,
        "details": f"{len(labels)} AI model result(s); {max(spoof_count, bonafide_count)} support the {consensus} consensus.",
        "models": model_results,
    }


def calculate_metadata_risk(metadata_analysis: dict[str, Any]) -> dict[str, Any]:
    """Use the existing metadata analyzer risk and findings without reweighting."""
    risk_level = _normalize_risk(metadata_analysis.get("risk_score"))
    findings = metadata_analysis.get("findings", [])
    return {
        "risk_level": risk_level,
        "findings": findings if isinstance(findings, list) else [],
        "details": metadata_analysis.get("summary", f"Metadata risk is {risk_level}."),
    }


def calculate_signal_risk(signal_analysis: dict[str, Any]) -> dict[str, Any]:
    """Use the existing signal analyzer risk and anomaly findings."""
    anomaly_block = signal_analysis.get("anomalies", {})
    risk_level = _normalize_risk(signal_analysis.get("signal_risk", anomaly_block.get("risk_score") if isinstance(anomaly_block, dict) else None))
    anomalies = anomaly_block.get("anomalies", []) if isinstance(anomaly_block, dict) else []
    return {
        "risk_level": risk_level,
        "anomalies": anomalies if isinstance(anomalies, list) else [],
        "details": signal_analysis.get("summary", f"Signal risk is {risk_level}."),
    }


def calculate_overall_risk(
    ai_risk: dict[str, Any],
    metadata_risk: dict[str, Any],
    signal_risk: dict[str, Any],
) -> dict[str, Any]:
    """Apply the documented rule-based fusion logic.

    A high source becomes high when another source is medium or high. Two or
    more medium sources produce medium risk. All-low sources produce low risk.
    Every remaining combination is medium because the evidence is mixed.
    """
    risks = [
        _normalize_risk(ai_risk.get("risk_level")),
        _normalize_risk(metadata_risk.get("risk_level")),
        _normalize_risk(signal_risk.get("risk_level")),
    ]
    high_count = risks.count("high")
    medium_count = risks.count("medium")
    if high_count and any(risk in {"medium", "high"} for risk in risks if risk != "high"):
        overall = "high"
    elif medium_count >= 2:
        overall = "medium"
    elif all(risk == "low" for risk in risks):
        overall = "low"
    else:
        overall = "medium"

    consensus = str(ai_risk.get("consensus", "unknown"))
    if overall == "high" and consensus == "spoof":
        assessment = "likely_synthetic"
    elif overall == "low" and consensus == "bonafide":
        assessment = "likely_authentic"
    else:
        assessment = "inconclusive"
    source_confidences = [float(ai_risk.get("confidence", 0.0) or 0.0)]
    confidence = sum(source_confidences) / len(source_confidences)
    reasoning = f"Source risks are AI={risks[0]}, metadata={risks[1]}, signal={risks[2]}; applied documented rule-based logic."
    return {"overall_risk": overall, "assessment": assessment, "confidence": confidence, "reasoning": reasoning}

