"""Conservative forensic checks for audio metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse common ISO and ExifTool timestamp representations."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        for format_string in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M:%S%z"):
            try:
                return datetime.strptime(value.strip(), format_string)
            except ValueError:
                continue
    return None


def check_timestamp_consistency(metadata: dict[str, Any]) -> dict[str, Any]:
    """Check ordering and separation of creation and modification timestamps."""
    creation = _parse_timestamp(metadata.get("creation_time"))
    modification = _parse_timestamp(metadata.get("modification_time"))
    if not creation or not modification:
        return {"has_issue": False, "severity": "low", "description": "Insufficient timestamps for consistency checking."}
    if creation.tzinfo != modification.tzinfo:
        creation = creation.replace(tzinfo=None)
        modification = modification.replace(tzinfo=None)
    difference = modification - creation
    if difference.total_seconds() < 0:
        return {"has_issue": True, "severity": "high", "description": "Modification time predates creation time; requires further investigation."}
    if difference.days > 30:
        return {"has_issue": True, "severity": "medium", "description": "Creation and modification timestamps are more than 30 days apart; this is inconsistent."}
    return {"has_issue": False, "severity": "low", "description": "Creation and modification timestamps are consistent."}


def check_codec_consistency(metadata: dict[str, Any]) -> dict[str, Any]:
    """Check for clearly unusual codec and container combinations."""
    container = str(metadata.get("format") or "").lower()
    codec = str(metadata.get("codec") or "").lower()
    expected = {
        "flac": {"flac"},
        "wav": {"pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_s16be"},
        "mp3": {"mp3"},
        "ogg": {"vorbis", "opus"},
        "opus": {"opus"},
        "m4a": {"aac", "alac", "mp3"},
        "mp4": {"aac", "alac", "mp3"},
    }
    match = next((codecs for name, codecs in expected.items() if name in container), None)
    if match is not None and codec and codec not in match:
        return {"has_issue": True, "severity": "medium", "description": f"Codec '{codec}' is unusual for container '{container}'; requires further investigation."}
    return {"has_issue": False, "severity": "low", "description": "No obvious codec/container inconsistency identified."}


def check_device_consistency(metadata: dict[str, Any]) -> dict[str, Any]:
    """Assess whether device provenance metadata is present."""
    if metadata.get("device_info"):
        return {"has_issue": False, "severity": "low", "description": "Device information is present."}
    return {"has_issue": True, "severity": "low", "description": "Device information is missing; provenance cannot be verified from metadata alone."}


def check_encoder_consistency(metadata: dict[str, Any]) -> dict[str, Any]:
    """Check encoder/software fields for missing or potentially suspicious values."""
    encoder = str(metadata.get("encoder") or metadata.get("software") or "").lower()
    if not encoder:
        return {"has_issue": True, "severity": "low", "description": "Encoder or software information is missing."}
    suspicious = ("deepfake", "voice clone", "synthetic", "ai-generated", "so-vits")
    if any(marker in encoder for marker in suspicious):
        return {"has_issue": True, "severity": "high", "description": "Encoder/software metadata contains a marker associated with synthetic generation; requires further investigation."}
    return {"has_issue": False, "severity": "low", "description": "No suspicious encoder/software marker identified."}


def analyze_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Run metadata checks and aggregate findings without asserting authenticity."""
    checks = {
        "timestamps": check_timestamp_consistency(metadata),
        "codec": check_codec_consistency(metadata),
        "device": check_device_consistency(metadata),
        "encoder": check_encoder_consistency(metadata),
    }
    findings = [
        {"field": field, **finding}
        for field, finding in checks.items()
        if finding["has_issue"]
    ]
    scores = {"low": 0, "medium": 1, "high": 2}
    risk_score = max((finding["severity"] for finding in findings), key=lambda value: scores[value], default="low")
    expected_fields = ("creation_time", "modification_time", "device_info", "encoder", "software", "comments")
    missing_fields = [field for field in expected_fields if not metadata.get(field)]
    suspicious_fields = [finding["field"] for finding in findings if finding["severity"] != "low"]
    return {
        "findings": findings,
        "risk_score": risk_score,
        "suspicious_fields": suspicious_fields,
        "missing_fields": missing_fields,
    }


def generate_metadata_report(analysis: dict[str, Any]) -> str:
    """Render metadata analysis as a human-readable text report."""
    lines = [
        "Metadata Forensics Report",
        f"Risk score: {analysis.get('risk_score', 'low')}",
        f"Suspicious fields: {', '.join(analysis.get('suspicious_fields', [])) or 'None'}",
        f"Missing fields: {', '.join(analysis.get('missing_fields', [])) or 'None'}",
        "",
        "Findings:",
    ]
    findings = analysis.get("findings", [])
    lines.extend(
        f"- [{finding['severity']}] {finding['field']}: {finding['description']}"
        for finding in findings
    )
    if not findings:
        lines.append("- No inconsistencies identified.")
    lines.append("Metadata findings are indicators only and require further investigation.")
    return "\n".join(lines)

