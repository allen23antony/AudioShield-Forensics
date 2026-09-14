"""Text and JSON reports for fused forensic evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _render_report(evidence: dict[str, Any]) -> str:
    """Render a readable report from the stable fusion schema."""
    fusion = evidence.get("fusion", {})
    lines = [
        "AudioShield Evidence Fusion Report",
        "=" * 36,
        f"Overall risk: {fusion.get('overall_risk', 'medium')}",
        f"Assessment: {fusion.get('assessment', 'inconclusive')}",
        f"Evidence confidence: {float(fusion.get('confidence', 0.0)):.2f}",
        "",
        "Evidence summary:",
    ]
    lines.extend(f"- {item}" for item in fusion.get("evidence_summary", []))
    lines.extend(
        [
            "",
            f"Reasoning: {fusion.get('reasoning', '')}",
            f"Recommendation: {fusion.get('recommendation', 'Requires further investigation.')}",
            "",
            "This report summarizes indicators from independent sources; it does not establish authenticity by itself.",
        ]
    )
    return "\n".join(lines)


def generate_fusion_report(evidence: dict[str, Any], output_path: Path) -> None:
    """Generate and save a human-readable fusion report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_report(evidence), encoding="utf-8")


def generate_json_report(evidence: dict[str, Any], output_path: Path) -> None:
    """Serialize the complete fused evidence to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")


def print_fusion_summary(evidence: dict[str, Any]) -> None:
    """Print the human-readable fusion report to the console."""
    print(_render_report(evidence))

