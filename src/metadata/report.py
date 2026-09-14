"""PDF report generation for metadata forensics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from .analyzer import generate_metadata_report
from .extractor import get_metadata_summary


def generate_pdf_report(
    metadata: dict[str, Any],
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """Create a simple PDF containing metadata and analysis findings."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Metadata Forensics Report", styles["Title"]), Spacer(1, 12)]
    story.append(Paragraph("Metadata", styles["Heading2"]))
    for key, value in get_metadata_summary(metadata).items():
        display_value = escape(str(value if value is not None else "N/A"))
        story.append(Paragraph(f"<b>{escape(key)}</b>: {display_value}", styles["BodyText"]))
    story.extend([Spacer(1, 12), Paragraph("Analysis", styles["Heading2"])])
    for line in generate_metadata_report(analysis).splitlines():
        story.append(Paragraph(escape(line), styles["BodyText"]))
    document.build(story)
