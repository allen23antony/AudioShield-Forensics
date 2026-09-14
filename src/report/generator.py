"""Generate professional, evidence-separated PDF forensic reports."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_report_id() -> str:
    """Return a unique report identifier in the ASF-YYYYMMDD-XXXX format."""
    date_part = datetime.now().strftime("%Y%m%d")
    return f"ASF-{date_part}-{uuid.uuid4().hex[:4].upper()}"


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object, returning an empty object when the file is unavailable."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _value(mapping: Mapping[str, Any], *keys: str, default: Any = "Not available") -> Any:
    """Get the first present value from a mapping."""
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def _text(value: Any) -> str:
    """Convert a value to safe display text."""
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=True)
    return str(value)


def _risk_color(risk: Any) -> colors.Color:
    risk_text = str(risk).lower()
    if risk_text == "high":
        return colors.HexColor("#B91C1C")
    if risk_text == "medium":
        return colors.HexColor("#B45309")
    if risk_text == "low":
        return colors.HexColor("#166534")
    return colors.HexColor("#475569")


def _table(rows: Iterable[Iterable[Any]], widths: Optional[List[float]] = None) -> Table:
    """Create a consistently styled report table."""
    converted = [[Paragraph(_text(cell), _BODY_STYLE) for cell in row] for row in rows]
    table = Table(converted, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


_STYLES = getSampleStyleSheet()
_TITLE_STYLE = ParagraphStyle("ReportTitle", parent=_STYLES["Title"], fontName="Helvetica-Bold",
                              fontSize=25, leading=31, alignment=TA_CENTER, textColor=colors.HexColor("#17324D"))
_SECTION_STYLE = ParagraphStyle("Section", parent=_STYLES["Heading1"], fontName="Helvetica-Bold",
                                fontSize=16, leading=20, textColor=colors.HexColor("#17324D"),
                                spaceBefore=8, spaceAfter=10)
_SUBSECTION_STYLE = ParagraphStyle("Subsection", parent=_STYLES["Heading2"], fontName="Helvetica-Bold",
                                   fontSize=11, leading=14, textColor=colors.HexColor("#285A87"),
                                   spaceBefore=7, spaceAfter=5)
_BODY_STYLE = ParagraphStyle("Body", parent=_STYLES["BodyText"], fontName="Helvetica",
                             fontSize=8.7, leading=12, spaceAfter=4)
_SMALL_STYLE = ParagraphStyle("Small", parent=_BODY_STYLE, fontSize=7.5, leading=9)
_CENTER_STYLE = ParagraphStyle("Center", parent=_BODY_STYLE, alignment=TA_CENTER)
_BULLET_STYLE = ParagraphStyle("Bullet", parent=_BODY_STYLE, leftIndent=14, firstLineIndent=-8)


def format_metadata_section(metadata: Dict[str, Any]) -> List[Any]:
    """Return flowables for metadata extraction and forensic findings."""
    evidence = metadata.get("metadata_evidence", {})
    details = metadata.get("metadata") or metadata.get("summary") or {}
    analysis = metadata.get("analysis", {})
    findings = evidence.get("findings") or analysis.get("findings") or []
    flowables: List[Any] = [Paragraph("Extracted metadata", _SUBSECTION_STYLE)]
    rows = [["Field", "Value"]]
    for key, value in details.items():
        if key not in {"raw_ffprobe", "raw_exiftool"}:
            rows.append([key.replace("_", " ").title(), _text(value)])
    flowables.append(_table(rows, [1.65 * inch, 5.3 * inch]))
    flowables.append(Spacer(1, 8))
    flowables.append(Paragraph(f"Risk score: <b>{_text(evidence.get('risk_score', analysis.get('risk_score')))}</b>",
                               _BODY_STYLE))
    flowables.append(Paragraph("Findings", _SUBSECTION_STYLE))
    finding_rows = [["Severity", "Field", "Description"]]
    for finding in findings:
        if isinstance(finding, dict):
            finding_rows.append([finding.get("severity", "unknown"), finding.get("field", "unknown"),
                                 finding.get("description", "No description provided.")])
    if len(finding_rows) == 1:
        finding_rows.append(["None", "None", "No metadata findings were reported."])
    flowables.append(_table(finding_rows, [0.8 * inch, 1.25 * inch, 4.9 * inch]))
    flowables.append(Spacer(1, 5))
    flowables.append(Paragraph("Metadata indicators are not conclusive on their own and require further investigation.",
                               _SMALL_STYLE))
    return flowables


def _feature_rows(features: Mapping[str, Any]) -> List[List[str]]:
    rows = [["Feature", "Mean", "Std", "Min", "Max"]]
    for name, values in features.items():
        if isinstance(values, dict):
            rows.append([name.replace("_", " ").title(), _text(values.get("mean")),
                         _text(values.get("std")), _text(values.get("min")), _text(values.get("max"))])
        else:
            rows.append([name.replace("_", " ").title(), _text(values), "-", "-", "-"])
    return rows


def format_signal_section(signal: Dict[str, Any]) -> List[Any]:
    """Return flowables for spectral, temporal, and anomaly evidence."""
    flowables: List[Any] = []
    flowables.append(Paragraph("Spectral features", _SUBSECTION_STYLE))
    spectral = signal.get("spectral_features", {})
    flowables.append(_table(_feature_rows(spectral), [2.0 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch]))
    flowables.append(Spacer(1, 8))
    flowables.append(Paragraph("Temporal features", _SUBSECTION_STYLE))
    temporal = signal.get("temporal_features", {})
    flowables.append(_table(_feature_rows(temporal), [2.0 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch]))
    anomalies = signal.get("anomalies", {})
    anomaly_list = anomalies.get("anomalies", []) if isinstance(anomalies, dict) else []
    flowables.append(Spacer(1, 8))
    flowables.append(Paragraph(
        f"Risk score: <b>{_text(signal.get('signal_risk', anomalies.get('risk_score', 'Not available')))}</b>",
        _BODY_STYLE))
    flowables.append(Paragraph("Anomalies", _SUBSECTION_STYLE))
    if anomaly_list:
        for anomaly in anomaly_list:
            flowables.append(Paragraph(f"• {_text(anomaly)}", _BULLET_STYLE))
    else:
        flowables.append(Paragraph("No signal-level anomalies were reported.", _BODY_STYLE))
    return flowables


def format_fusion_section(fusion: Dict[str, Any]) -> List[Any]:
    """Return flowables for AI, metadata, signal, and combined fusion evidence."""
    combined = fusion.get("fusion", fusion)
    ai = fusion.get("ai_evidence", {})
    metadata = fusion.get("metadata_evidence", {})
    signal = fusion.get("signal_evidence", {})
    rows = [
        ["Evidence source", "Risk / result"],
        ["AI", f"{_text(ai.get('consensus'))} (risk: {_text(ai.get('ai_risk'))})"],
        ["Metadata", _text(metadata.get("risk_score"))],
        ["Signal", _text(signal.get("signal_risk"))],
        ["Overall", _text(combined.get("overall_risk"))],
    ]
    flowables = [_table(rows, [2.0 * inch, 4.95 * inch]), Spacer(1, 8)]
    flowables.append(Paragraph(f"Assessment: <b>{_text(combined.get('assessment'))}</b> | "
                               f"Confidence: <b>{_text(combined.get('confidence'))}</b>", _BODY_STYLE))
    flowables.append(Paragraph(f"Reasoning: {_text(combined.get('reasoning'))}", _BODY_STYLE))
    flowables.append(Paragraph(f"Recommendation: {_text(combined.get('recommendation'))}", _BODY_STYLE))
    flowables.append(Paragraph("AI model breakdown", _SUBSECTION_STYLE))
    model_rows = [["Model", "Label", "Confidence"]]
    for model_name in ("cnn", "svm", "aasist"):
        result = ai.get(model_name, {})
        model_rows.append([model_name.upper(), result.get("label", "unknown"), result.get("confidence", "Not available")])
    flowables.append(_table(model_rows, [1.5 * inch, 2.0 * inch, 3.45 * inch]))
    return flowables


def _bullet_list(items: Iterable[Any]) -> List[Any]:
    return [Paragraph(f"• {_text(item)}", _BULLET_STYLE) for item in items]


def _section(title: str) -> List[Any]:
    return [Paragraph(title, _SECTION_STYLE)]


def _load_ai_evaluations() -> Dict[str, Any]:
    """Load Eval-set metrics for each model, tolerating unavailable result files."""
    paths = {
        "CNN": Path("results") / "cnn_final_eval_results.json",
        "SVM": Path("results") / "svm_eval_results.json",
        "AASIST": Path("results") / "aasist" / "aasist_eval_results.json",
    }
    evaluations: Dict[str, Any] = {}
    for model, path in paths.items():
        payload = _load_json(path)
        metrics = payload.get("metrics", payload)
        if model == "CNN":
            eer = payload.get("threshold_independent_metrics", {}).get("eer")
            f1 = metrics.get("f1_score")
        else:
            eer = metrics.get("eer")
            f1 = metrics.get("f1_score", metrics.get("f1"))
        evaluations[model] = {
            "accuracy": metrics.get("accuracy"),
            "f1": f1,
            "eer": eer,
            "path": str(path),
            "raw": payload,
        }
    available = [
        (model, values)
        for model, values in evaluations.items()
        if all(values.get(metric) is not None for metric in ("accuracy", "f1", "eer"))
    ]
    metric_ranks: Dict[str, Dict[str, int]] = {model: {} for model, _ in available}
    for metric, reverse in (("accuracy", True), ("f1", True), ("eer", False)):
        ordered = sorted(available, key=lambda item: float(item[1][metric]), reverse=reverse)
        for rank, (model, _) in enumerate(ordered, start=1):
            metric_ranks[model][metric] = rank
    ranked = sorted(
        available,
        key=lambda item: (
            sum(metric_ranks[item[0]].values()) / 3,
            float(item[1]["eer"]),
        ),
    )
    for rank, (model, values) in enumerate(ranked, start=1):
        values["rank"] = rank
    return evaluations


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_ai_detection_section(evaluations: Dict[str, Any]) -> List[Any]:
    """Return flowables for the fair official-Eval model comparison."""
    rows = [["Model", "Accuracy", "F1", "EER", "Rank"]]
    for model in ("CNN", "SVM", "AASIST"):
        values = evaluations.get(model, {})
        rank = values.get("rank")
        rows.append([
            model,
            _format_percent(values.get("accuracy")),
            _text(f"{float(values['f1']):.4f}") if values.get("f1") is not None else "N/A",
            _text(f"{float(values['eer']):.4f}") if values.get("eer") is not None else "N/A",
            f"{rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}" if rank else "N/A",
        ])
    best = next((model for model in ("CNN", "SVM", "AASIST") if evaluations.get(model, {}).get("rank") == 1), None)
    return [
        Paragraph("Official Eval-set comparison", _SUBSECTION_STYLE),
        _table(rows, [1.15 * inch, 1.35 * inch, 1.2 * inch, 1.2 * inch, 1.05 * inch]),
        Spacer(1, 6),
        Paragraph(
            f"Best model: <b>{best or 'Not available'}</b> (best balanced rank across accuracy, F1, and EER; "
            "lower EER is better).",
            _BODY_STYLE,
        ),
    ]


def _resolve_plot(path_value: Any, roots: Iterable[Path]) -> Optional[Path]:
    if not path_value:
        return None
    candidate = Path(str(path_value))
    candidates = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
    for item in candidates:
        if item.exists() and item.is_file():
            return item
    return None


def _page_header_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(0.65 * inch, height - 0.48 * inch, width - 0.65 * inch, height - 0.48 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(0.65 * inch, height - 0.37 * inch, "AudioShield Forensics | Confidential")
    canvas.drawRightString(width - 0.65 * inch, 0.38 * inch, f"Page {document.page}")
    canvas.restoreState()


def generate_comprehensive_report(
    audio_path: Path,
    metadata_report_path: Path,
    signal_report_path: Path,
    fusion_report_path: Path,
    output_path: Path,
) -> Path:
    """Generate a multi-page PDF combining all available forensic evidence."""
    audio_path = Path(audio_path)
    metadata_path = Path(metadata_report_path)
    signal_path = Path(signal_report_path)
    fusion_path = Path(fusion_report_path)
    output_path = Path(output_path)
    metadata = _load_json(metadata_path)
    signal = _load_json(signal_path)
    fusion = _load_json(fusion_path)
    combined = fusion.get("fusion", fusion)
    metadata_summary = metadata.get("metadata") or metadata.get("summary") or {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output_path), pagesize=A4, rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.68 * inch, bottomMargin=0.58 * inch, title="Audio Forensic Analysis Report",
        author="AudioShield Forensics",
    )
    story: List[Any] = []
    report_id = generate_report_id()
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    story.extend([Spacer(1, 1.0 * inch), Paragraph("AUDIOSHIELD FORENSICS", _TITLE_STYLE),
                  Spacer(1, 0.18 * inch), Paragraph("Audio Forensic Analysis Report", _TITLE_STYLE),
                  Spacer(1, 0.45 * inch)])
    title_rows = [["Report ID", report_id], ["Generated", generated], ["File analyzed", audio_path.name]]
    story.append(_table(title_rows, [1.4 * inch, 5.55 * inch]))
    story.extend([Spacer(1, 2.0 * inch),
                  Paragraph("This report presents separate evidence streams and conservative forensic indicators. "
                            "It is not a substitute for expert examination of the original evidence.", _CENTER_STYLE),
                  PageBreak()])

    story.extend(_section("1. Executive Summary"))
    risk = combined.get("overall_risk", "Not available")
    assessment = combined.get("assessment", "Not available")
    ai_evaluations = _load_ai_evaluations()
    story.append(Paragraph(f"Overall risk assessment: <b><font color='{_risk_color(risk).hexval()}'>{_text(risk)}</font></b>",
                           _BODY_STYLE))
    story.append(Paragraph(f"Overall assessment: <b>{_text(assessment)}</b> | Confidence: "
                           f"<b>{_text(combined.get('confidence'))}</b>", _BODY_STYLE))
    story.append(Paragraph("Key findings", _SUBSECTION_STYLE))
    key_findings = combined.get("evidence_summary", [])
    story.extend(_bullet_list(key_findings or ["No combined findings were reported."]))
    ai_models = ", ".join(
        f"{name} {_format_percent(values.get('accuracy'))}" for name, values in ai_evaluations.items()
    )
    story.append(Paragraph(f"Official Eval AI comparison: {ai_models}.", _BODY_STYLE))
    story.append(Paragraph(f"Recommendation: {_text(combined.get('recommendation'))}", _BODY_STYLE))

    story.extend(_section("2. File Information"))
    file_rows = [["Property", "Value"]]
    info = {
        "File name": _value(metadata_summary, "file_name", default=audio_path.name),
        "File size (bytes)": _value(metadata_summary, "file_size", default=audio_path.stat().st_size if audio_path.exists() else "Not available"),
        "Format": _value(metadata_summary, "format"),
        "Duration (seconds)": _value(metadata_summary, "duration"),
        "Sample rate (Hz)": _value(metadata_summary, "sample_rate"),
        "Channels": _value(metadata_summary, "channels"),
        "Codec": _value(metadata_summary, "codec"),
        "Bitrate (bits/sec)": _value(metadata_summary, "bitrate"),
        "Creation time": _value(metadata_summary, "creation_time"),
        "Modification time": _value(metadata_summary, "modification_time"),
    }
    file_rows.extend([[key, value] for key, value in info.items()])
    story.append(_table(file_rows, [2.15 * inch, 4.8 * inch]))

    story.extend(_section("3. AI Detection Results"))
    story.extend(format_ai_detection_section(ai_evaluations))
    story.extend(_section("4. Metadata Forensics"))
    story.extend(format_metadata_section(metadata))
    story.extend(_section("5. Signal Forensics"))
    story.extend(format_signal_section(signal))
    story.extend(_section("6. Evidence Fusion"))
    story.extend(format_fusion_section(fusion))

    story.extend(_section("7. Visual Analysis"))
    plot_values = signal.get("plots", [])
    plot_names = [("Waveform", "Time-domain waveform"), ("Spectrogram", "Frequency content over time"),
                  ("MFCC", "Mel-frequency cepstral coefficients"), ("Pitch contour", "Estimated fundamental frequency")]
    roots = [signal_path.parent, output_path.parent, Path.cwd(), audio_path.parent]
    for index, (name, caption) in enumerate(plot_names):
        plot_value = plot_values[index] if index < len(plot_values) else None
        plot_path = _resolve_plot(plot_value, roots)
        if plot_path:
            image = Image(str(plot_path), width=5.9 * inch, height=2.25 * inch)
            image.hAlign = "CENTER"
            story.extend([image, Paragraph(f"{name}: {caption}.", _CENTER_STYLE), Spacer(1, 8)])
        else:
            story.append(Paragraph(f"{name}: plot not available.", _BODY_STYLE))

    story.extend(_section("8. Conclusion"))
    story.append(Paragraph(f"The combined evidence indicates an overall assessment of <b>{_text(assessment)}</b> "
                           f"with a reported risk level of <b>{_text(risk)}</b>. These are indicators from the "
                           "available reports and should be interpreted with the original audio and chain of custody.",
                           _BODY_STYLE))
    story.append(Paragraph("Limitations: AI results may be unavailable; metadata can be edited or lost; signal "
                           "features are influenced by recording conditions and preprocessing; and this report does "
                           "not establish authenticity by itself.", _BODY_STYLE))

    story.extend(_section("9. Appendix"))
    story.append(Paragraph("Raw AI evaluation results", _SUBSECTION_STYLE))
    story.append(Paragraph(f"<font name='Courier'>{_text({model: values['raw'] for model, values in ai_evaluations.items()})}</font>", _SMALL_STYLE))
    story.append(Paragraph("Raw metadata report", _SUBSECTION_STYLE))
    story.append(Paragraph(f"<font name='Courier'>{_text(metadata)}</font>", _SMALL_STYLE))
    story.append(Paragraph("Raw signal report", _SUBSECTION_STYLE))
    story.append(Paragraph(f"<font name='Courier'>{_text(signal)}</font>", _SMALL_STYLE))
    story.append(Paragraph("Methodology", _SUBSECTION_STYLE))
    story.append(Paragraph("This report compiles the supplied AI, metadata, signal, and fusion JSON outputs. "
                           "No new model inference is performed during report generation. Evidence sources remain "
                           "separated to preserve interpretability.", _BODY_STYLE))
    document.build(story, onFirstPage=_page_header_footer, onLaterPages=_page_header_footer)
    return output_path
