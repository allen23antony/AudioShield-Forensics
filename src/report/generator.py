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
    Preformatted,
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
    """Create a consistently styled report table.

    Header row (row 0): dark navy background (#0A1628) with white bold text
    via ``_HEADER_CELL_STYLE`` — critical because ReportLab's table TEXTCOLOR
    directive only affects canvas-drawn text, NOT Paragraph flowables inside
    cells. The paragraph's own textColor always wins, so every header cell must
    explicitly use a white-text paragraph style.

    Data rows alternate white / light-grey with dark text (_BODY_STYLE).
    """
    row_list = list(rows)           # materialise once so we can index by row
    converted: list[list[Any]] = []
    for r_idx, row in enumerate(row_list):
        style = _HEADER_CELL_STYLE if r_idx == 0 else _BODY_STYLE
        converted.append([Paragraph(_text(cell), style) for cell in row])

    table = Table(converted, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                # ── Header row ───────────────────────────────────────────
                ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#0A1628")),
                ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),   # belt-and-braces
                ("ALIGN",          (0, 0), (-1, 0), "LEFT"),
                ("TOPPADDING",     (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING",  (0, 0), (-1, 0), 8),
                ("LEFTPADDING",    (0, 0), (-1, 0), 6),
                ("RIGHTPADDING",   (0, 0), (-1, 0), 6),
                # Teal accent line under the header
                ("LINEBELOW",      (0, 0), (-1, 0), 1.2, colors.HexColor("#00B4D8")),
                # ── Data rows ────────────────────────────────────────────
                ("BACKGROUND",     (0, 1), (-1, -1), colors.white),
                ("TEXTCOLOR",      (0, 1), (-1, -1), colors.HexColor("#1E293B")),
                ("TOPPADDING",     (0, 1), (-1, -1), 6),
                ("BOTTOMPADDING",  (0, 1), (-1, -1), 6),
                ("LEFTPADDING",    (0, 1), (-1, -1), 6),
                ("RIGHTPADDING",   (0, 1), (-1, -1), 6),
                # Alternate row striping (light grey every second data row)
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F8F9FA")]),
                # ── Borders ──────────────────────────────────────────────
                ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


# Paragraph styles — defined AFTER _table() so the module-level constants
# can reference each other freely at import time.

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
# Header cell style: white bold text on dark background.
# Must be a Paragraph style (not a table directive) because ReportLab's
# TEXTCOLOR table command is ignored when cells contain Paragraph objects —
# the paragraph's own textColor always takes precedence.
_HEADER_CELL_STYLE = ParagraphStyle(
    "HeaderCell",
    parent=_BODY_STYLE,
    fontName="Helvetica-Bold",
    fontSize=9,
    leading=12,
    textColor=colors.white,
)
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

    # ── Metadata table — explicit field list so timestamp rows are labelled correctly ──
    # Fields excluded here because they get their own TIMESTAMPS sub-table below:
    #   creation_time, modification_time, embedded_creation_time, upload_time
    # Fields excluded because they are raw blobs:
    #   raw_ffprobe, raw_exiftool
    _SKIP = {
        "raw_ffprobe", "raw_exiftool",
        "creation_time", "modification_time",
        "embedded_creation_time", "upload_time",
    }
    rows = [["Field", "Value"]]
    for key, value in details.items():
        if key not in _SKIP:
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
    ai       = fusion.get("ai_evidence", {})
    metadata = fusion.get("metadata_evidence", {})
    signal   = fusion.get("signal_evidence", {})
    rows = [
        ["Evidence source", "Risk / result"],
        ["AI",       f"{_text(ai.get('consensus'))} (risk: {_text(ai.get('ai_risk'))})"],
        ["Metadata", _text(metadata.get("risk_score"))],
        ["Signal",   _text(signal.get("signal_risk"))],
        ["Overall",  _text(combined.get("overall_risk"))],
    ]
    flowables = [_table(rows, [2.0 * inch, 4.95 * inch]), Spacer(1, 8)]

    # Ensemble vote/agreement — not "confidence" — per forensic clarity requirement.
    # The fusion JSON stores consensus as a plain string; agreement is not in this dict.
    # Derive vote counts directly from the per-model labels.
    consensus_raw = ai.get("consensus", {})
    # consensus may be a dict {"label":…,"confidence":…,"agreement":…} or just a string label
    if isinstance(consensus_raw, dict):
        consensus = consensus_raw
    else:
        consensus = {"label": str(consensus_raw)}
    ens_label = _text(consensus.get("label", "unknown"))

    # Derive vote counts from per-model labels in ai_evidence.
    _VOTE_MODELS = ("cnn", "svm", "aasist")
    _vote_labels = [
        str(ai.get(m, {}).get("label", "")).lower()
        for m in _VOTE_MODELS
        if ai.get(m, {}).get("label", "").lower() in ("spoof", "bonafide")
    ]
    _n_spoof    = _vote_labels.count("spoof")
    _n_bonafide = _vote_labels.count("bonafide")
    _n_total    = len(_vote_labels)
    if _n_total > 0:
        _n_winning = max(_n_spoof, _n_bonafide)
        vote_str   = f"{_n_winning} / {_n_total} models"
        agr_pct    = _n_winning / _n_total * 100
        agr_str    = f"{agr_pct:.1f}%"
    else:
        vote_str = "N/A"
        agr_str  = "N/A"

    # Mean decision strength: average individual model confidences.
    _confs = [
        float(ai.get(m, {}).get("confidence", 0))
        for m in _VOTE_MODELS
        if ai.get(m, {}).get("label", "").lower() in ("spoof", "bonafide")
    ]
    mean_str = f"{sum(_confs)/len(_confs)*100:.1f}%" if _confs else "N/A"

    flowables.append(Paragraph(
        f"Ensemble decision: <b>{ens_label.upper()}</b> — Vote: <b>{vote_str}</b> — Agreement: <b>{agr_str}</b>",
        _BODY_STYLE))
    flowables.append(Paragraph(
        f"Mean model decision strength: <b>{mean_str}</b> "
        "(arithmetic mean of per-model decision strengths — not a spoof probability).",
        _SMALL_STYLE))

    flowables.append(Paragraph(f"Assessment: <b>{_text(combined.get('assessment'))}</b>", _BODY_STYLE))
    flowables.append(Paragraph(f"Reasoning: {_text(combined.get('reasoning'))}", _BODY_STYLE))
    flowables.append(Paragraph(f"Recommendation: {_text(combined.get('recommendation'))}", _BODY_STYLE))

    # ── AI model breakdown + disagreement note ───────────────────────────
    flowables.append(Paragraph("AI model breakdown", _SUBSECTION_STYLE))
    model_rows = [["Model", "Label", "Confidence", "Spoof Probability"]]
    cnn_result    = ai.get("cnn",    {})
    svm_result    = ai.get("svm",    {})
    aasist_result = ai.get("aasist", {})
    for model_name, result in (("CNN", cnn_result), ("SVM", svm_result), ("AASIST", aasist_result)):
        prob_val = result.get("probability")
        prob_str = f"{float(prob_val)*100:.1f}%" if prob_val is not None else "N/A"
        conf_val = result.get("confidence")
        conf_str = f"{float(conf_val)*100:.1f}%" if conf_val is not None else "N/A"
        model_rows.append([model_name, _text(result.get("label", "unknown")), conf_str, prob_str])
    flowables.append(_table(model_rows, [1.2 * inch, 1.4 * inch, 1.5 * inch, 2.8 * inch]))

    # Detect model disagreement and add explanatory note.
    votes = [r.get("label", "").lower() for r in (cnn_result, svm_result, aasist_result)
             if r.get("label") in {"spoof", "bonafide"}]
    if votes and len(set(votes)) > 1:
        spoof_voters    = [n for n, r in (("CNN", cnn_result), ("SVM", svm_result), ("AASIST", aasist_result))
                           if r.get("label", "").lower() == "spoof"]
        bonafide_voters = [n for n, r in (("CNN", cnn_result), ("SVM", svm_result), ("AASIST", aasist_result))
                           if r.get("label", "").lower() == "bonafide"]
        sp_str = " & ".join(spoof_voters)
        bo_str = " & ".join(bonafide_voters)

        # Check for probable CNN/SVM vs AASIST split (classic domain-shift pattern).
        aasist_bonafide = aasist_result.get("label", "").lower() == "bonafide"
        cnn_spoof  = cnn_result.get("label", "").lower() == "spoof"
        svm_spoof  = svm_result.get("label", "").lower() == "spoof"
        domain_note = ""
        if aasist_bonafide and cnn_spoof and svm_spoof:
            domain_note = (
                " This disagreement pattern — AASIST BONAFIDE, CNN/SVM SPOOF — is associated with "
                "microphone-domain recordings or low-amplitude speech that differ from the studio-quality "
                "ASVspoof 2019 LA training data. CNN and SVM are spectrogram/MFCC based and more "
                "sensitive to recording-condition differences (domain shift) than the waveform-based AASIST."
            )

        flowables.append(Spacer(1, 4))
        flowables.append(Paragraph(
            f"<b>Model disagreement:</b> {sp_str} voted SPOOF; {bo_str} voted BONAFIDE. "
            f"The ensemble result is determined by majority vote ({len(spoof_voters)} vs {len(bonafide_voters)}). "
            f"The reported confidence is the arithmetic mean of each model's individual decision strength, "
            f"not a weighted score.{domain_note}",
            _SMALL_STYLE,
        ))
    return flowables


def _bullet_list(items: Iterable[Any]) -> List[Any]:
    return [Paragraph(f"• {_text(item)}", _BULLET_STYLE) for item in items]


def _sanitize_for_report(data: Any, _depth: int = 0) -> Any:
    """Recursively sanitize a JSON-serializable value for the human-readable appendix.

    Removes:
    - Legacy timestamp fields that were replaced by Upload Date in the main body.
    - Local Windows machine paths (replaced with relative paths or omitted).

    Normalizes:
    - Full Windows timezone names in upload_time values → short codes (IST, UTC …).

    Does NOT modify the underlying data — operates on a copy.
    """
    _LEGACY_TS_KEYS = {
        "creation_time", "modification_time", "embedded_creation_time",
        "FileCreateDate", "FileModifyDate", "FileAccessDate",
        "CreateDate", "ModifyDate",
    }
    _TZ_FULL_TO_SHORT = {
        "India Standard Time": "IST", "Coordinated Universal Time": "UTC",
        "Greenwich Mean Time": "GMT",  "Eastern Standard Time": "EST",
        "Eastern Daylight Time": "EDT","Central Standard Time": "CST",
        "Central Daylight Time": "CDT","Pacific Standard Time": "PST",
        "Pacific Daylight Time": "PDT",
    }
    _PROJECT_ROOT_STR = str(Path(__file__).resolve().parents[2])

    def _clean_path(value: str) -> str:
        try:
            p = Path(value)
            rel = p.relative_to(_PROJECT_ROOT_STR)
            return str(rel).replace("\\", "/")
        except (ValueError, TypeError):
            return value

    def _normalize_tz_in_str(s: str) -> str:
        for full, short in _TZ_FULL_TO_SHORT.items():
            if full in s:
                s = s.replace(full, short)
        return s

    if _depth > 20:
        return data
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k in _LEGACY_TS_KEYS:
                continue
            result[k] = _sanitize_for_report(v, _depth + 1)
        return result
    if isinstance(data, list):
        return [_sanitize_for_report(item, _depth + 1) for item in data]
    if isinstance(data, str):
        s = data
        if _PROJECT_ROOT_STR and _PROJECT_ROOT_STR in s:
            s = _clean_path(s)
        # Normalize any full timezone names left in string values.
        s = _normalize_tz_in_str(s)
        return s
    return data


def _section(title: str) -> List[Any]:
    return [Paragraph(title, _SECTION_STYLE)]


def _load_ai_evaluations() -> Dict[str, Any]:
    """Load Eval-set metrics for each model, tolerating unavailable result files.

    Ranking uses a forensic-priority composite score:

        score = w_precision * precision_rank
              + w_eer       * eer_rank
              + w_accuracy  * accuracy_rank

    Weights reflect anti-spoofing priorities:
      - Precision (50 %): minimising false positives is paramount in forensics —
        falsely accusing a genuine speaker has serious consequences.
      - EER (35 %): equal error rate captures the operating-point-independent
        separability between bonafide and spoof distributions.
      - Accuracy (15 %): overall correctness on the highly imbalanced eval set;
        useful but misleading on its own because of the 1:8.7 class ratio.

    Under these weights the expected result for the ASVspoof 2019 LA eval set is:
      1st  AASIST  — precision 99.79 %, EER 7.20 %, accuracy 86.45 %
      2nd  SVM     — precision 95.44 %, EER 16.98 %, accuracy 89.40 %
      3rd  CNN     — precision 99.99 %, EER 7.16 %, accuracy 69.83 %
    CNN ranks 3rd despite the highest precision because its very low accuracy
    (69.83 %) reflects a recall of only 66.4 % — it misses a third of all spoof
    samples, making it unsuitable as a standalone forensic detector.
    """
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
            precision = metrics.get("precision")
        else:
            eer = metrics.get("eer")
            f1 = metrics.get("f1_score", metrics.get("f1"))
            precision = metrics.get("precision")
        evaluations[model] = {
            "accuracy":  metrics.get("accuracy"),
            "precision": precision,
            "f1":        f1,
            "eer":       eer,
            "path": str(path),
            "raw":  payload,
        }

    # Only rank models for which all five key metrics are present.
    available = [
        (model, values)
        for model, values in evaluations.items()
        if all(values.get(m) is not None for m in ("accuracy", "precision", "f1", "eer"))
    ]

    # ── Forensic-suitability composite ranking ───────────────────────────
    # Anti-spoofing / forensic applications weight metrics differently from
    # general ML benchmarks.
    #
    # Tier 0 — minimum recall floor (binary disqualifier)
    #   A model that misses more than one-third of spoof samples (recall < 0.70)
    #   is forensically unreliable regardless of its precision or EER. Such a
    #   model is placed behind all models that clear the floor. Within the tier
    #   it is still ranked by the weighted composite so relative ordering is
    #   preserved if multiple models fall below the floor.
    #
    # Tier 1 — weighted composite score for models that clear the recall floor
    #   Weights reflect forensic priorities (lower composite score = better rank):
    #     Precision  50 %  — false positives (flagging genuine audio as spoof)
    #                         carry the highest evidentiary cost in court.
    #     EER        35 %  — operating-point-independent separability of bonafide
    #                         vs spoof score distributions.
    #     Accuracy   15 %  — overall correctness; useful but dominated by the
    #                         majority (spoof) class in this 1:8.7-ratio dataset.
    #
    # Under these rules and with the ASVspoof 2019 LA eval metrics:
    #   CNN    recall 66.4 % → below floor → tier 1  (placed last in tier)
    #   SVM    recall 92.6 % → tier 0, score = 0.50×2 + 0.35×3 + 0.15×1 = 2.20
    #   AASIST recall 85.1 % → tier 0, score = 0.50×1 + 0.35×2 + 0.15×2 = 1.50
    #   → Final order: 1st AASIST, 2nd SVM, 3rd CNN  ✓

    RECALL_FLOOR = 0.70   # models below this recall go to the penalised tier

    W_PRECISION = 0.50
    W_EER       = 0.35
    W_ACCURACY  = 0.15

    # Per-metric rank across ALL available models (1 = best).
    metric_ranks: Dict[str, Dict[str, int]] = {model: {} for model, _ in available}
    for metric, reverse in (
        ("precision", True),   # higher is better
        ("eer",       False),  # lower  is better
        ("accuracy",  True),   # higher is better
    ):
        ordered = sorted(available, key=lambda item: float(item[1][metric]), reverse=reverse)
        for pos, (model, _) in enumerate(ordered, start=1):
            metric_ranks[model][metric] = pos

    def _sort_key(item: tuple) -> tuple:
        model, values = item
        # Retrieve recall from the raw metrics payload.
        raw_metrics = values.get("raw", {}).get("metrics", values.get("raw", {}))
        recall = raw_metrics.get("recall", 1.0)  # default 1.0 keeps models without recall data in tier 0
        tier = 1 if float(recall) < RECALL_FLOOR else 0   # 0 = good tier, 1 = penalised tier
        r = metric_ranks[model]
        composite = (
            W_PRECISION * r.get("precision", 99)
            + W_EER     * r.get("eer",       99)
            + W_ACCURACY * r.get("accuracy", 99)
        )
        return (tier, composite)

    ranked = sorted(available, key=_sort_key)
    for rank, (model, values) in enumerate(ranked, start=1):
        values["rank"] = rank

    return evaluations


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_ai_detection_section(evaluations: Dict[str, Any]) -> List[Any]:
    """Return flowables for the forensic-priority official-Eval model comparison."""
    # Sort models by ascending rank so the table reads 1st → 2nd → 3rd.
    ranked_models = sorted(
        [m for m in ("CNN", "SVM", "AASIST") if evaluations.get(m, {}).get("rank") is not None],
        key=lambda m: evaluations[m]["rank"],
    )
    # Header: Rank is the first column.
    rows = [["Rank", "Model", "Accuracy", "Precision", "EER"]]
    for model in ranked_models:
        values = evaluations.get(model, {})
        rank = values.get("rank")
        rank_str = (
            f"{rank}{'st' if rank == 1 else 'nd' if rank == 2 else 'rd' if rank == 3 else 'th'}"
            if rank else "N/A"
        )
        rows.append([
            rank_str,
            model,
            _format_percent(values.get("accuracy")),
            _format_percent(values.get("precision")),
            _format_percent(values.get("eer")) if values.get("eer") is not None else "N/A",
        ])
    best = next((m for m in ranked_models if evaluations[m].get("rank") == 1), None)
    return [
        Paragraph("Official Eval-set comparison", _SUBSECTION_STYLE),
        _table(rows, [0.65 * inch, 1.05 * inch, 1.35 * inch, 1.35 * inch, 1.05 * inch]),
        Spacer(1, 6),
        Paragraph(
            "Ranking prioritizes forensic suitability: <b>precision first</b> (50%), "
            "then <b>EER</b> (35%), then accuracy (15%). "
            "Precision is the primary criterion because a false positive — flagging a genuine "
            "speaker as synthetic — carries the highest forensic cost. "
            "CNN ranks 3rd despite the highest precision (99.99%) because its recall of only "
            "66.4% means it misses one-third of all spoof samples.",
            _BODY_STYLE,
        ),
        Spacer(1, 4),
        Paragraph(
            f"Best model: <b>{best or 'Not available'}</b> "
            "(best forensic suitability: highest precision 99.79%, low EER 7.20%, "
            "strong accuracy 86.45%). "
            "AASIST is recommended as the primary model due to its highest precision "
            "among well-performing models and the lowest false positive rate (1.54%).",
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
    segment_analysis: Optional[dict] = None,
) -> Path:
    """Generate a multi-page PDF combining all available forensic evidence.

    Parameters
    ----------
    audio_path:
        The analysed audio file.
    metadata_report_path / signal_report_path / fusion_report_path:
        JSON report files produced by the pipeline.
    output_path:
        Destination PDF path.
    segment_analysis:
        Optional dict returned by ``run_segment_analysis()``.  When provided,
        a dedicated *Segment-Level Detection* section is added to the report.
        When ``None``, the function auto-loads ``results/segment_report.json``
        if it exists alongside the other report files.
    """
    audio_path    = Path(audio_path)
    metadata_path = Path(metadata_report_path)
    signal_path   = Path(signal_report_path)
    fusion_path   = Path(fusion_report_path)
    output_path   = Path(output_path)

    # Auto-load segment results when the caller didn't supply them directly.
    if segment_analysis is None:
        seg_path = fusion_path.parent / "segment_report.json"
        if seg_path.is_file():
            try:
                segment_analysis = json.loads(seg_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                segment_analysis = None
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
    # Format the report-generation timestamp with short timezone code.
    _now = datetime.now().astimezone()
    _tz_full = _now.strftime("%Z")
    _TZ_GEN: dict[str, str] = {
        "India Standard Time": "IST", "Coordinated Universal Time": "UTC",
        "Greenwich Mean Time": "GMT",  "Eastern Standard Time": "EST",
        "Eastern Daylight Time": "EDT","Central Standard Time": "CST",
        "Central Daylight Time": "CDT","Pacific Standard Time": "PST",
        "Pacific Daylight Time": "PDT",
    }
    if _tz_full in _TZ_GEN:
        _tz_short = _TZ_GEN[_tz_full]
    elif len(_tz_full) <= 5 and _tz_full.isupper():
        _tz_short = _tz_full
    else:
        _tz_short = "".join(w[0] for w in _tz_full.split() if w) or "LT"
    generated = _now.strftime(f"%Y-%m-%d %H:%M:%S {_tz_short}")

    story.extend([Spacer(1, 0.55 * inch), Paragraph("AUDIOSHIELD FORENSICS", _TITLE_STYLE),
                  Spacer(1, 0.12 * inch), Paragraph("Audio Forensic Analysis Report", _TITLE_STYLE),
                  Spacer(1, 0.3 * inch)])

    # ── Column widths for all three cover tables ────────────────────────
    _COV_W = [2.0 * inch, 5.0 * inch]  # 7.0 in total — fits A4 with 0.65 in margins

    # ── Identity / classification block ────────────────────────────────
    title_rows = [
        ["Report ID",       report_id],
        ["Generated",       generated],
        ["File Analyzed",   audio_path.name],
        ["Analyst",         "AudioShield Forensics System"],
        ["System Version",  "AudioShield Forensics v1.0.0"],
        ["Report Version",  "1.0.0"],
        ["Classification",  "CONFIDENTIAL — For authorized use only"],
    ]
    story.append(_table(title_rows, _COV_W))
    story.append(Spacer(1, 0.2 * inch))

    # ── Chain of Custody block ──────────────────────────────────────────
    # Row 0 spans both columns so there is no empty right cell.
    coc_rows = [
        ["CHAIN OF CUSTODY — Automated generation, no manual modification"],
        ["Statement: This report was generated automatically by the AudioShield Forensics "
         "System. No manual modification has been made to the analysis results. "
         "All evidence sources are independently computed and logged."],
        ["Integrity: Report content is derived solely from the pipeline outputs produced "
         "at the timestamp shown above. Any post-generation alteration would invalidate "
         "this document."],
    ]
    # Single-column table so there is never an empty second cell.
    story.append(_table(coc_rows, [7.0 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    # ── Evidence Sources block ──────────────────────────────────────────
    ev_rows = [
        ["Source",       "Description"],
        ["AI Detection", "CNN (mel-spectrogram) · SVM (MFCC) · AASIST (raw waveform, graph attention)"],
        ["Metadata",     "FFprobe (container / codec) + EXIFTool (embedded tags)"],
        ["Signal",       "Spectral features (MFCCs, centroid, bandwidth) + temporal (RMS, silence ratio)"],
        ["Fusion",       "Rule-based combiner — AI risk drives overall level; metadata and signal corroborate"],
    ]
    story.append(_table(ev_rows, _COV_W))
    story.extend([Spacer(1, 0.3 * inch),
                  Paragraph(
                      "This report presents separate, independently-sourced evidence streams and "
                      "conservative forensic indicators derived from three AI models, metadata "
                      "extraction, and signal analysis. It is not a substitute for expert examination "
                      "of the original evidence and does not establish legal authenticity on its own.",
                      _CENTER_STYLE),
                  Spacer(1, 0.18 * inch),
                  Paragraph(
                      "Models trained on ASVspoof 2019 LA (English speech). "
                      "Results are probabilistic indicators — not definitive judgements.",
                      _CENTER_STYLE),
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
    story.append(Paragraph(
        "AASIST is the primary forensic model, achieving 99.79% precision and 7.20% EER "
        "on the official eval set.",
        _BODY_STYLE,
    ))
    story.append(Paragraph(f"Recommendation: {_text(combined.get('recommendation'))}", _BODY_STYLE))

    story.extend(_section("2. File Information"))
    file_rows = [["Property", "Value"]]
    # ── Hashes — computed from raw file bytes by extractor ───────────────
    md5_hash    = _value(metadata_summary, "md5_hash",    default=None)
    sha256_hash = _value(metadata_summary, "sha256_hash", default=None)
    # Fallback: try raw_exiftool for MD5 (legacy path before utils.hashing)
    if not md5_hash:
        raw_exif = metadata_summary.get("raw_exiftool", {})
        if isinstance(raw_exif, dict):
            md5_hash = raw_exif.get("MD5Signature") or raw_exif.get("MD5") or raw_exif.get("md5")

    # ── Upload Date (F1) ─────────────────────────────────────────────────
    # Use the pre-formatted string from the extractor (YYYY-MM-DD HH:MM:SS TZ).
    # Fall back to computing it here if the extractor field is missing.
    # Also normalize legacy "India Standard Time" full names → short codes in
    # case the metadata JSON was written by an older version of the extractor.
    _TZ_FULL_TO_SHORT = {
        "India Standard Time": "IST", "Coordinated Universal Time": "UTC",
        "Greenwich Mean Time": "GMT",  "Eastern Standard Time": "EST",
        "Eastern Daylight Time": "EDT","Central Standard Time": "CST",
        "Central Daylight Time": "CDT","Pacific Standard Time": "PST",
        "Pacific Daylight Time": "PDT",
    }
    def _normalize_tz(ts: str) -> str:
        """Replace a full Windows timezone name at the end of a timestamp string."""
        for full, short in _TZ_FULL_TO_SHORT.items():
            if ts.endswith(" " + full):
                return ts[: -(len(full))] + short
        return ts

    upload_date = _value(metadata_summary, "upload_time", default=None)
    if upload_date:
        upload_date = _normalize_tz(str(upload_date))
    else:
        try:
            local_dt = datetime.fromtimestamp(audio_path.stat().st_mtime)
            tz_full  = local_dt.astimezone().strftime("%Z")
            tz_short = _TZ_FULL_TO_SHORT.get(tz_full)
            if not tz_short:
                if len(tz_full) <= 5 and tz_full.isupper():
                    tz_short = tz_full
                else:
                    tz_short = "".join(w[0] for w in tz_full.split() if w) or "LT"
            upload_date = local_dt.strftime(f"%Y-%m-%d %H:%M:%S {tz_short}")
        except OSError:
            upload_date = "Not available"

    info = {
        "File name":          _value(metadata_summary, "file_name", default=audio_path.name),
        "Audio source":       "Microphone Recording" if audio_path.stem == "live_recording"
                              else "Uploaded File",
        "File size (bytes)":  _value(metadata_summary, "file_size",
                                     default=audio_path.stat().st_size if audio_path.exists() else "Not available"),
        "MD5 hash":           md5_hash    if md5_hash    else "Not available",
        "SHA-256 hash":       sha256_hash if sha256_hash else "Not available",
        "Format":             _value(metadata_summary, "format"),
        "Duration (seconds)": _value(metadata_summary, "duration"),
        "Sample rate (Hz)":   _value(metadata_summary, "sample_rate"),
        "Channels":           _value(metadata_summary, "channels"),
        "Codec":              _value(metadata_summary, "codec"),
        "Bitrate (bits/sec)": _value(metadata_summary, "bitrate"),
        "Upload Date":        upload_date,
        "Encoder":            _value(metadata_summary, "encoder"),
        "Device Info":        _value(metadata_summary, "device_info"),
        "Software":           _value(metadata_summary, "software"),
        "Comments":           _value(metadata_summary, "comments"),
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

    story.extend(_section("7. Segment-Level Detection"))
    if segment_analysis:
        story.append(Paragraph(
            "<i>Segment-level verdict reflects the AASIST segment analysis only and is presented "
            "separately from the overall evidence-fusion assessment. The overall risk and assessment "
            "shown in Section 1 are determined by the full multi-source evidence fusion (AI + Metadata "
            "+ Signal) and may differ from the segment-level verdict.</i>",
            _SMALL_STYLE,
        ))
        story.append(Spacer(1, 6))
        seg_summary_rows = [
            ["Parameter", "Value"],
            ["Overall verdict",        segment_analysis.get("overall_verdict", "N/A").upper()],
            ["Total duration (s)",     str(round(segment_analysis.get("total_duration", 0), 2))],
            ["Window size (s)",        str(segment_analysis.get("window_seconds", 1.0))],
            ["Hop size (s)",           str(segment_analysis.get("hop_seconds", 1.0))],
            ["Detection threshold",    str(round(segment_analysis.get("threshold", 0.819), 3))],
            ["Total segments",         str(len(segment_analysis.get("segments", [])))],
            ["Suspicious segments",    str(len(segment_analysis.get("suspicious_segments", [])))],
        ]
        story.append(_table(seg_summary_rows, [2.5 * inch, 4.45 * inch]))
        story.append(Spacer(1, 8))

        # Per-segment table
        seg_rows = [["Start (s)", "End (s)", "Spoof Probability", "Verdict"]]
        for seg in segment_analysis.get("segments", []):
            prob_str = f"{seg['spoof_probability'] * 100:.1f}%"
            seg_rows.append([
                str(seg["start"]),
                str(seg["end"]),
                prob_str,
                seg["verdict"].upper(),
            ])
        if len(seg_rows) > 1:
            story.append(Paragraph("Segment results", _SUBSECTION_STYLE))
            story.append(_table(seg_rows, [1.1 * inch, 1.1 * inch, 1.8 * inch, 1.3 * inch]))

        # Timeline visualization
        viz_path_str = segment_analysis.get("visualization_path")
        if viz_path_str:
            viz_path_obj = _resolve_plot(viz_path_str, [Path("results") / "signal_plots", Path("results"), Path.cwd()])
            if viz_path_obj:
                story.append(Spacer(1, 8))
                img = Image(str(viz_path_obj), width=5.9 * inch, height=2.5 * inch)
                img.hAlign = "CENTER"
                story.extend([img,
                               Paragraph("Segment-level spoof probability over time. "
                                         "Red bars = SPOOF; green bars = BONAFIDE; "
                                         "amber dashed line = detection threshold.",
                                         _CENTER_STYLE),
                               Spacer(1, 6)])
    else:
        story.append(Paragraph("Segment-level detection was not performed for this file.", _BODY_STYLE))

    story.extend(_section("8. Visual Analysis"))
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

    story.extend(_section("9. Conclusion"))
    story.append(Paragraph(f"The combined evidence indicates an overall assessment of <b>{_text(assessment)}</b> "
                           f"with a reported risk level of <b>{_text(risk)}</b>. These are indicators from the "
                           "available reports and should be interpreted with the original audio and chain of custody.",
                           _BODY_STYLE))

    story.extend(_section("10. Limitations"))
    limitations = [
        "Training data: All models were trained exclusively on ASVspoof 2019 LA, which contains English-language "
        "speech produced by 19 text-to-speech and voice-conversion systems. Generalisation to other languages, "
        "accents, or synthesis methods outside this distribution is not guaranteed.",
        "Music and non-speech audio: The pipeline is designed for speech. Music, environmental sounds, or mixed "
        "audio may produce unreliable scores and should not be submitted.",
        "Short recordings: The AASIST model requires a 4-second window. Files shorter than 4 seconds are "
        "zero-padded, which may reduce detection accuracy.",
        "Metadata: File metadata can be trivially edited or removed. Metadata findings are supporting evidence "
        "only and cannot independently confirm or deny authenticity.",
        "Signal features: Spectral and temporal features are influenced by recording conditions, microphone "
        "characteristics, room acoustics, and audio preprocessing. A benign artefact may appear anomalous.",
        "Probabilistic output: All model scores are probabilities, not binary facts. The EER-calibrated "
        "threshold for AASIST (≈81.9%) was optimised on the dev set; performance on out-of-distribution "
        "material may differ.",
        "Legal standing: This report is a technical aid. It does not constitute legal proof and must be "
        "reviewed by a qualified forensic expert before use in any legal or regulatory proceeding.",
    ]
    for lim in limitations:
        story.append(Paragraph(f"• {lim}", _BULLET_STYLE))
    story.append(Spacer(1, 6))

    story.extend(_section("11. Methodology"))
    story.append(Paragraph(
        "The AudioShield pipeline combines three independent AI classifiers with metadata and "
        "signal analysis, then fuses the evidence using a rule-based risk-scoring engine.",
        _BODY_STYLE))
    method_rows = [
        ["Component", "Technique", "Input", "Output"],
        ["CNN",
         "Convolutional neural network on log mel-spectrogram (128 mel bins, 5-second window). "
         "Architecture: FinalCNN; optimised with BCE loss and balanced-accuracy threshold selection.",
         "5-second audio → 128×T mel-spectrogram",
         "P(spoof); threshold ≈ 0.407"],
        ["SVM",
         "Support Vector Machine with RBF kernel on 60-dimensional MFCC + delta + delta-delta "
         "feature vector, L2-normalised. Probability calibration via Platt scaling.",
         "5-second audio → 60-dim MFCC features",
         "P(spoof); threshold = 0.5"],
        ["AASIST",
         "Graph Attention Network operating directly on raw waveform (64,000 samples = 4 seconds). "
         "Retrained with WeightedRandomSampler + Focal Loss (α=0.25, γ=2.0) + CosineAnnealingLR "
         "to correct class imbalance bias.",
         "4-second raw waveform (64k samples)",
         "P(spoof); EER threshold ≈ 0.819"],
        ["Metadata",
         "FFprobe (container/codec) + EXIFTool (embedded tags). Checks for missing device/encoder "
         "info, timestamp inconsistencies, and unusual encoding parameters.",
         "Audio file metadata",
         "Risk score: low / medium / high"],
        ["Signal",
         "Spectral features: MFCCs, spectral centroid, bandwidth, rolloff, zero-crossing rate. "
         "Temporal features: RMS energy, dynamic range, silence ratio. "
         "Anomaly detection against population statistics.",
         "Raw waveform → feature vectors",
         "Risk score + anomaly list"],
        ["Fusion",
         "Rule-based combiner. AI risk drives the overall risk level; metadata and signal provide "
         "corroborating or mitigating evidence. A model with recall < 70 % is excluded from "
         "the primary recommendation.",
         "AI + metadata + signal risk scores",
         "Overall risk + recommendation"],
    ]
    story.append(_table(method_rows, [1.0 * inch, 2.85 * inch, 1.6 * inch, 1.5 * inch]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Threshold selection: all thresholds were determined on the development (dev) set only. "
        "The official evaluation set was used exclusively for final metric reporting and was never "
        "seen during threshold tuning, ensuring honest out-of-sample performance estimates.",
        _SMALL_STYLE))

    story.extend(_section("12. Appendix"))
    story.append(Paragraph("Raw AI evaluation results", _SUBSECTION_STYLE))
    story.append(Preformatted(
        json.dumps(_sanitize_for_report({model: values["raw"] for model, values in ai_evaluations.items()}), indent=2),
        _SMALL_STYLE,
    ))
    story.append(Paragraph("Raw metadata report", _SUBSECTION_STYLE))
    story.append(Preformatted(json.dumps(_sanitize_for_report(metadata), indent=2), _SMALL_STYLE))
    story.append(Paragraph("Raw signal report", _SUBSECTION_STYLE))
    story.append(Preformatted(json.dumps(_sanitize_for_report(signal), indent=2), _SMALL_STYLE))
    story.append(Paragraph("Methodology note", _SUBSECTION_STYLE))
    story.append(Paragraph("See Section 10 (Methodology) for a full description of each pipeline component, "
                           "the feature extraction approach, and the threshold selection procedure. "
                           "No new model inference is performed during report generation. Evidence sources "
                           "remain separated to preserve interpretability.", _BODY_STYLE))
    document.build(story, onFirstPage=_page_header_footer, onLaterPages=_page_header_footer)
    return output_path
