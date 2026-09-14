"""AudioShield Forensics – FastAPI web application.

Serves the single-page frontend and exposes a REST API that wraps the
existing forensic pipeline without modifying any existing module.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..inference.pipeline import run_full_pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"
STATIC_DIR = Path(__file__).parent / "static"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AudioShield Forensics API",
    version="1.0.0",
    description="Audio deepfake detection API backed by CNN, SVM, and AASIST models.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount signal plots directory so the frontend can fetch PNG images.
if (RESULTS_DIR / "signal_plots").exists():
    app.mount(
        "/results/signal_plots",
        StaticFiles(directory=str(RESULTS_DIR / "signal_plots")),
        name="signal_plots",
    )

# Mount static frontend files.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def _assert_file_exists(file_id: str) -> Path:
    """Return the uploaded file path or raise 404 if it does not exist."""
    for ext in ALLOWED_EXTENSIONS:
        candidate = UPLOADS_DIR / f"{file_id}{ext}"
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail=f"File '{file_id}' not found. Upload it first.")


def _safe_get(data: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts without raising KeyError."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
    return current


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the single-page frontend application."""
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index), media_type="text/html")


@app.get("/api/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Return service health status and version."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/upload", tags=["Analysis"])
async def upload_audio(file: UploadFile) -> dict[str, str]:
    """Accept an audio file upload and persist it with a unique ID.

    Returns:
        file_id: UUID string used to reference this upload in later calls.
        filename: Original filename provided by the client.
        status: Always ``"uploaded"`` on success.
    """
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_id = str(uuid.uuid4())
    dest = UPLOADS_DIR / f"{file_id}{suffix}"

    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc
    finally:
        await file.close()

    return {"file_id": file_id, "filename": file.filename or dest.name, "status": "uploaded"}


@app.post("/api/analyze/{file_id}", tags=["Analysis"])
async def analyze_audio(file_id: str) -> dict[str, Any]:
    """Run the full forensic pipeline on a previously uploaded file.

    The pipeline runs CNN, SVM, and AASIST inference, metadata extraction,
    signal analysis, evidence fusion, and PDF report generation.

    Returns:
        file_id: Echo of the requested file_id.
        ai_results: Per-model predictions plus consensus.
        metadata_results: Metadata risk and findings.
        signal_results: Signal anomalies and risk level.
        fusion_results: Overall assessment and recommendation.
        report_available: Whether the PDF report was generated successfully.
    """
    audio_path = _assert_file_exists(file_id)

    try:
        pipeline_output = run_full_pipeline(audio_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError, KeyError, ImportError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    ai_raw: dict[str, Any] = pipeline_output.get("ai", {})
    metadata_raw: dict[str, Any] = pipeline_output.get("metadata", {})
    signal_raw: dict[str, Any] = pipeline_output.get("signal", {})
    fusion_raw: dict[str, Any] = pipeline_output.get("fusion", {})
    report_path_str: str = pipeline_output.get("report_path", "")

    # Flatten AI results into a stable response shape.
    ai_results: dict[str, Any] = {
        "cnn": ai_raw.get("cnn", {}),
        "svm": ai_raw.get("svm", {}),
        "aasist": ai_raw.get("aasist", {}),
        "consensus": ai_raw.get("consensus", {}),
        "ai_risk": ai_raw.get("ai_risk", "unknown"),
        "device": ai_raw.get("device", "cpu"),
    }

    # Metadata results.
    metadata_results: dict[str, Any] = {
        "summary": metadata_raw.get("summary", {}),
        "analysis": metadata_raw.get("analysis", {}),
        "metadata": metadata_raw.get("metadata", {}),
    }

    # Signal results.
    signal_results: dict[str, Any] = dict(signal_raw)

    # Fusion results.
    fusion_results: dict[str, Any] = dict(fusion_raw)

    report_available = bool(report_path_str and Path(report_path_str).is_file())

    return {
        "file_id": file_id,
        "ai_results": ai_results,
        "metadata_results": metadata_results,
        "signal_results": signal_results,
        "fusion_results": fusion_results,
        "report_available": report_available,
    }


@app.get("/api/report/{file_id}", tags=["Reports"])
async def download_report(file_id: str) -> FileResponse:
    """Return the PDF forensic report for a previously analysed file.

    The report is the shared ``results/comprehensive_report.pdf`` generated
    by the last pipeline run (one report per server instance at a time).
    """
    # Confirm the upload exists so we return 404 for unknown IDs.
    _assert_file_exists(file_id)

    report_path = RESULTS_DIR / "comprehensive_report.pdf"
    if not report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Report not found. Run /api/analyze/{file_id} first.",
        )

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=f"audioshield_report_{file_id[:8]}.pdf",
    )


@app.get("/results/signal_plots/{plot}", tags=["Visualizations"])
async def serve_signal_plot(plot: str) -> FileResponse:
    """Return a signal analysis PNG plot by filename.

    Valid plot names: ``waveform.png``, ``spectrogram.png``,
    ``mfcc.png``, ``pitch_contour.png``.
    """
    allowed_plots = {"waveform.png", "spectrogram.png", "mfcc.png", "pitch_contour.png"}
    if plot not in allowed_plots:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown plot '{plot}'. Allowed: {', '.join(sorted(allowed_plots))}",
        )

    plot_path = RESULTS_DIR / "signal_plots" / plot
    if not plot_path.is_file():
        raise HTTPException(status_code=404, detail=f"Plot '{plot}' not yet generated.")

    return FileResponse(path=str(plot_path), media_type="image/png")
