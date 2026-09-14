# AudioShield Forensics — Web Interface

Professional web interface for the AudioShield audio deepfake detection system.
Built with FastAPI (backend) and a single-file HTML/CSS/JS SPA (frontend).

---

## Quick Start

### Option A — Batch file (Windows)

```bat
run_web.bat
```

### Option B — Command line

```powershell
# Activate virtual environment
venv\Scripts\activate

# Start the server
python -m src.api.run
```

Then open your browser at **http://localhost:8000**

---

## Requirements

Install FastAPI and Uvicorn if they are not already in the environment:

```powershell
pip install fastapi uvicorn[standard] python-multipart
```

All other dependencies (torch, librosa, etc.) are already required by the
existing pipeline.

---

## Usage

1. Open **http://localhost:8000** in your browser.
2. Drag and drop an audio file (WAV, FLAC, MP3, OGG, M4A, AAC) onto the upload
   area, or click **Browse**.
3. Click **🔍 ANALYZE AUDIO**.
4. Watch the six-stage progress bar as the pipeline runs:
   `Upload → AI Models → Metadata → Signal → Fusion → Report`
5. Review results across all cards:
   - **Overall Risk** — colour-coded HIGH / MEDIUM / LOW banner with confidence bar
   - **AI Detection** — CNN, SVM, AASIST verdicts + ensemble consensus
   - **Metadata Analysis** — risk level and findings list
   - **Signal Analysis** — anomalies and signal risk level
   - **Evidence Fusion** — per-source risks, evidence summary, recommendation
   - **Visualizations** — waveform, spectrogram, MFCC, pitch contour plots
6. Click **📄 Download Report** to save the comprehensive PDF report.
7. Click **🔄 Analyze Another** to reset and process a new file.

---

## API Reference

All endpoints are also accessible directly for integration or testing.

### `GET /api/health`

Returns service status.

```json
{ "status": "healthy", "version": "1.0.0" }
```

---

### `POST /api/upload`

Upload an audio file for analysis.

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio file (WAV, FLAC, MP3, OGG, M4A, AAC) |

**Response**

```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "recording.wav",
  "status": "uploaded"
}
```

---

### `POST /api/analyze/{file_id}`

Run the full forensic pipeline on an uploaded file.

**Response**

```json
{
  "file_id": "550e8400-...",
  "ai_results": {
    "cnn":    { "label": "spoof", "confidence": 0.92, "probability": 0.92 },
    "svm":    { "label": "spoof", "confidence": 0.87, "probability": 0.87 },
    "aasist": { "label": "spoof", "confidence": 0.95, "probability": 0.95 },
    "consensus": { "label": "spoof", "confidence": 0.91, "agreement": 1.0 },
    "ai_risk": "high",
    "device": "cpu"
  },
  "metadata_results": { "summary": {}, "analysis": {}, "metadata": {} },
  "signal_results":   { "risk_level": "high", "anomalies": [...] },
  "fusion_results": {
    "ai_evidence":       { "ai_risk": "high", "consensus": "spoof", ... },
    "metadata_evidence": { "risk_score": "medium", "findings": [...] },
    "signal_evidence":   { "signal_risk": "high", "anomalies": [...] },
    "fusion": {
      "risk_level": "high",
      "assessment": "likely_synthetic",
      "recommendation": "...",
      "evidence_summary": [...]
    }
  },
  "report_available": true
}
```

---

### `GET /api/report/{file_id}`

Download the PDF forensic report for the most recent analysis.

Returns `application/pdf` as `audioshield_report_<id_prefix>.pdf`.

---

### `GET /results/signal_plots/{plot}`

Fetch a signal analysis PNG plot.

Valid values for `{plot}`:

| Plot | Description |
|------|-------------|
| `waveform.png` | Time-domain waveform |
| `spectrogram.png` | Mel spectrogram |
| `mfcc.png` | MFCC feature map |
| `pitch_contour.png` | Fundamental frequency contour |

---

## File Structure

```
AudioShield-Forensics/
├── src/
│   └── api/
│       ├── __init__.py       # Package marker
│       ├── main.py           # FastAPI app + all endpoints
│       ├── run.py            # Uvicorn entry-point
│       └── static/
│           └── index.html    # Single-page frontend
├── uploads/                  # Temporary uploaded files (git-ignored)
│   └── .gitkeep
├── results/
│   ├── comprehensive_report.pdf
│   └── signal_plots/
│       ├── waveform.png
│       ├── spectrogram.png
│       ├── mfcc.png
│       └── pitch_contour.png
├── run_web.bat               # One-click Windows launcher
└── README_WEB.md             # This file
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'fastapi'`

```powershell
pip install fastapi uvicorn[standard] python-multipart
```

### `ModuleNotFoundError: No module named 'src'`

Run from the project root (`AudioShield-Forensics\`), not from inside `src\`:

```powershell
# Correct
python -m src.api.run

# Wrong
cd src && python api/run.py
```

### Port 8000 already in use

Edit `src/api/run.py` and change `port=8000` to another port, e.g. `port=8001`,
then open `http://localhost:8001`.

### Plots show "Plot not available"

The signal plots are generated by the pipeline during analysis. They appear in
`results/signal_plots/` after the first successful analysis.

### PDF report not downloading

The report is generated at `results/comprehensive_report.pdf`. If the analysis
fails partway through, the file may not exist. Re-run the analysis on the file.

### Analysis takes a long time

All three models (CNN, SVM, AASIST) run sequentially on CPU by default. On a
machine without a CUDA GPU this is expected. GPU acceleration is used
automatically when available.

---

## Notes

- Uploaded files are stored in `uploads/` and are **not** deleted automatically.
  Clear this folder periodically to free disk space.
- The `results/` directory is shared — running a new analysis overwrites the
  previous report and plots.
- The server runs in `reload=True` mode by default (development). For production
  remove the `reload=True` flag in `src/api/run.py`.

---

*AudioShield Forensics © 2026 — Fr. C. Rodrigues Institute of Technology*
