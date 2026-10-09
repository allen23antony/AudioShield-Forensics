# AudioShield Forensics

AudioShield Forensics is a research-oriented audio deepfake detection and forensic analysis application. It combines three independently trained classifiers with file metadata and signal-feature checks, then produces a transparent, rule-based evidence summary and a PDF report. Results are indicators for further review, not definitive proof that audio is authentic or manipulated.

## Key Features

- CNN classification using log-mel spectrograms, MFCC-based SVM classification, and an AASIST-inspired waveform/spectral model.
- Per-model spoof probabilities, majority consensus, and visible model disagreement.
- Audio metadata extraction, MD5 and SHA-256 integrity hashes, and metadata risk findings.
- Signal and temporal feature analysis with waveform, spectrogram, MFCC, pitch, and segment-timeline visualizations.
- Supplementary segment-level spoof scoring using the existing AASIST-inspired checkpoint.
- Rule-based evidence fusion across model, metadata, and signal findings.
- JSON analysis artifacts and a comprehensive PDF forensic report.
- Browser audio upload and microphone recording, with the recording converted to WAV in the browser.

## Technology Stack

- Python 3.10 or later
- FastAPI and Uvicorn for the web application and REST API
- PyTorch for the CNN and AASIST-inspired neural classifiers
- scikit-learn and joblib for the SVM and fitted scaler
- librosa, NumPy, SciPy, SoundFile, and SoXR for audio/features
- pandas for dataset and protocol handling
- Matplotlib and Seaborn for plots
- ReportLab for PDF reports
- A single-page HTML/CSS/JavaScript frontend served by FastAPI

## Project Structure

```text
.
├── data/LA/                     # ASVspoof 2019 LA audio, protocols, and scores
├── models/
│   ├── aasist/                   # AASIST-inspired checkpoints
│   ├── cnn/                      # CNN checkpoint
│   └── svm/                      # SVM model and scaler
├── results/                      # Evaluation artifacts and pipeline JSON/plots
│   ├── aasist/
│   └── signal_plots/
├── src/
│   ├── api/                      # FastAPI app, Uvicorn entry point, web frontend
│   ├── aasist/                   # Model, dataset, training, and evaluation
│   ├── forensics/                # Signal, temporal, visualization, and segment analysis
│   ├── fusion/                   # Rule-based risk and evidence combination
│   ├── inference/                # Model prediction and end-to-end pipeline
│   ├── metadata/                 # Metadata extraction, analysis, and report support
│   ├── report/                   # Comprehensive PDF report generator
│   ├── utils/                    # File hashing helpers
│   ├── audio_preprocessing.py
│   ├── feature_extraction.py
│   ├── train_cnn_final.py
│   ├── train_svm_baseline.py
│   ├── evaluate_cnn_final.py
│   └── evaluate_svm_eval.py
├── uploads/                      # Files uploaded to the web app
├── requirements.txt
├── run_web.bat                   # Windows launcher (expects a venv/ directory)
├── README.md
└── README_WEB.md
```

## Prerequisites and External Tools

- Python 3.10+ and pip.
- FFmpeg on `PATH` for compressed audio decoding. The preprocessing code uses `ffmpeg`; WAV/FLAC files that SoundFile can read can be processed without it.
- FFprobe on `PATH` and ExifTool (`exiftool`/`exiftool.exe`) on `PATH` for the full metadata report. If either metadata tool is unavailable, its extractor records an error and metadata results may be incomplete.
- A modern browser with microphone support. Microphone permission is required; localhost is treated as a secure context by current browsers. Other network hosts generally require HTTPS for microphone access.
- A CUDA-capable PyTorch installation is optional. Inference selects CUDA when PyTorch reports it available and otherwise uses CPU. Install the appropriate PyTorch wheel for your platform if you need GPU acceleration.

FFmpeg, FFprobe, and ExifTool are external executables, not Python packages; do not install them through `requirements.txt`. Confirm the commands are available in a new PowerShell window:

```powershell
ffmpeg -version
ffprobe -version
exiftool -ver
```

## Installation

Run these commands from the repository root in Windows PowerShell:

```powershell
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.10` is unavailable, create the environment with an installed Python 3.10+ executable. PyTorch may need a platform-specific install command from the official PyTorch installation selector for CUDA support; the application itself falls back to CPU.

## Model Checkpoints

Inference expects these files at the following repository-relative paths:

| Component | Required artifact |
| --- | --- |
| CNN | `models/cnn/cnn_final_best.pth` |
| CNN normalization | `results/cnn_final_norm.json` |
| SVM classifier | `models/svm/svm_model.pkl` |
| SVM feature scaler | `models/svm/scaler.pkl` |
| AASIST-inspired model | `models/aasist/aasist_balanced_best.pth` |
| AASIST-inspired decision threshold | `results/aasist/aasist_balanced_threshold.json` |

These paths are defined in `src/inference/predict.py`. The checkpoint files are present in this repository's `models/` tree. Preserve the directory layout and filenames, or update the corresponding paths in the source if you intentionally relocate the artifacts. The analysis and report outputs are written under `results/` relative to the current working directory, so start the app from the repository root.

## Run the Application

With the virtual environment active, run from the repository root:

```powershell
python -m src.api.run
```

The server listens on `http://localhost:8000`. The source entry point binds to `0.0.0.0`, uses port `8000`, and enables Uvicorn reload for development. `run_web.bat` is also provided, but it activates a directory named `venv`; use the PowerShell command above when following the `.venv` installation steps.

## Web Interface and Audio Input

Open [http://localhost:8000](http://localhost:8000), choose or drop an audio file, then select **Analyze Audio**. The interface shows model outputs, metadata and signal findings, evidence fusion, visualizations, and a PDF download when report generation succeeds.

Select **Record Voice**, grant microphone access, and stop the recording when finished. The browser records using its supported MediaRecorder format, decodes the recording with Web Audio, converts it to mono 16 kHz PCM WAV, and then submits it through the same analysis flow. This is client-side conversion; the server does not accept the raw browser recording format through the upload API.

The file picker advertises WAV, FLAC, MP3, OGG, M4A, and AAC. The upload API accepts those plus OPUS and WMA. Audio preprocessing reads formats supported by SoundFile directly and uses FFmpeg for compressed formats and as a fallback. Therefore, compressed-format support depends on FFmpeg being installed and available on `PATH`; an extension being accepted does not guarantee that a particular codec or damaged file can be decoded.

## Architecture and Processing Pipeline

1. The API saves an uploaded file in `uploads/` under a generated ID. Preprocessing converts audio to mono at 16 kHz and pads or truncates model inputs to their configured durations.
2. CNN and SVM predictions use five-second audio windows; AASIST-inspired inference uses a four-second waveform window. Each model returns a spoof probability and label when inference succeeds. The ensemble consensus is a majority vote; ties are inconclusive, and individual model failures are surfaced in the result.
3. Metadata analysis calls FFprobe and ExifTool and computes MD5 and SHA-256 hashes from the file bytes.
4. Signal analysis extracts audio features and heuristic anomalies. Segment analysis is supplementary and scores one-second intervals with a one-second hop using the existing AASIST-inspired model; individual intervals are padded or truncated to the model's four-second input length.
5. The fusion module combines AI, metadata, and signal risk levels with explicit rules. It is not a separately trained model, and no measured fusion-performance result is provided.
6. The pipeline writes JSON/text artifacts, plots, and `results/comprehensive_report.pdf`.

The web application writes shared files under `results/`; a later analysis can overwrite the previous report and plots. Uploaded files remain in `uploads/` and are not automatically deleted.

## Classifiers

- **CNN:** `FinalCNN` classifies 128-bin log-mel spectrograms extracted from five seconds of audio. Its trained weights and normalization parameters are loaded separately.
- **SVM:** an RBF-kernel Support Vector Classifier uses MFCC, delta, and delta-delta features with a fitted `StandardScaler`. The feature extraction helper produces a 39-value vector.
- **AASIST-inspired:** a custom waveform-first network combines convolutional waveform features with convolutional STFT magnitude features, then applies graph-attention blocks and a classifier. It follows selected AASIST concepts but is not claimed to be a complete implementation of the original AASIST architecture.

## Evaluation Results

The following rounded metrics are recorded in the individual evaluation artifacts for 71,237 ASVspoof 2019 LA evaluation samples. Threshold-dependent scores use the stored thresholds; ROC-AUC and EER are score-based metrics.

| Model | Accuracy | Precision | Recall | F1 | EER | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 69.83% | 99.99% | 66.36% | 79.78% | 7.16% | 98.00% |
| SVM | 89.40% | 95.44% | 92.61% | 94.00% | 16.98% | 87.49% |
| AASIST-inspired | 86.45% | 99.79% | 85.07% | 91.84% | 7.20% | 97.69% |

Sources: [CNN evaluation](results/cnn_final_eval_results.json), [SVM evaluation](results/svm_eval_results.json), and [AASIST-inspired evaluation](results/aasist/aasist_eval_results.json). The AASIST-inspired evaluator uses `aasist_best.pth` and its frozen dev-set threshold. The web inference path instead loads `aasist_balanced_best.pth` and the balanced threshold file; the table should not be read as a measured evaluation of that separate web-loaded checkpoint. Development-set thresholds are selected separately from the official evaluation set.

## Metadata, Signal, Segments, and Fusion

Metadata findings include container/codec details, embedded fields when available, file size, and MD5/SHA-256 hashes. These hashes identify the analyzed bytes; they do not establish who created a file or whether it was manipulated.

Signal checks summarize spectral and temporal features and may produce waveform, spectrogram, MFCC, and pitch-contour plots. Segment scoring writes a timeline image and reports per-interval scores. These checks are heuristic and are sensitive to codec, recording device, room acoustics, resampling, and other domain differences.

Evidence fusion preserves the three source risk levels, derives an overall risk and assessment using the rules in `src/fusion/risk.py`, and gives a recommendation. It is rule-based, not a learned ensemble, and its accuracy or calibration has not been measured as a separate system.

## JSON and PDF Reports

The full pipeline writes `results/ai_predictions.json`, `metadata_report.json`, `signal_report.json`, `fusion_report.json`, `fusion_report.txt`, and, when segment analysis succeeds, `segment_report.json`. It generates `results/comprehensive_report.pdf` using ReportLab. The API analysis response also includes structured JSON results. The report and plot paths are shared across requests; the report download route returns the latest generated PDF.

## API

FastAPI's interactive documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs); ReDoc is at [http://localhost:8000/redoc](http://localhost:8000/redoc).

| Method and path | Purpose |
| --- | --- |
| `GET /` | Serve the web interface |
| `GET /api/health` | Return service status and version |
| `POST /api/upload` | Accept an audio file as multipart form data (`file`) and return a `file_id` |
| `POST /api/analyze/{file_id}` | Run inference and the forensic pipeline, returning JSON results |
| `GET /api/report/{file_id}` | Download the latest shared PDF report after analysis |
| `GET /results/signal_plots/{plot}` | Retrieve an allowlisted generated PNG plot |

The plot endpoint allows `waveform.png`, `spectrogram.png`, `mfcc.png`, `pitch_contour.png`, and `segment_timeline.png`. The static frontend is also mounted at `/static`.

## Limitations and Troubleshooting

- A model prediction is probabilistic and must not be treated as definitive proof. Review the audio, provenance, and independent evidence; forensic or legal conclusions require qualified human review.
- The models were developed around ASVspoof 2019 LA speech. Performance on other languages, accents, synthesis systems, recording conditions, music, or non-speech audio is not established. The system is intended for speech.
- Audio shorter than four seconds is padded for AASIST-inspired inference, which may reduce reliability. Microphone recordings can differ from the training data and can cause model disagreement.
- Metadata can be removed or edited. Signal anomalies can be caused by benign recording or processing conditions. Neither is conclusive on its own.
- No separate measured performance is available for the rule-based fusion system. Evaluation metrics above are for the individual classifier artifacts and do not establish fusion performance.
- The development server has reload enabled, allows all CORS origins, has no authentication, and retains uploads. Do not expose it as a production service without appropriate security and deployment changes.
- If imports fail, confirm the project virtual environment is active and install `requirements.txt`. If uploads fail for compressed audio, verify `ffmpeg` is on `PATH`; for incomplete metadata, verify `ffprobe` and `exiftool` are available.
- If the browser cannot record, grant microphone permission and use localhost or an HTTPS origin. If the API reports a missing checkpoint, confirm the required artifact paths above from the repository root.