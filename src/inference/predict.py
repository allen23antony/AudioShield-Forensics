"""Run the trained CNN, SVM, and AASIST models on one audio file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import torch

from ..audio_preprocessing import preprocess_audio
from ..feature_extraction import extract_mel_spectrogram, extract_mfcc_with_deltas
from ..aasist.model import AASISTModel
from ..train_cnn_final import FinalCNN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CNN_PATH = PROJECT_ROOT / "models" / "cnn" / "cnn_final_best.pth"
CNN_NORM_PATH = PROJECT_ROOT / "results" / "cnn_final_norm.json"
SVM_PATH = PROJECT_ROOT / "models" / "svm" / "svm_model.pkl"
SVM_SCALER_PATH = PROJECT_ROOT / "models" / "svm" / "scaler.pkl"
AASIST_PATH = PROJECT_ROOT / "models" / "aasist" / "aasist_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_torch_state(path: Path) -> Any:
    """Load a PyTorch artifact on the selected CPU/GPU device."""
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)


def load_cnn_model() -> Tuple[torch.nn.Module, dict[str, Any]]:
    """Load the trained CNN and its training-time min-max normalization."""
    model = FinalCNN().to(DEVICE)
    state = _load_torch_state(CNN_PATH)
    if not isinstance(state, dict):
        raise ValueError(f"CNN checkpoint must contain a state dictionary: {CNN_PATH}")
    model.load_state_dict(state)
    if not CNN_NORM_PATH.is_file():
        raise FileNotFoundError(f"CNN normalization file not found: {CNN_NORM_PATH}")
    with CNN_NORM_PATH.open("r", encoding="utf-8") as handle:
        normalization = json.load(handle)
    if not isinstance(normalization, dict):
        raise ValueError(f"CNN normalization must be a JSON object: {CNN_NORM_PATH}")
    model.eval()
    return model, normalization


def load_svm_model() -> Tuple[Any, Any]:
    """Load the trained SVM classifier and its fitted feature scaler."""
    if not SVM_PATH.is_file():
        raise FileNotFoundError(f"SVM model file not found: {SVM_PATH}")
    if not SVM_SCALER_PATH.is_file():
        raise FileNotFoundError(f"SVM scaler file not found: {SVM_SCALER_PATH}")
    return joblib.load(SVM_PATH), joblib.load(SVM_SCALER_PATH)


def load_aasist_model() -> torch.nn.Module:
    """Load the trained 4-second AASIST waveform model."""
    model = AASISTModel().to(DEVICE)
    checkpoint = _load_torch_state(AASIST_PATH)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"AASIST checkpoint has no model_state_dict: {AASIST_PATH}")
    model.load_state_dict(state)
    model.eval()
    return model


def _result(label: str, probability: float) -> Dict[str, Any]:
    """Create a normalized model result, where probability is P(spoof)."""
    bounded = float(np.clip(probability, 0.0, 1.0))
    return {
        "label": label,
        "confidence": float(max(bounded, 1.0 - bounded)),
        "probability": bounded,
    }


def _label_from_probability(probability: float) -> str:
    return "spoof" if probability >= 0.5 else "bonafide"


def predict_cnn(audio_path: Path) -> Dict[str, Any]:
    """Predict spoof probability with the trained mel-spectrogram CNN."""
    audio, sample_rate = preprocess_audio(audio_path, duration=5.0)
    mel = extract_mel_spectrogram(audio, sample_rate)
    model, normalization = load_cnn_model()
    train_min = float(normalization.get("train_min", -80.0))
    train_max = float(normalization.get("train_max", 0.0))
    denominator = train_max - train_min + 1e-8
    mel = np.clip((mel - train_min) / denominator, 0.0, 1.0)
    tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(tensor).reshape(-1)[0]).item())
    return _result(_label_from_probability(probability), probability)


def predict_svm(audio_path: Path) -> Dict[str, Any]:
    """Predict spoof probability with the trained MFCC/delta SVM."""
    audio, sample_rate = preprocess_audio(audio_path, duration=5.0)
    features = extract_mfcc_with_deltas(audio, sample_rate).reshape(1, -1)
    model, scaler = load_svm_model()
    scaled = scaler.transform(features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        spoof_index = classes.index(1) if 1 in classes else len(classes) - 1
        probability = float(probabilities[spoof_index])
    else:
        decision = float(np.asarray(model.decision_function(scaled)).reshape(-1)[0])
        probability = float(1.0 / (1.0 + np.exp(-np.clip(decision, -60.0, 60.0))))
    return _result(_label_from_probability(probability), probability)


def predict_aasist(audio_path: Path) -> Dict[str, Any]:
    """Predict spoof probability with AASIST using its 4-second waveform input."""
    audio, _ = preprocess_audio(audio_path, duration=4.0)
    model = load_aasist_model()
    tensor = torch.from_numpy(audio).float().unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(tensor).reshape(-1)[0]).item())
    return _result(_label_from_probability(probability), probability)


def _failed_result(exc: Exception) -> Dict[str, Any]:
    return {"label": "unknown", "confidence": 0.0, "probability": None, "error": str(exc)}


def predict_all(audio_path: Path) -> Dict[str, Any]:
    """Run all models and return model results plus transparent consensus."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    results: Dict[str, Any] = {}
    for name, predictor in (("cnn", predict_cnn), ("svm", predict_svm), ("aasist", predict_aasist)):
        try:
            results[name] = predictor(path)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            results[name] = _failed_result(exc)

    valid = [results[name] for name in ("cnn", "svm", "aasist") if results[name]["label"] in {"spoof", "bonafide"}]
    labels = [item["label"] for item in valid]
    spoof_count = labels.count("spoof")
    bonafide_count = labels.count("bonafide")
    if not labels or spoof_count == bonafide_count:
        consensus_label = "inconclusive" if labels else "unknown"
    else:
        consensus_label = "spoof" if spoof_count > bonafide_count else "bonafide"
    agreement = (max(spoof_count, bonafide_count) / len(labels)) if labels else 0.0
    confidence = (sum(float(item["confidence"]) for item in valid) / len(valid)) if valid else 0.0
    ai_risk = "high" if consensus_label == "spoof" and agreement >= 2 / 3 else (
        "low" if consensus_label == "bonafide" and agreement >= 2 / 3 else "medium"
    )
    return {
        **results,
        "consensus": {"label": consensus_label, "confidence": float(confidence), "agreement": float(agreement)},
        "ai_risk": ai_risk,
        "device": str(DEVICE),
    }
