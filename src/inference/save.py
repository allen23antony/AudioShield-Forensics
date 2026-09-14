"""Serialization helpers for AI inference results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def save_predictions(predictions: Dict[str, Any], output_path: Path) -> None:
    """Save predictions in the top-level format consumed by fusion."""
    payload = dict(predictions)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    payload.setdefault(
        "model_versions",
        {"cnn": "cnn_final_best", "svm": "svm_model", "aasist": "aasist_best"},
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_predictions(path: Path) -> Dict[str, Any]:
    """Load a saved AI prediction object."""
    prediction_path = Path(path)
    with prediction_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"AI predictions must contain a JSON object: {prediction_path}")
    return payload
