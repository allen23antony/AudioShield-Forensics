"""Command-line interface for single-file AI inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .predict import predict_all
from .save import save_predictions


def main() -> int:
    """Run all trained AI models and save their predictions."""
    parser = argparse.ArgumentParser(description="Run CNN, SVM, and AASIST inference.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to the audio file.")
    parser.add_argument("--output", type=Path, default=Path("results") / "ai_predictions.json")
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")
    predictions = predict_all(args.audio)
    save_predictions(predictions, args.output)
    print(json.dumps(predictions, indent=2))
    print(f"\nAI predictions saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
