import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
import librosa
from src.aasist.model import AASISTModel
from src.aasist.config import SAMPLE_RATE, INPUT_SAMPLES


def predict(model, audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    if len(audio) > INPUT_SAMPLES:
        audio = audio[:INPUT_SAMPLES]
    else:
        audio = np.pad(audio, (0, INPUT_SAMPLES - len(audio)))
    with torch.no_grad():
        tensor = torch.FloatTensor(audio).unsqueeze(0)
        logit = model(tensor).reshape(-1)[0]
        return torch.sigmoid(logit).item()


def main():
    print("=" * 60)
    print("AASIST VERIFICATION")
    print("=" * 60)

    model = AASISTModel()
    checkpoint = torch.load('models/aasist/aasist_balanced_best.pth', map_location='cpu')
    state = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    print("[PASS] Model loaded")

    print("\nBONAFIDE file: LA_E_5849185.flac")
    prob = predict(model, 'data/LA/ASVspoof2019_LA_eval/flac/LA_E_5849185.flac')
    print(f"  Probability: {prob:.6f}")
    print(f"  Threshold 0.5:   {'spoof' if prob > 0.5 else 'bonafide'}")
    print(f"  Threshold 0.972: {'spoof' if prob > 0.972 else 'bonafide'}")

    print("\nSPOOF file: LA_E_2834763.flac")
    prob = predict(model, 'data/LA/ASVspoof2019_LA_eval/flac/LA_E_2834763.flac')
    print(f"  Probability: {prob:.6f}")
    print(f"  Threshold 0.5:   {'spoof' if prob > 0.5 else 'bonafide'}")
    print(f"  Threshold 0.972: {'spoof' if prob > 0.972 else 'bonafide'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
