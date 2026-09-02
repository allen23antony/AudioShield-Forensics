from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "LA"
PROTOCOL_DIR = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
TRAIN_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"
EVAL_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_eval" / "flac"
MODEL_DIR = PROJECT_ROOT / "models" / "aasist"
RESULTS_DIR = PROJECT_ROOT / "results" / "aasist"

SEED = 42
SAMPLE_RATE = 16000
INPUT_DURATION_SECONDS = 4.0
INPUT_SAMPLES = int(SAMPLE_RATE * INPUT_DURATION_SECONDS)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

LABEL_MAP = {"bonafide": 0, "spoof": 1}
INVERSE_LABEL_MAP = {0: "bonafide", 1: "spoof"}

TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"

EXPECTED_SPLIT_COUNTS = {
    "train": {"total": 25380, "bonafide": 2580, "spoof": 22800},
    "dev": {"total": 24844, "bonafide": 2548, "spoof": 22296},
    "eval": {"total": 71237, "bonafide": 7355, "spoof": 63882},
}

ATTACK_IDS = [f"A{i:02d}" for i in range(7, 20)]


@dataclass(frozen=True)
class ExperimentConfig:
    """Central configuration for the AASIST experiment.

    Source basis: the faithful AASIST architecture is a raw-waveform, spectro-temporal
    graph attention model. The exact official implementation uses a waveform front-end,
    spectrogram tokens, and graph-attention interaction over time-frequency nodes. This
    project keeps the same conceptual structure while remaining implementable in the
    current environment and preserving the scientific evaluation protocol.
    """

    seed: int = SEED
    sample_rate: int = SAMPLE_RATE
    input_duration_seconds: float = INPUT_DURATION_SECONDS
    input_samples: int = INPUT_SAMPLES
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    num_classes: int = 1
    label_map: dict[str, int] = field(default_factory=lambda: LABEL_MAP.copy())


EXPERIMENT_CONFIG = ExperimentConfig()
