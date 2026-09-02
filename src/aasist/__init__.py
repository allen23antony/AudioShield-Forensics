"""AASIST experiment package for ASVspoof 2019 LA."""

from .config import EXPERIMENT_CONFIG
from .model import AASISTModel

__all__ = ["AASISTModel", "EXPERIMENT_CONFIG"]
