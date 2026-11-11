"""Model loaders for baseline and compression experiments."""

from .temponet import TEMPONet
from .vision import load_vision_model, VISION_MODELS

__all__ = ["TEMPONet", "load_vision_model", "VISION_MODELS"]
