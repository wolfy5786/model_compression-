"""ImageNet vision baselines: MobileNetV2, EfficientNet-B0, ResNet-50."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torchvision import models

VISION_MODELS = ("mobilenet_v2", "efficientnet_b0", "resnet50")


def load_vision_model(name: str, pretrained: bool = True) -> tuple[nn.Module, Callable]:
    """Load a torchvision classification model and its eval preprocess.

    Returns
    -------
    model, weights.transforms()
    """
    name = name.lower().replace("-", "_")
    if name not in VISION_MODELS:
        raise ValueError(f"Unknown vision model {name!r}; expected one of {VISION_MODELS}")

    weight_enum = {
        "mobilenet_v2": models.MobileNet_V2_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
    }[name]
    weights = weight_enum if pretrained else None
    ctor = {
        "mobilenet_v2": models.mobilenet_v2,
        "efficientnet_b0": models.efficientnet_b0,
        "resnet50": models.resnet50,
    }[name]
    model = ctor(weights=weights)
    model.eval()
    transforms = weight_enum.transforms() if pretrained else models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    return model, transforms


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
