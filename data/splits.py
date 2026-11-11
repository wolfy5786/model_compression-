"""Deterministic split configuration shared by all baseline runners.

ImageNet (vision models)
------------------------
Official ILSVRC2012 protocol: train (~1.28M) / val (50k) / test (unlabeled for
public leaderboards). Torchvision ImageNet1K weights were trained on the
official train split. This project evaluates on a fixed 5-images-per-class
subset of the official **validation** set (5,000 images, 1,000 classes).
No re-splitting; val images never enter training.

NinaPro DB2 (TEMPONet)
----------------------
DB2 records **6** repetitions per movement. Repetition-based split (no window
from a test/val repetition appears in training):

  train repetitions: 1, 3, 4
  val   repetitions: 6
  test  repetitions: 2, 5

Ratios of the 6 gesture repetitions: train 50% / val ~16.7% / test ~33.3%.
Global seed controls DataLoader shuffling and TEMPONet weight init only.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

SEED = 42

SPLIT_CONFIG: dict[str, Any] = {
    "seed": SEED,
    "imagenet": {
        "protocol": "ILSVRC2012 official train/val; eval on fixed val subset",
        "subset": "5 images/class from official val (ndb796 Small-ImageNet-Validation-Dataset-1000-Classes)",
        "train_ratio": None,  # not retrained here; pretrained on full official train
        "val_images_per_class": 5,
        "num_classes": 1000,
        "eval_split": "val_subset",
        "notes": "Test split unused. No leakage: subset is drawn only from official val.",
    },
    "ninapro_db2": {
        "protocol": "repetition hold-out (DB2 has 6 reps/movement)",
        "train_repetitions": [1, 3, 4],
        "val_repetitions": [6],
        "test_repetitions": [2, 5],
        "train_ratio": 0.5,
        "val_ratio": 1.0 / 6.0,
        "test_ratio": 1.0 / 3.0,
        "window_ms": 150,
        "stride_ms": 15,
        "sampling_hz": 2000,
        "notes": "Splits by repetition id so temporally adjacent windows cannot cross splits.",
    },
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
