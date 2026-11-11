# Dataset splits (baselines)

Global seed: **42** (`data/splits.py`).

## ImageNet — MobileNetV2 / EfficientNet-B0 / ResNet-50

| Split | Role in this repo |
|-------|-------------------|
| Official ILSVRC2012 train | Used only by torchvision pretraining (not re-run here) |
| Official ILSVRC2012 val | Source of the local eval subset |
| Official test | Unused (labels not public) |

**Local eval set:** fixed 5 images/class subset of official val
(`ILSVRC2012_img_val_subset`, 5,000 images, 1,000 classes).
Folder names are numeric ImageNet class indices; the loader maps them
numerically (not lexicographically).

No re-splitting → no train/test leakage.

## NinaPro DB2 — TEMPONet

Subject: `DB2_s1` (12 EMG channels @ 2 kHz). Windows: 150 ms (300 samples),
stride 15 ms. Rest (`stimulus==0`) excluded. Homogeneous label+repetition
required inside each window.

DB2 has **6** repetitions per movement:

| Split | Repetitions | Share of reps |
|-------|-------------|---------------|
| Train | 1, 3, 4 | 50% |
| Val | 6 | ≈16.7% |
| Test | 2, 5 | ≈33.3% |

Split is by repetition id so no window from a held-out repetition enters
training.
