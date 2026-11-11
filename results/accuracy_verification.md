# Accuracy verification vs published benchmarks

## Vision (torchvision ImageNet-1K weights)

Published Acc@1/Acc@5 are official **full** ILSVRC2012 validation (50,000 images).
Ours are on a **fixed 5-images/class subset** (5,000 images) of that same val set.
Subset Acc@1 within ~±2.5 pp of the published full-val number is expected binomial noise.

| Model | Weights | Published Acc@1 | Our Acc@1 (subset) | Δ pp | Our Acc@5 | Published Acc@5 | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| mobilenet_v2 | `MobileNet_V2_Weights.IMAGENET1K_V1` | 71.878% | 72.56% | +0.68 | 90.26% | 90.286% | PASS (within expected subset sampling noise) |
| efficientnet_b0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` | 77.692% | 77.90% | +0.21 | 93.46% | 93.532% | PASS (within expected subset sampling noise) |
| resnet50 | `ResNet50_Weights.IMAGENET1K_V1` | 76.130% | 77.34% | +1.21 | 93.38% | 92.862% | PASS (within expected subset sampling noise) |

## TEMPONet

- Paper: Zanghieri et al., IEEE TBioCAS 2020
- Paper dataset: NinaPro DB6 (grasp recognition; inter-session)
- Paper Acc (steady inter-session): 65.2%
- Paper Acc (full inter-session): 49.6%
- Paper Acc (20-session custom): 93.7%
- Our Acc (NinaPro DB2 s1, 49-class test): **34.58%** (chance ≈ 2.04%)
- Comparability: Not directly comparable: paper uses NinaPro DB6 / custom multi-session grasp setups (~few grasp classes, inter-session). Ours is NinaPro DB2 subject 1, 49 movement classes, repetition hold-out.

## Confusion matrix artifacts

Directory: `D:\SanJose\personal_projects\model_compression\results\confusion_matrices`

- **mobilenet_v2**: `mobilenet_v2_confusion_matrix.png`, `mobilenet_v2_confusion_matrix.npz`, `mobilenet_v2_top_confused_pairs.csv` (mean per-class recall=0.726)
- **efficientnet_b0**: `efficientnet_b0_confusion_matrix.png`, `efficientnet_b0_confusion_matrix.npz`, `efficientnet_b0_top_confused_pairs.csv` (mean per-class recall=0.779)
- **resnet50**: `resnet50_confusion_matrix.png`, `resnet50_confusion_matrix.npz`, `resnet50_top_confused_pairs.csv` (mean per-class recall=0.773)
- **temponet**: `temponet_confusion_matrix.png`, `temponet_confusion_matrix.npz`, `temponet_top_confused_pairs.csv` (mean per-class recall=0.330)
