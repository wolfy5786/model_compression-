# FP32 Baseline Results

Generated (UTC): 2026-11-11T21:54:11.278772+00:00
Seed: `42`

## Split details

### ImageNet (MobileNetV2, EfficientNet-B0, ResNet-50)

- Protocol: ILSVRC2012 official train/val; eval on fixed val subset
- Eval subset: 5 images/class from official val (ndb796 Small-ImageNet-Validation-Dataset-1000-Classes)
- Notes: Test split unused. No leakage: subset is drawn only from official val.

### NinaPro DB2 (TEMPONet)

- Train reps: [1, 3, 4] (50%)
- Val reps: [6] (17%)
- Test reps: [2, 5] (33%)
- Window: 150 ms @ 2000 Hz; stride 15 ms
- Notes: Splits by repetition id so temporally adjacent windows cannot cross splits.

## Metrics

| Model | Dataset | Accuracy | Params | Size (MB) | Latency median (ms) |
|---|---|---:|---:|---:|---:|
| mobilenet_v2 | ImageNet-1k val subset (5/class) | 0.7256 | 3,504,872 | 13.609 | 32.182 |
| efficientnet_b0 | ImageNet-1k val subset (5/class) | 0.7790 | 5,288,548 | 20.467 | 52.148 |
| resnet50 | ImageNet-1k val subset (5/class) | 0.7734 | 25,557,032 | 97.800 | 38.919 |
| temponet | NinaPro DB2 (DB2_s1) | 0.3458 | 263,225 | 1.034 | 8.622 |
