# Model Compression for Embedded Inference

Research project benchmarking **latency** and **classification quality** (confusion matrices / accuracy) of quantized deep neural networks against their uncompressed baselines, with a focus on deployment under tight compute and memory budgets.

## Motivation

Modern deep neural networks deliver strong accuracy, but full-precision (FP32) models are often too large and too slow for **embedded / edge** hardware. This project studies whether quantization can shrink models enough for low-power devices while keeping accuracy and inference latency close to the uncompressed baseline.

Primary questions:

1. How do **8-bit**, **16-bit**, and **32-bit** quantized models compare to an uncompressed baseline on latency, memory, and per-class accuracy?
2. Can a model designed for **myoelectric prosthetic control** remain usable after aggressive quantization on constrained hardware?
3. Can **RLibm** (correctly rounded math) reduce numerical error that accumulates under reduced precision and help recover accuracy?

## Goals

| Goal | Description |
|------|-------------|
| Compress | Apply post-training / aware quantization at 8-bit, 16-bit, and 32-bit |
| Benchmark | Measure latency, model size / memory, accuracy, and confusion matrices |
| Compare | Quantized variants vs. no-compression (FP32) baseline |
| Deploy mindset | Prefer setups realistic for minimal-compute, minimal-memory embedded targets |
| Accuracy recover | Evaluate RLibm for elementary-function accuracy under low-precision formats |

## Candidate Models

We evaluate **four** pretrained (or publicly established) architectures: three widely used vision / edge CNNs, plus one domain-specific network for myoelectric prosthetics.

| Model | Role | Why it is included |
|-------|------|--------------------|
| **TEMPONet** | Myoelectric / sEMG gesture decoding | Temporal CNN designed for embedded EMG recognition and prosthetic / HMI control; already studied with INT8 on GAP8 (~12.8 ms, ~0.9 mJ per classification in prior work) |
| **MobileNetV2** | Lightweight edge CNN | Standard mobile/edge baseline; small footprint and strong INT8 tooling support |
| **EfficientNet-B0** | Accuracy–efficiency CNN | Widely used compound-scaled network; common in edge quantization studies |
| **ResNet-50** | High-capacity reference | De-facto quantization benchmark; useful as an upper-bound / reference model |

### TEMPONet (myoelectric prosthetics)

[TEMPONet](https://doi.org/10.1109/tbcas.2019.2959160) (*Temporal Embedded Muscular Processing Online Network*) is a temporal convolutional network for **surface EMG (sEMG)** gesture classification. It is purpose-built for wearable / prosthetic control:

- Operates on short raw sEMG windows (on the order of ~150 ms)
- Validated on NinaPro-style EMG benchmarks and multi-session wearable setups
- Prior embedded work shows INT8 quantization cutting memory footprint ~4× with only a small accuracy drop, while meeting real-time budgets

This model anchors the project in a **real embedded control** setting, not only ImageNet-style vision benchmarks.

### Vision / edge baselines

- **MobileNetV2** — inverted residuals + depthwise separable convolutions; typical first choice for MCU / NPU demos.
- **EfficientNet-B0** — compound scaling; often retains accuracy under quantization better than heavier nets at similar cost.
- **ResNet-50** — residual CNN used throughout PTQ / QAT literature; provides a well-understood FP32 → INT8 reference curve.

> Pretrained ImageNet weights are used for MobileNetV2, EfficientNet-B0, and ResNet-50. TEMPONet is trained / loaded from EMG datasets (e.g. NinaPro) following published configurations.

## Quantization Plan

Each model is evaluated under four numeric settings:

| Precision | Meaning in this project | Expected tradeoff |
|-----------|-------------------------|-------------------|
| **No compression (FP32)** | Full-precision baseline | Highest accuracy / largest memory |
| **32-bit quantized / calibrated** | Controlled 32-bit path (e.g. fixed-scale or framework “float32 quantized” pipeline) for fair tooling comparison | Near-baseline accuracy; isolates framework overhead |
| **16-bit (FP16 / BF16)** | Half-precision floating point | ~2× size reduction; usually small accuracy loss |
| **8-bit (INT8)** | Integer weights + activations (PTQ and/or QAT) | ~4× size reduction; largest latency / memory win; highest risk of accuracy drop |

### Methods

1. **Post-Training Quantization (PTQ)** — calibrate on a representative subset; fast to apply to pretrained weights.
2. **Quantization-Aware Training (QAT)** — optional fine-tuning when PTQ accuracy drops too far (especially INT8 on EfficientNet / TEMPONet).
3. **Per-tensor / per-channel** weight quantization as supported by the chosen framework (PyTorch / TensorFlow Lite / ONNX Runtime).

All compressed variants are compared against the **same uncompressed FP32 checkpoint** for that architecture.

## Metrics

### Latency

- End-to-end **inference latency** (ms / sample or ms / batch=1)
- Optional: model load time, warm vs. cold start
- Target platforms: host CPU as reference, then embedded-class device when available

### Classification quality

- Overall accuracy / top-1 (vision) or gesture accuracy (TEMPONet)
- **Confusion matrix** per model × precision
- Per-class precision / recall / F1 where useful (especially fine gestures in EMG)

### Resource cost

- Model size on disk (MB)
- Peak inference memory
- Parameter count and estimated ops (optional)

### Comparison format

For each model, produce a table of the form:

| Precision | Size (MB) | Latency (ms) | Accuracy (%) | Notes |
|-----------|-----------|--------------|--------------|-------|
| FP32 (baseline) | | | | |
| 32-bit | | | | |
| 16-bit | | | | |
| 8-bit | | | | |

Plus side-by-side confusion matrices (baseline vs. best compressed).

## RLibm: Improving Numerical Accuracy

[RLibm](https://people.cs.rutgers.edu/~sn349/rlibm/) (Rutgers Architecture and Programming Languages Lab) builds **correctly rounded elementary functions** (`exp`, `log`, `sin`, …) for multiple floating-point representations, including:

- 32-bit float
- bfloat16
- tensorfloat32
- posit formats

### Why it matters here

Quantization and reduced-precision inference amplify rounding error in:

- Activation functions and their compositions
- Softmax / log-sum-exp style normalizations
- Any custom math-heavy preprocessing (e.g. EMG envelope / spectral features)

Standard `libm` implementations are often **not correctly rounded** for every input. That means two platforms (or two precisions) can disagree on the same elementary function, which shows up as:

- Softmax probability drift
- Unstable decision boundaries between similar classes (visible in confusion matrices)
- Extra accuracy loss beyond what weight quantization alone would cause

### How we plan to use RLibm

1. **Replace or wrap** critical elementary functions in the inference / feature path with RLibm implementations for the active format (FP32, BF16, etc.).
2. **A/B test** the same quantized model with default math vs. RLibm math.
3. Measure delta in:
   - overall accuracy
   - per-class confusion (especially near-confused EMG gestures)
   - latency overhead of correctly rounded calls
4. Prefer RLibm where accuracy recovery outweighs any extra cycles — especially on **16-bit / BF16** paths where RLibm has first-class support.

Relevant resources:

- Project page: https://people.cs.rutgers.edu/~sn349/rlibm/
- Source: https://github.com/rutgers-apl/rlibm
- RLibm polynomials are already influencing production math (e.g. LLVM `logf` / `log2f` / `expf` improvements)

## Experiment Workflow

```text
1. Load pretrained model (vision ImageNet or TEMPONet / EMG)
2. Evaluate FP32 baseline → latency + confusion matrix
3. Quantize to 32-bit / 16-bit / 8-bit (PTQ; QAT if needed)
4. Re-benchmark each precision on the same dataset and hardware
5. Optional: re-run with RLibm-backed math for accuracy recovery
6. Summarize size / latency / accuracy tradeoffs
```

## Repository Layout (planned)

```text
model_compression/
├── README.md
├── models/                 # model loaders & TEMPONet definition
├── quantization/           # PTQ / QAT scripts (8 / 16 / 32 bit)
├── benchmarks/             # latency + confusion-matrix runners
├── rlibm_integration/      # hooks / wrappers for correctly rounded math
├── data/                   # dataset configs (ImageNet subset, NinaPro / EMG)
└── results/                # tables, plots, confusion matrices
```

## Success Criteria

- Quantized models (especially INT8) show **clear reductions** in size and, on suitable hardware, latency vs. FP32.
- Accuracy remains **competitive** with the uncompressed baseline; drops are quantified per class via confusion matrices.
- At least one configuration is credible for **embedded myoelectric** use (TEMPONet + INT8/FP16 under a real-time latency budget).
- RLibm experiments show whether correctly rounded elementary functions **recover** measurable accuracy under reduced precision without unacceptable latency cost.

## References

1. Zanghieri et al., *Robust Real-Time Embedded EMG Recognition Framework Using Temporal Convolutional Networks* (TEMPONet), IEEE TBioCAS, 2020. https://doi.org/10.1109/tbcas.2019.2959160
2. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018.
3. Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
4. He et al., *Deep Residual Learning for Image Recognition* (ResNet), CVPR 2016.
5. Lim, Aanjaneya, Gustafson, Nagarakatte — RLibm project. https://people.cs.rutgers.edu/~sn349/rlibm/
6. NinaPro EMG datasets (for TEMPONet training / evaluation).
