"""Build confusion matrices + published-benchmark verification for FP32 baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Minimal CM helper (avoids sklearn/scipy DLL policy issues on this host)."""
    n = len(labels)
    index = {int(lab): i for i, lab in enumerate(labels)}
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.imagenet import make_imagenet_loader  # noqa: E402
from data.ninapro import make_ninapro_db2_loaders  # noqa: E402
from data.splits import SEED, set_seed  # noqa: E402
from models.temponet import TEMPONet  # noqa: E402
from models.vision import VISION_MODELS, load_vision_model  # noqa: E402

RESULTS = REPO / "results"
CM_DIR = RESULTS / "confusion_matrices"
CKPT = REPO / "checkpoints"

# Official torchvision ImageNet-1K single-crop Acc@1 for the exact weight enums we use.
# Source: https://docs.pytorch.org/vision/stable/models.html
PUBLISHED_IMAGENET = {
    "mobilenet_v2": {
        "weights": "MobileNet_V2_Weights.IMAGENET1K_V1",
        "acc1_full_val": 0.71878,
        "acc5_full_val": 0.90286,
        "source": "https://docs.pytorch.org/vision/stable/models.html",
    },
    "efficientnet_b0": {
        "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "acc1_full_val": 0.77692,
        "acc5_full_val": 0.93532,
        "source": "https://docs.pytorch.org/vision/stable/models.html",
    },
    "resnet50": {
        "weights": "ResNet50_Weights.IMAGENET1K_V1",
        "acc1_full_val": 0.76130,
        "acc5_full_val": 0.92862,
        "source": "https://docs.pytorch.org/vision/stable/models.html",
    },
}

# TEMPONet paper numbers are NOT the same protocol as our DB2 49-class baseline.
PUBLISHED_TEMPONET = {
    "paper": "Zanghieri et al., IEEE TBioCAS 2020",
    "dataset_in_paper": "NinaPro DB6 (grasp recognition; inter-session)",
    "acc_steady_intersession": 0.652,
    "acc_full_intersession": 0.496,
    "acc_20session_custom": 0.937,
    "source": "https://doi.org/10.1109/tbcas.2019.2959160",
    "comparability": (
        "Not directly comparable: paper uses NinaPro DB6 / custom multi-session "
        "grasp setups (~few grasp classes, inter-session). Ours is NinaPro DB2 "
        "subject 1, 49 movement classes, repetition hold-out."
    ),
}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def collect_preds(model, loader, device, *, topk: int = 5):
    model.eval()
    y_true, y_pred = [], []
    topk_correct = 0
    total = 0
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        if topk and logits.size(1) >= topk:
            topk_pred = logits.topk(topk, dim=1).indices
            topk_correct += int((topk_pred == y.unsqueeze(1)).any(dim=1).sum().item())
        total += int(y.numel())
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    acc1 = float((yt == yp).mean()) if len(yt) else float("nan")
    acc5 = float(topk_correct / total) if total and topk else None
    return yt, yp, acc1, acc5


def save_cm_artifacts(name: str, yt: np.ndarray, yp: np.ndarray, labels: np.ndarray) -> dict:
    CM_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(yt, yp, labels=labels)
    npz_path = CM_DIR / f"{name}_confusion_matrix.npz"
    np.savez_compressed(npz_path, cm=cm, labels=labels, y_true=yt, y_pred=yp)

    # Normalized by true class
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    # Top confused off-diagonal pairs
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append((int(labels[i]), int(labels[j]), int(cm[i, j]), float(cm_norm[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    top_pairs = pairs[:30]
    pairs_path = CM_DIR / f"{name}_top_confused_pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_label", "pred_label", "count", "row_normalized"])
        w.writerows(top_pairs)

    # Plot: full heatmap only if manageable; else downsample / show diagonal summary
    fig_path = CM_DIR / f"{name}_confusion_matrix.png"
    n = cm.shape[0]
    if n <= 60:
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{name} confusion matrix (row-normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        if n <= 50:
            ticks = np.arange(n)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels, fontsize=6, rotation=90)
            ax.set_yticklabels(labels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
    else:
        # 1000-class: plot per-class recall bars + a 50x50 zoom of most-confused block is heavy;
        # instead plot diagonal (per-class recall) and a sparse top-confusion strip.
        recalls = np.diag(cm_norm)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(labels, recalls, lw=0.6)
        axes[0].set_title(f"{name}: per-class recall on eval set (diag of row-normalized CM)")
        axes[0].set_xlabel("Class index")
        axes[0].set_ylabel("Recall")
        axes[0].set_ylim(0, 1.05)
        axes[0].axhline(float(np.mean(recalls)), color="C1", ls="--", label=f"mean={np.mean(recalls):.3f}")
        axes[0].legend()

        # Show top-30 confused as a small matrix of those unique labels
        involved = sorted({p[0] for p in top_pairs[:20]} | {p[1] for p in top_pairs[:20]})
        idx = [int(np.where(labels == lab)[0][0]) for lab in involved]
        sub = cm_norm[np.ix_(idx, idx)]
        im = axes[1].imshow(sub, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Row-normalized submatrix over labels in top confused pairs")
        axes[1].set_xticks(range(len(involved)))
        axes[1].set_yticks(range(len(involved)))
        axes[1].set_xticklabels(involved, fontsize=7, rotation=90)
        axes[1].set_yticklabels(involved, fontsize=7)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)

    return {
        "npz": str(npz_path),
        "png": str(fig_path),
        "top_confused_pairs_csv": str(pairs_path),
        "num_classes": int(n),
        "mean_per_class_recall": float(np.mean(np.diag(cm_norm))),
        "top_confused_pairs": [
            {"true": a, "pred": b, "count": c, "row_norm": r} for a, b, c, r in top_pairs[:10]
        ],
    }


def run_vision(name: str, device: torch.device, batch_size: int) -> dict:
    set_seed(SEED)
    model, transform = load_vision_model(name, pretrained=True)
    model = model.to(device)
    loader, _ = make_imagenet_loader(transform, batch_size=batch_size)
    yt, yp, acc1, acc5 = collect_preds(model, loader, device, topk=5)
    labels = np.arange(1000)
    cm_info = save_cm_artifacts(name, yt, yp, labels)
    pub = PUBLISHED_IMAGENET[name]
    delta = acc1 - pub["acc1_full_val"]
    return {
        "model": name,
        "dataset": "ImageNet-1k val subset (5/class)",
        "our_acc1": acc1,
        "our_acc5": acc5,
        "published_acc1_full_val": pub["acc1_full_val"],
        "published_acc5_full_val": pub["acc5_full_val"],
        "delta_acc1_vs_published": delta,
        "weights": pub["weights"],
        "published_source": pub["source"],
        "verification": (
            "PASS (within expected subset sampling noise)"
            if abs(delta) <= 0.025
            else "CHECK (larger than ~2.5pp vs full val)"
        ),
        "confusion_matrix": cm_info,
        "n_eval": int(len(yt)),
    }


def run_temponet(device: torch.device, batch_size: int) -> dict:
    set_seed(SEED)
    data = make_ninapro_db2_loaders(batch_size=batch_size)
    model = TEMPONet(
        in_channels=data["in_channels"],
        num_classes=data["num_classes"],
        input_samples=data["window_samples"],
    ).to(device)
    best = CKPT / "temponet_ninapro_db2_fp32_best.pth"
    blob = torch.load(best, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    yt, yp, acc1, _ = collect_preds(model, data["loaders"]["test"], device, topk=0)
    labels = np.arange(data["num_classes"])
    cm_info = save_cm_artifacts("temponet", yt, yp, labels)
    return {
        "model": "temponet",
        "dataset": f"NinaPro DB2 ({Path(data['subject_dir']).name}) test split",
        "our_acc1": acc1,
        "published_reference": PUBLISHED_TEMPONET,
        "verification": "NOT DIRECTLY COMPARABLE (different dataset/classes/protocol)",
        "chance_accuracy_49_classes": 1.0 / data["num_classes"],
        "confusion_matrix": cm_info,
        "n_eval": int(len(yt)),
        "window_counts": data["counts"],
    }


def write_summary(rows: list[dict]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / "accuracy_verification.json"
    out_md = RESULTS / "accuracy_verification.md"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Accuracy verification vs published benchmarks",
        "",
        "## Vision (torchvision ImageNet-1K weights)",
        "",
        "Published Acc@1/Acc@5 are official **full** ILSVRC2012 validation (50,000 images).",
        "Ours are on a **fixed 5-images/class subset** (5,000 images) of that same val set.",
        "Subset Acc@1 within ~±2.5 pp of the published full-val number is expected binomial noise.",
        "",
        "| Model | Weights | Published Acc@1 | Our Acc@1 (subset) | Δ pp | Our Acc@5 | Published Acc@5 | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r["model"] == "temponet":
            continue
        lines.append(
            f"| {r['model']} | `{r['weights']}` | {100*r['published_acc1_full_val']:.3f}% | "
            f"{100*r['our_acc1']:.2f}% | {100*r['delta_acc1_vs_published']:+.2f} | "
            f"{100*r['our_acc5']:.2f}% | {100*r['published_acc5_full_val']:.3f}% | {r['verification']} |"
        )

    tempo = next((r for r in rows if r["model"] == "temponet"), None)
    if tempo is not None:
        pub = tempo["published_reference"]
        lines += [
            "",
            "## TEMPONet",
            "",
            f"- Paper: {pub['paper']}",
            f"- Paper dataset: {pub['dataset_in_paper']}",
            f"- Paper Acc (steady inter-session): {100*pub['acc_steady_intersession']:.1f}%",
            f"- Paper Acc (full inter-session): {100*pub['acc_full_intersession']:.1f}%",
            f"- Paper Acc (20-session custom): {100*pub['acc_20session_custom']:.1f}%",
            f"- Our Acc (NinaPro DB2 s1, 49-class test): **{100*tempo['our_acc1']:.2f}%** "
            f"(chance ≈ {100*tempo['chance_accuracy_49_classes']:.2f}%)",
            f"- Comparability: {pub['comparability']}",
        ]
    lines += [
        "",
        "## Confusion matrix artifacts",
        "",
        f"Directory: `{CM_DIR}`",
        "",
    ]
    for r in rows:
        cm = r["confusion_matrix"]
        lines.append(
            f"- **{r['model']}**: `{Path(cm['png']).name}`, `{Path(cm['npz']).name}`, "
            f"`{Path(cm['top_confused_pairs_csv']).name}` "
            f"(mean per-class recall={cm['mean_per_class_recall']:.3f})"
        )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["mobilenet_v2", "efficientnet_b0", "resnet50", "temponet"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--temponet-batch-size", type=int, default=256)
    args = p.parse_args()

    device = _device()
    print(f"Device: {device}")
    rows = []
    for name in args.models:
        key = name.lower().replace("-", "_")
        if key in VISION_MODELS:
            print(f"=== CM + verify: {key} ===")
            rows.append(run_vision(key, device, args.batch_size))
        elif key == "temponet":
            print("=== CM + verify: temponet ===")
            rows.append(run_temponet(device, args.temponet_batch_size))
        else:
            raise SystemExit(f"Unknown model {name}")
        write_summary(rows)

    path = write_summary(rows)
    print(f"Wrote {path}")
    print(f"Confusion matrices under {CM_DIR}")


if __name__ == "__main__":
    main()
