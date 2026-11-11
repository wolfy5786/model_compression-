"""Run FP32 baselines for README model × dataset pairs.

Pairs
-----
- MobileNetV2      × ImageNet val subset (pretrained, eval only)
- EfficientNet-B0  × ImageNet val subset (pretrained, eval only)
- ResNet-50        × ImageNet val subset (pretrained, eval only)
- TEMPONet         × NinaPro DB2 subject 1 (train + eval)

Usage
-----
  .\\.venv\\Scripts\\python.exe benchmarks/run_baselines.py
  .\\.venv\\Scripts\\python.exe benchmarks/run_baselines.py --models resnet50 temponet
  .\\.venv\\Scripts\\python.exe benchmarks/run_baselines.py --temponet-epochs 5 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.measure import (  # noqa: E402
    count_parameters,
    evaluate_accuracy,
    measure_latency_ms,
    save_checkpoint,
    write_json,
)
from data.imagenet import make_imagenet_loader  # noqa: E402
from data.ninapro import make_ninapro_db2_loaders  # noqa: E402
from data.splits import SEED, SPLIT_CONFIG, set_seed  # noqa: E402
from models.temponet import TEMPONet  # noqa: E402
from models.vision import VISION_MODELS, load_vision_model  # noqa: E402

RESULTS_DIR = REPO / "results"
CKPT_DIR = REPO / "checkpoints"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_vision_baseline(name: str, device: torch.device, batch_size: int) -> dict:
    set_seed(SEED)
    model, transform = load_vision_model(name, pretrained=True)
    model = model.to(device)
    loader, dataset = make_imagenet_loader(transform, batch_size=batch_size)

    ckpt_path = CKPT_DIR / f"{name}_imagenet_fp32.pth"
    size_mb = save_checkpoint(model, ckpt_path)

    metrics = evaluate_accuracy(model, loader, device, desc=f"{name} eval")
    # Latency at batch=1
    example, _ = dataset[0]
    example = example.unsqueeze(0)
    latency = measure_latency_ms(model, example, device)

    result = {
        "model": name,
        "dataset": "ImageNet-1k val subset (5/class)",
        "precision": "FP32",
        "pretrained": True,
        "params": count_parameters(model),
        "size_mb": round(size_mb, 4),
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "total": metrics["total"],
        "checkpoint": str(ckpt_path),
        "split": SPLIT_CONFIG["imagenet"],
        "seed": SEED,
        **latency,
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return result


def train_temponet(
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    ckpt_path: Path,
    resume: bool,
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    history = []

    if resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(blob, dict) and "model" in blob:
            model.load_state_dict(blob["model"])
            if "optimizer" in blob:
                opt.load_state_dict(blob["optimizer"])
            start_epoch = int(blob.get("epoch", 0))
            history = list(blob.get("history", []))
            print(f"Resumed TEMPONet from epoch {start_epoch} @ {ckpt_path}")
        else:
            model.load_state_dict(blob)
            print(f"Loaded TEMPONet weights from {ckpt_path}")

    best_val = max((h.get("val_acc", 0.0) for h in history), default=-1.0)

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(loaders["train"], desc=f"TEMPONet train {epoch+1}/{epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(y.numel())

        train_acc = correct / total if total else float("nan")
        val_metrics = evaluate_accuracy(model, loaders["val"], device, desc="TEMPONet val")
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(total, 1),
            "train_acc": train_acc,
            "val_acc": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch+1}/{epochs}  loss={row['train_loss']:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_metrics['accuracy']:.4f}"
        )

        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch + 1,
            "history": history,
            "best_val": max(best_val, val_metrics["accuracy"]),
            "seed": SEED,
        }
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ckpt_path)
        if val_metrics["accuracy"] >= best_val:
            best_val = val_metrics["accuracy"]
            torch.save(payload, ckpt_path.with_name("temponet_ninapro_db2_fp32_best.pth"))

    return {"history": history, "best_val": best_val, "epochs_done": epochs}


def run_temponet_baseline(
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    resume: bool,
) -> dict:
    set_seed(SEED)
    data = make_ninapro_db2_loaders(batch_size=batch_size)
    model = TEMPONet(
        in_channels=data["in_channels"],
        num_classes=data["num_classes"],
        input_samples=data["window_samples"],
    ).to(device)

    ckpt_path = CKPT_DIR / "temponet_ninapro_db2_fp32.pth"
    train_info = train_temponet(
        model,
        data["loaders"],
        device,
        epochs=epochs,
        lr=lr,
        ckpt_path=ckpt_path,
        resume=resume,
    )

    # Reload best if available
    best_path = CKPT_DIR / "temponet_ninapro_db2_fp32_best.pth"
    load_path = best_path if best_path.exists() else ckpt_path
    blob = torch.load(load_path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)

    size_mb = save_checkpoint(model, CKPT_DIR / "temponet_ninapro_db2_fp32_weights.pth")
    test_metrics = evaluate_accuracy(model, data["loaders"]["test"], device, desc="TEMPONet test")
    example = torch.zeros(1, data["in_channels"], data["window_samples"])
    latency = measure_latency_ms(model, example, device)

    return {
        "model": "temponet",
        "dataset": f"NinaPro DB2 ({Path(data['subject_dir']).name})",
        "precision": "FP32",
        "pretrained": False,
        "params": count_parameters(model),
        "size_mb": round(size_mb, 4),
        "accuracy": test_metrics["accuracy"],
        "correct": test_metrics["correct"],
        "total": test_metrics["total"],
        "val_best_accuracy": train_info["best_val"],
        "checkpoint": str(load_path),
        "split": data["config"],
        "window_counts": data["counts"],
        "num_classes": data["num_classes"],
        "in_channels": data["in_channels"],
        "seed": SEED,
        "epochs": epochs,
        "lr": lr,
        **latency,
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_history": train_info["history"],
    }


def _merge_rows(new_rows: list[dict]) -> list[dict]:
    """Merge new rows into existing baseline_results.json by model name."""
    latest_json = RESULTS_DIR / "baseline_results.json"
    by_model: dict[str, dict] = {}
    if latest_json.exists():
        try:
            existing = json.loads(latest_json.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                for row in existing:
                    if isinstance(row, dict) and "model" in row:
                        by_model[str(row["model"])] = row
        except json.JSONDecodeError:
            pass
    for row in new_rows:
        by_model[str(row["model"])] = row
    order = ["mobilenet_v2", "efficientnet_b0", "resnet50", "temponet"]
    ordered = [by_model[k] for k in order if k in by_model]
    ordered.extend(v for k, v in by_model.items() if k not in order)
    return ordered


def persist_results(rows: list[dict]) -> tuple[Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"baseline_results_{stamp}.json"
    latest_json = RESULTS_DIR / "baseline_results.json"
    csv_path = RESULTS_DIR / "baseline_results.csv"
    summary_path = RESULTS_DIR / "baseline_summary.md"

    rows = _merge_rows(rows)
    write_json(json_path, rows)
    write_json(latest_json, rows)

    fieldnames = [
        "model",
        "dataset",
        "precision",
        "accuracy",
        "params",
        "size_mb",
        "latency_ms_median",
        "latency_ms_mean",
        "total",
        "seed",
        "device",
        "checkpoint",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    lines = [
        "# FP32 Baseline Results",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Seed: `{SEED}`",
        "",
        "## Split details",
        "",
        "### ImageNet (MobileNetV2, EfficientNet-B0, ResNet-50)",
        "",
        f"- Protocol: {SPLIT_CONFIG['imagenet']['protocol']}",
        f"- Eval subset: {SPLIT_CONFIG['imagenet']['subset']}",
        f"- Notes: {SPLIT_CONFIG['imagenet']['notes']}",
        "",
        "### NinaPro DB2 (TEMPONet)",
        "",
        f"- Train reps: {SPLIT_CONFIG['ninapro_db2']['train_repetitions']} "
        f"({SPLIT_CONFIG['ninapro_db2']['train_ratio']:.0%})",
        f"- Val reps: {SPLIT_CONFIG['ninapro_db2']['val_repetitions']} "
        f"({SPLIT_CONFIG['ninapro_db2']['val_ratio']:.0%})",
        f"- Test reps: {SPLIT_CONFIG['ninapro_db2']['test_repetitions']} "
        f"({SPLIT_CONFIG['ninapro_db2']['test_ratio']:.0%})",
        f"- Window: {SPLIT_CONFIG['ninapro_db2']['window_ms']} ms @ "
        f"{SPLIT_CONFIG['ninapro_db2']['sampling_hz']} Hz; "
        f"stride {SPLIT_CONFIG['ninapro_db2']['stride_ms']} ms",
        f"- Notes: {SPLIT_CONFIG['ninapro_db2']['notes']}",
        "",
        "## Metrics",
        "",
        "| Model | Dataset | Accuracy | Params | Size (MB) | Latency median (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | {r['accuracy']:.4f} | "
            f"{r['params']:,} | {r['size_mb']:.3f} | {r['latency_ms_median']:.3f} |"
        )
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return latest_json, csv_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenet_v2", "efficientnet_b0", "resnet50", "temponet"],
        help="Subset of models to run",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temponet-epochs", type=int, default=15)
    parser.add_argument("--temponet-batch-size", type=int, default=128)
    parser.add_argument("--temponet-lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Resume TEMPONet from checkpoint")
    args = parser.parse_args()

    device = _device()
    print(f"Device: {device}")
    rows: list[dict] = []

    for name in args.models:
        key = name.lower().replace("-", "_")
        if key in VISION_MODELS:
            print(f"\n=== Vision baseline: {key} ===")
            rows.append(run_vision_baseline(key, device, args.batch_size))
            # Persist incrementally
            persist_results(rows)
        elif key == "temponet":
            print("\n=== TEMPONet baseline (NinaPro DB2) ===")
            rows.append(
                run_temponet_baseline(
                    device,
                    epochs=args.temponet_epochs,
                    batch_size=args.temponet_batch_size,
                    lr=args.temponet_lr,
                    resume=args.resume,
                )
            )
            persist_results(rows)
        else:
            raise SystemExit(f"Unknown model: {name}")

    json_path, csv_path, summary_path = persist_results(rows)
    print("\nWrote:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
