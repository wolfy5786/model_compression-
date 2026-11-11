"""Shared measurement helpers for FP32 baselines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def save_checkpoint(model: nn.Module, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return checkpoint_size_mb(path)


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    desc: str = "eval",
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    for bi, (x, y) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    acc = (correct / total) if total else float("nan")
    return {"accuracy": acc, "correct": correct, "total": total}


@torch.no_grad()
def measure_latency_ms(
    model: nn.Module,
    example: torch.Tensor,
    device: torch.device,
    *,
    warmup: int = 20,
    iters: int = 100,
) -> dict[str, float]:
    model.eval()
    example = example.to(device)
    # Warmup
    for _ in range(warmup):
        _ = model(example)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(example)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = sorted(times)
    mid = len(arr) // 2
    median = arr[mid] if len(arr) % 2 else 0.5 * (arr[mid - 1] + arr[mid])
    return {
        "latency_ms_mean": float(sum(times) / len(times)),
        "latency_ms_median": float(median),
        "latency_ms_p90": float(arr[int(0.9 * (len(arr) - 1))]),
        "latency_batch_size": int(example.shape[0]),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
