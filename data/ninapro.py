"""NinaPro DB2 windowed dataset with repetition-based splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from .splits import SEED, SPLIT_CONFIG, set_seed


def default_db2_subject_dir(repo_root: Path | None = None, subject: int = 1) -> Path:
    root = repo_root or Path(__file__).resolve().parents[1]
    path = root / "data" / "ninapro" / "db2" / f"DB2_s{subject}"
    if not path.is_dir():
        raise FileNotFoundError(
            f"NinaPro DB2 subject folder missing: {path}. "
            "Run: python scripts/download_datasets.py --ninapro"
        )
    return path


def _load_subject_arrays(subject_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats = sorted(subject_dir.glob("S*_E*_A1.mat"))
    if not mats:
        mats = sorted(subject_dir.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files in {subject_dir}")

    emg_parts, stim_parts, rep_parts = [], [], []
    for mat_path in mats:
        m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        emg = np.asarray(m["emg"], dtype=np.float32)
        # Prefer relabelled stimulus / repetition when present
        stim = np.asarray(m.get("restimulus", m["stimulus"]), dtype=np.int32).reshape(-1)
        rep = np.asarray(m.get("rerepetition", m["repetition"]), dtype=np.int32).reshape(-1)
        n = min(len(emg), len(stim), len(rep))
        emg_parts.append(emg[:n])
        stim_parts.append(stim[:n])
        rep_parts.append(rep[:n])

    return (
        np.concatenate(emg_parts, axis=0),
        np.concatenate(stim_parts, axis=0),
        np.concatenate(rep_parts, axis=0),
    )


class NinaProWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.windows = torch.from_numpy(windows)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = self.windows[idx]
        # Per-window z-score across time for each channel
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        return x, self.labels[idx]


def make_ninapro_db2_loaders(
    subject_dir: Path | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = SEED,
) -> dict:
    set_seed(seed)
    cfg = SPLIT_CONFIG["ninapro_db2"]
    subject_dir = Path(subject_dir) if subject_dir else default_db2_subject_dir()

    window_samples = int(round(cfg["window_ms"] * cfg["sampling_hz"] / 1000.0))
    stride_samples = int(round(cfg["stride_ms"] * cfg["sampling_hz"] / 1000.0))

    emg, stim, rep = _load_subject_arrays(subject_dir)

    # Build train first to freeze label remap, then apply same remap logic by
    # building each split independently but aligning classes via shared union.
    splits = {
        "train": cfg["train_repetitions"],
        "val": cfg["val_repetitions"],
        "test": cfg["test_repetitions"],
    }

    # Collect windows with original labels, then remap using train+val+test union
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, reps in splits.items():
        xs: list[np.ndarray] = []
        ys: list[int] = []
        n = len(stim)
        i = 0
        while i + window_samples <= n:
            stim_win = stim[i : i + window_samples]
            rep_win = rep[i : i + window_samples]
            label = int(stim_win[window_samples // 2])
            r = int(rep_win[window_samples // 2])
            if (
                np.all(stim_win == label)
                and np.all(rep_win == r)
                and label > 0
                and r in reps
            ):
                xs.append(emg[i : i + window_samples].T)
                ys.append(label)
            i += stride_samples
        if xs:
            raw[name] = (np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64))
        else:
            raw[name] = (
                np.zeros((0, emg.shape[1], window_samples), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

    all_labels = np.concatenate([raw[s][1] for s in splits if len(raw[s][1])]).astype(np.int64)
    uniq = np.unique(all_labels)
    remap = {int(v): i for i, v in enumerate(uniq)}

    loaders = {}
    counts = {}
    for name in splits:
        x, y = raw[name]
        if len(y):
            y = np.asarray([remap[int(v)] for v in y], dtype=np.int64)
        ds = NinaProWindowDataset(x, y)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed) if name == "train" else None,
        )
        counts[name] = len(ds)

    return {
        "loaders": loaders,
        "num_classes": len(uniq),
        "in_channels": int(emg.shape[1]),
        "window_samples": window_samples,
        "label_remap": remap,
        "counts": counts,
        "subject_dir": str(subject_dir),
        "config": cfg,
    }
