"""Download baseline datasets used by this repo."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REPO = Path(__file__).resolve().parents[1]
IMAGENET_URL = (
    "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes/"
    "archive/refs/heads/main.zip"
)
NINAPRO_DB2_S1 = "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s1.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")


def download_imagenet(force: bool = False) -> Path:
    out_root = REPO / "data" / "imagenet"
    subset = out_root / "Small-ImageNet-Validation-Dataset-1000-Classes-main" / "ILSVRC2012_img_val_subset"
    if subset.is_dir() and not force:
        print(f"ImageNet subset already present: {subset}")
        return subset
    zip_path = REPO / "data" / "imagenet_val_subset.zip"
    if force or not zip_path.exists():
        _download(IMAGENET_URL, zip_path)
    out_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_root)
    print(f"Extracted ImageNet subset to {subset}")
    return subset


def download_ninapro_db2(subject: int = 1, force: bool = False) -> Path:
    dest_dir = REPO / "data" / "ninapro" / "db2" / f"DB2_s{subject}"
    if list(dest_dir.glob("*.mat")) and not force:
        print(f"NinaPro DB2 s{subject} already present: {dest_dir}")
        return dest_dir
    url = f"https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{subject}.zip"
    zip_path = REPO / "data" / "ninapro" / "db2" / f"DB2_s{subject}.zip"
    if force or not zip_path.exists():
        _download(url, zip_path)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"Extracted NinaPro DB2 s{subject} to {dest_dir}")
    return dest_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--imagenet", action="store_true")
    p.add_argument("--ninapro", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not (args.imagenet or args.ninapro or args.all):
        args.all = True
    if args.imagenet or args.all:
        download_imagenet(force=args.force)
    if args.ninapro or args.all:
        download_ninapro_db2(force=args.force)


if __name__ == "__main__":
    main()
