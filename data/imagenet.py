"""ImageNet validation-subset loader for vision baselines."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.folder import default_loader


def default_imagenet_root(repo_root: Path | None = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[1]
    candidate = (
        root
        / "data"
        / "imagenet"
        / "Small-ImageNet-Validation-Dataset-1000-Classes-main"
        / "ILSVRC2012_img_val_subset"
    )
    if candidate.is_dir():
        return candidate
    # Fallback: search
    base = root / "data" / "imagenet"
    matches = list(base.rglob("ILSVRC2012_img_val_subset"))
    if not matches:
        raise FileNotFoundError(
            f"ImageNet val subset not found under {base}. "
            "Run: python scripts/download_datasets.py --imagenet"
        )
    return matches[0]


class ImageNetClassIndexFolder(datasets.DatasetFolder):
    """ImageFolder variant whose labels are numeric folder names (0..999).

    Plain ``ImageFolder`` sorts class names as strings, so folder ``10`` is
    assigned index 2 instead of label 10. That silently destroys accuracy for
    this subset layout.
    """

    def __init__(self, root: str, transform=None):
        super().__init__(
            root,
            loader=default_loader,
            extensions=datasets.folder.IMG_EXTENSIONS,
            transform=transform,
            is_valid_file=None,
        )

    @staticmethod
    def find_classes(directory: str):  # type: ignore[override]
        classes = sorted(
            (p.name for p in Path(directory).iterdir() if p.is_dir()),
            key=lambda name: int(name) if name.isdigit() else name,
        )
        if not classes:
            raise FileNotFoundError(f"No class folders found in {directory}")
        # Map folder name -> ImageNet class index (identity for 0..999 dirs)
        class_to_idx = {
            cls_name: int(cls_name) if cls_name.isdigit() else i
            for i, cls_name in enumerate(classes)
        }
        return classes, class_to_idx


def make_imagenet_loader(
    transform,
    batch_size: int = 64,
    num_workers: int = 2,
    root: Path | None = None,
) -> tuple[DataLoader, ImageNetClassIndexFolder]:
    data_root = Path(root) if root else default_imagenet_root()
    dataset = ImageNetClassIndexFolder(str(data_root), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset
