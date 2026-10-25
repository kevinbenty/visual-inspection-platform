"""MVTec AD dataset indexing and splits.

Expected layout (as distributed by MVTec):
    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect_type>/*.png        (defect_type 'good' = normal)
    <root>/<category>/ground_truth/<defect_type>/*_mask.png

Split policy (documented in the README / model card):
    * fit:          train/good minus the calibration subset
    * calibration:  a seeded random fraction of train/good, used ONLY to set the
                    decision threshold (no defective images, no test data)
    * test:         the official test split, used only for final reporting
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import preprocess, preprocess_mask

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int               # 0 = normal, 1 = anomalous
    defect_type: str
    mask_path: Path | None


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def index_category(root: str | Path, category: str) -> dict[str, list[Sample]]:
    base = Path(root) / category
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Download MVTec AD and extract it to {root} (see README)."
        )
    train = [Sample(p, 0, "good", None) for p in _images_in(base / "train" / "good")]
    test: list[Sample] = []
    for defect_dir in sorted((base / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        dt = defect_dir.name
        for p in _images_in(defect_dir):
            if dt == "good":
                test.append(Sample(p, 0, dt, None))
            else:
                mask = base / "ground_truth" / dt / f"{p.stem}_mask.png"
                test.append(Sample(p, 1, dt, mask if mask.exists() else None))
    return {"train": train, "test": test}


def split_train(
    train: list[Sample], calibration_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train))
    n_cal = max(1, int(round(len(train) * calibration_fraction)))
    cal = [train[i] for i in sorted(idx[:n_cal])]
    fit = [train[i] for i in sorted(idx[n_cal:])]
    return fit, cal


class MVTecDataset:
    """Minimal map-style dataset, compatible with torch DataLoader."""

    def __init__(self, samples: list[Sample], image_size: int, crop_size: int):
        self.samples = samples
        self.image_size = image_size
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = preprocess(Image.open(s.path).convert("RGB"), self.image_size, self.crop_size)
        if s.mask_path is not None:
            mask = preprocess_mask(Image.open(s.mask_path), self.image_size, self.crop_size)
        else:
            mask = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
        return img, s.label, mask, str(s.path)
