"""Heatmap overlays for the API, dashboard and failure-case reports."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.preprocessing import resize_and_crop


def overlay_heatmap(img: Image.Image, anomaly_map: np.ndarray, image_size: int,
                    crop_size: int, vmax: float | None = None, alpha: float = 0.45
                    ) -> Image.Image:
    base = np.asarray(resize_and_crop(img, image_size, crop_size), dtype=np.uint8)
    m = anomaly_map.astype(np.float32)
    lo, hi = float(m.min()), float(vmax if vmax is not None else m.max())
    norm = np.clip((m - lo) / (hi - lo + 1e-8), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray((base * (1 - alpha) + heat * alpha).astype(np.uint8))
