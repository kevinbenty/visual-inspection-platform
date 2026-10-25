"""Loads a trained PatchCore model and runs single-image inspection."""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.patchcore import PatchCoreScorer
from src.preprocessing import preprocess
from src.visualize import overlay_heatmap


class Inspector:
    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        import torch

        from src.models.feature_extractor import PatchFeatureExtractor

        self._torch = torch
        self.device = device
        self.scorer, self.meta = PatchCoreScorer.load(model_dir, device)
        self.extractor = PatchFeatureExtractor(
            self.meta["backbone"], tuple(self.meta["layers"])).to(device)
        self.model_version = f"{self.meta['model']}-{self.meta['category']}"

    def predict(self, img: Image.Image, with_overlay: bool = True) -> dict:
        t0 = time.perf_counter()
        size, crop = self.meta["image_size"], self.meta["crop_size"]
        x = self._torch.from_numpy(preprocess(img, size, crop))[None].to(self.device)
        fmap = self.extractor(x).cpu().numpy()
        scores, maps = self.scorer.score(fmap, crop)
        score, amap = float(scores[0]), maps[0]
        thr = float(self.scorer.threshold)
        result = {
            "anomaly_score": score,
            "threshold": thr,
            "is_anomalous": score >= thr,
            # Relative score: >1 means above threshold. NOT a calibrated probability.
            "score_ratio": score / thr if thr > 0 else None,
            "category": self.meta["category"],
            "model_version": self.model_version,
        }
        if with_overlay:
            ov = overlay_heatmap(img, amap, size, crop, vmax=max(thr * 1.5, float(amap.max())))
            result["heatmap_png_base64"] = encode_png(ov)
            ys, xs = np.unravel_index(int(np.argmax(amap)), amap.shape)
            result["peak_location"] = {"x": int(xs), "y": int(ys), "frame": f"{crop}x{crop} crop"}
        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result


def encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
