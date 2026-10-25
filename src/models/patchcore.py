"""PatchCore-style anomaly detector (Roth et al., CVPR 2022), simplified.

Idea: store a memory bank of patch features from NORMAL images only. At test time,
each patch's anomaly score is its distance to the nearest memory-bank feature. The
image score is the maximum patch score; the anomaly map is the upsampled patch grid.

The scoring core works on NumPy arrays so it can be unit-tested without a GPU;
heavy loops use PyTorch when it is installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import torch
except ImportError:  # scoring core still works with NumPy only
    torch = None


def _sq_dists_to_point(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    return ((x - p) ** 2).sum(axis=1)


def greedy_coreset(features: np.ndarray, ratio: float, seed: int = 0,
                   proj_dim: int = 128, device: str = "cpu") -> np.ndarray:
    """Greedy k-center selection. Returns indices of the selected rows.

    A random projection reduces cost; selection quality is close to the
    full-dimensional version (as in the PatchCore paper).
    """
    n = features.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    d = features.shape[1]
    if d > proj_dim:
        proj = rng.normal(size=(d, proj_dim)).astype(np.float32) / np.sqrt(proj_dim)
        x = features @ proj
    else:
        x = features.astype(np.float32)
    start = int(rng.integers(n))

    if torch is not None:
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        min_d = ((xt - xt[start]) ** 2).sum(1)
        selected = [start]
        for _ in range(k - 1):
            nxt = int(torch.argmax(min_d))
            selected.append(nxt)
            min_d = torch.minimum(min_d, ((xt - xt[nxt]) ** 2).sum(1))
        return np.array(selected)

    min_d = _sq_dists_to_point(x, x[start])
    selected = [start]
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        min_d = np.minimum(min_d, _sq_dists_to_point(x, x[nxt]))
    return np.array(selected)


def knn_distance(queries: np.ndarray, bank: np.ndarray, k: int = 1,
                 chunk: int = 4096, device: str = "cpu") -> np.ndarray:
    """Mean Euclidean distance from each query row to its k nearest bank rows."""
    out = np.empty(queries.shape[0], dtype=np.float32)
    if torch is not None:
        b = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32)).to(device)
        for i in range(0, len(queries), chunk):
            q = torch.from_numpy(np.ascontiguousarray(queries[i:i + chunk], dtype=np.float32)).to(device)
            dist = torch.cdist(q, b)
            out[i:i + chunk] = dist.topk(k, largest=False).values.mean(1).cpu().numpy()
        return out
    bank_sq = (bank ** 2).sum(1)
    for i in range(0, len(queries), chunk):
        q = queries[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + bank_sq[None, :] - 2.0 * q @ bank.T
        d2 = np.maximum(d2, 0.0)
        part = np.partition(d2, kth=min(k, d2.shape[1]) - 1, axis=1)[:, :k]
        out[i:i + chunk] = np.sqrt(part).mean(1)
    return out


class PatchCoreScorer:
    """Memory bank + scoring. Feature extraction is done outside this class."""

    def __init__(self, coreset_ratio: float = 0.1, knn_k: int = 1, seed: int = 0,
                 device: str = "cpu"):
        self.coreset_ratio = coreset_ratio
        self.knn_k = knn_k
        self.seed = seed
        self.device = device
        self.memory_bank: np.ndarray | None = None
        self.threshold: float | None = None

    @staticmethod
    def flatten(fmap: np.ndarray) -> np.ndarray:
        """(B, C, h, w) -> (B*h*w, C)."""
        b, c, h, w = fmap.shape
        return fmap.transpose(0, 2, 3, 1).reshape(-1, c)

    def fit(self, fmaps: np.ndarray) -> "PatchCoreScorer":
        patches = self.flatten(fmaps).astype(np.float32)
        idx = greedy_coreset(patches, self.coreset_ratio, self.seed, device=self.device)
        self.memory_bank = patches[idx]
        return self

    def score(self, fmaps: np.ndarray, out_size: int, sigma: float = 4.0
              ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (image_scores [B], anomaly_maps [B, out_size, out_size])."""
        if self.memory_bank is None:
            raise RuntimeError("Model is not fitted.")
        b, _, h, w = fmaps.shape
        d = knn_distance(self.flatten(fmaps), self.memory_bank, self.knn_k,
                         device=self.device).reshape(b, h, w)
        maps = np.stack([_upsample(m, out_size, sigma) for m in d])
        return d.reshape(b, -1).max(1), maps

    def calibrate(self, normal_scores: np.ndarray, percentile: float) -> float:
        """Threshold from held-out NORMAL scores only (no test data, no defects)."""
        self.threshold = float(np.percentile(normal_scores, percentile))
        return self.threshold

    def save(self, directory: str | Path, meta: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "memory_bank.npy", self.memory_bank)
        info = {"coreset_ratio": self.coreset_ratio, "knn_k": self.knn_k,
                "seed": self.seed, "threshold": self.threshold, **(meta or {})}
        (directory / "meta.json").write_text(json.dumps(info, indent=2))

    @classmethod
    def load(cls, directory: str | Path, device: str = "cpu") -> tuple["PatchCoreScorer", dict]:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        obj = cls(meta["coreset_ratio"], meta["knn_k"], meta["seed"], device)
        obj.memory_bank = np.load(directory / "memory_bank.npy")
        obj.threshold = meta["threshold"]
        return obj, meta


def _upsample(patch_map: np.ndarray, size: int, sigma: float) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(patch_map.astype(np.float32), mode="F")
    up = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32)
    return gaussian_filter(up, sigma=sigma) if sigma > 0 else up
