"""Baseline: kNN on whole-image embeddings. Image-level scores only, no localisation.

Useful as a sanity check: PatchCore should beat it, especially on small defects.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.patchcore import knn_distance


class GlobalKNNScorer:
    def __init__(self, knn_k: int = 3):
        self.knn_k = knn_k
        self.bank: np.ndarray | None = None
        self.threshold: float | None = None

    def fit(self, embeddings: np.ndarray) -> "GlobalKNNScorer":
        self.bank = embeddings.astype(np.float32)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        return knn_distance(embeddings.astype(np.float32), self.bank, self.knn_k)

    def calibrate(self, normal_scores: np.ndarray, percentile: float) -> float:
        self.threshold = float(np.percentile(normal_scores, percentile))
        return self.threshold

    def save(self, directory: str | Path, meta: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "bank.npy", self.bank)
        (directory / "meta.json").write_text(json.dumps(
            {"knn_k": self.knn_k, "threshold": self.threshold, **(meta or {})}, indent=2))
