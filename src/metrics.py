"""Evaluation metrics. Threshold-free (AUROC) plus thresholded (P/R/F1, confusion)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
    }


def pixel_auroc(masks: np.ndarray, maps: np.ndarray, max_pixels: int = 5_000_000,
                seed: int = 0) -> float:
    """Pixel-level AUROC over all test images (subsampled for memory if huge)."""
    y, s = masks.ravel(), maps.ravel()
    if y.size > max_pixels:
        idx = np.random.default_rng(seed).choice(y.size, max_pixels, replace=False)
        y, s = y[idx], s[idx]
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))
