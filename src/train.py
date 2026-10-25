"""Fit a model on normal training images and calibrate its threshold.

Usage:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --set data.category=screw
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_device, set_seed
from src.data import MVTecDataset, index_category, split_train
from src.models.feature_extractor import PatchFeatureExtractor
from src.models.global_knn import GlobalKNNScorer
from src.models.patchcore import PatchCoreScorer


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def extract(extractor, loader, device, mode: str = "patch") -> np.ndarray:
    chunks = []
    for imgs, *_ in loader:
        x = torch.as_tensor(imgs).to(device)
        f = extractor(x) if mode == "patch" else extractor.global_embedding(x)
        chunks.append(f.cpu().numpy())
    return np.concatenate(chunks)


def model_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["model_dir"]) / cfg["model"]["name"] / cfg["data"]["category"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. data.category=screw")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    device = resolve_device(cfg["runtime"]["device"])
    d, m, rt = cfg["data"], cfg["model"], cfg["runtime"]

    splits = index_category(d["root"], d["category"])
    fit_s, cal_s = split_train(splits["train"], d["calibration_fraction"], cfg["seed"])
    print(f"[{d['category']}] fit={len(fit_s)} calibration={len(cal_s)} test={len(splits['test'])}")

    mk = lambda s: DataLoader(MVTecDataset(s, d["image_size"], d["crop_size"]),
                              batch_size=rt["batch_size"], num_workers=rt["num_workers"])
    layers = m["layers"] if m["name"] == "patchcore" else ["layer1", "layer2", "layer3", "layer4"]
    extractor = PatchFeatureExtractor(m["backbone"], tuple(layers)).to(device)

    t0 = time.perf_counter()
    meta = {"category": d["category"], "backbone": m["backbone"], "layers": layers,
            "image_size": d["image_size"], "crop_size": d["crop_size"], "model": m["name"],
            "n_fit": len(fit_s), "n_calibration": len(cal_s), "seed": cfg["seed"]}
    if m["name"] == "patchcore":
        scorer = PatchCoreScorer(m["coreset_ratio"], m["knn_k"], cfg["seed"], device)
        scorer.fit(extract(extractor, mk(fit_s), device))
        cal_scores, _ = scorer.score(extract(extractor, mk(cal_s), device), d["crop_size"])
        meta["memory_bank_size"] = int(scorer.memory_bank.shape[0])
    elif m["name"] == "global_knn":
        scorer = GlobalKNNScorer(m["knn_k"])
        scorer.fit(extract(extractor, mk(fit_s), device, "global"))
        cal_scores = scorer.score(extract(extractor, mk(cal_s), device, "global"))
    else:
        raise ValueError(f"Unknown model {m['name']}")

    thr = scorer.calibrate(cal_scores, m["threshold_percentile"])
    meta["fit_seconds"] = round(time.perf_counter() - t0, 2)
    out = model_dir(cfg)
    scorer.save(out, meta)
    print(f"threshold={thr:.4f}  saved to {out}  ({meta['fit_seconds']}s)")


if __name__ == "__main__":
    main()
