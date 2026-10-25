"""Evaluate a trained model on the official MVTec test split.

Writes reports/<model>/<category>/metrics.json and a failure-case gallery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import load_config, resolve_device, set_seed
from src.data import MVTecDataset, index_category
from src.metrics import image_metrics, pixel_auroc
from src.models.feature_extractor import PatchFeatureExtractor
from src.models.global_knn import GlobalKNNScorer
from src.models.patchcore import PatchCoreScorer
from src.train import extract, model_dir, parse_overrides
from src.visualize import overlay_heatmap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--n-failures", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    device = resolve_device(cfg["runtime"]["device"])
    d, m = cfg["data"], cfg["model"]

    test = index_category(d["root"], d["category"])["test"]
    ds = MVTecDataset(test, d["image_size"], d["crop_size"])
    loader = DataLoader(ds, batch_size=cfg["runtime"]["batch_size"])
    labels = np.array([s.label for s in test])
    mdir = model_dir(cfg)
    meta = json.loads((mdir / "meta.json").read_text())
    extractor = PatchFeatureExtractor(meta["backbone"], tuple(meta["layers"])).to(device)

    report: dict = {"category": d["category"], "model": m["name"], "n_test": len(test),
                    "n_test_anomalous": int(labels.sum())}
    maps = None
    if m["name"] == "patchcore":
        scorer, _ = PatchCoreScorer.load(mdir, device)
        scores, maps = scorer.score(extract(extractor, loader, device), d["crop_size"])
        masks = np.stack([ds[i][2] for i in range(len(ds))])
        report["pixel_auroc"] = pixel_auroc(masks, maps)
    else:
        scorer = GlobalKNNScorer(meta["knn_k"])
        scorer.bank = np.load(mdir / "bank.npy")
        scores = scorer.score(extract(extractor, loader, device, "global"))
    report.update(image_metrics(labels, scores, meta["threshold"]))

    # Per-defect-type recall shows where the model struggles.
    preds = scores >= meta["threshold"]
    report["recall_by_defect"] = {
        dt: float(preds[[s.defect_type == dt for s in test]].mean())
        for dt in sorted({s.defect_type for s in test})
    }

    out = Path(cfg["output"]["report_dir"]) / m["name"] / d["category"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    # Failure cases: highest-scoring normals (false alarms) and lowest-scoring defects (misses).
    if maps is not None:
        fdir = out / "failures"
        fdir.mkdir(exist_ok=True)
        normals = np.where(labels == 0)[0]
        defects = np.where(labels == 1)[0]
        worst_fp = normals[np.argsort(-scores[normals])][: args.n_failures // 2]
        worst_fn = defects[np.argsort(scores[defects])][: args.n_failures // 2]
        for tag, idxs in (("false_alarm", worst_fp), ("missed", worst_fn)):
            for i in idxs:
                img = Image.open(test[i].path).convert("RGB")
                ov = overlay_heatmap(img, maps[i], d["image_size"], d["crop_size"])
                ov.save(fdir / f"{tag}_{test[i].defect_type}_{test[i].path.stem}_{scores[i]:.2f}.png")

    print(json.dumps({k: v for k, v in report.items() if k != "recall_by_defect"}, indent=2))


if __name__ == "__main__":
    main()
