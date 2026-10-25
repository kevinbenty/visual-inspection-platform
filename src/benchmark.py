"""Measure end-to-end inference latency, throughput and memory for the saved model.

Reports hardware details so results can be reproduced.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

from app.inference import Inspector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    insp = Inspector(args.model_dir, device=args.device)
    img = Image.open(args.image).convert("RGB")
    for _ in range(args.warmup):
        insp.predict(img)
    if args.device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        insp.predict(img, with_overlay=False)
        if args.device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    t = np.array(times)
    bank = Path(args.model_dir) / "memory_bank.npy"
    result = {
        "latency_ms": {"mean": float(t.mean()), "p50": float(np.percentile(t, 50)),
                       "p95": float(np.percentile(t, 95))},
        "throughput_img_per_s": float(1000 / t.mean()),
        "process_rss_mb": round(proc.memory_info().rss / 2**20, 1),
        "model_load_rss_delta_mb": round((proc.memory_info().rss - rss_before) / 2**20, 1),
        "memory_bank_mb": round(bank.stat().st_size / 2**20, 1) if bank.exists() else None,
        "hardware": {"device": args.device, "cpu": platform.processor() or platform.machine(),
                     "cpu_threads": torch.get_num_threads(), "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None},
        "method": f"{args.warmup} warm-up + {args.runs} timed single-image runs, "
                  "preprocessing + backbone + kNN, overlay excluded",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
