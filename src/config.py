"""Configuration loading and reproducibility helpers."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for dotted_key, value in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted_key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
