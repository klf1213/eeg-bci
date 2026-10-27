from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def artifact_dirs(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    paths = {name: root / name for name in ("cache", "checkpoints", "figures", "metrics")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

