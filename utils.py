from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/experiment.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: Dict[str, Any]) -> torch.device:
    requested_device = config.get("device", "cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"

    return torch.device(requested_device)


def ensure_dirs(config: Dict[str, Any]) -> None:

    
    paths = config.get("paths", {})

    for key in ["results_dir", "checkpoint_dir"]:
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)
