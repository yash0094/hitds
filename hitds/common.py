"""Shared helpers: config loading, paths, seeding, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        p = Path(rel)
        cfg["paths"][key] = str(p if p.is_absolute() else ROOT / p)
    for key in ("raw", "processed", "models", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    return cfg


def use_dataset(cfg: dict, name: str) -> dict:
    """Point processed/models/reports paths at a per-dataset sub-folder."""
    cfg = {**cfg, "paths": dict(cfg["paths"]), "dataset": name}
    for key in ("processed", "models", "reports"):
        p = Path(cfg["paths"][key]) / name
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(p)
    cfg["paths"]["db"] = str(Path(cfg["paths"]["db"]).with_name(f"hitds_{name}.sqlite"))
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_logger(name: str = "hitds") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger
