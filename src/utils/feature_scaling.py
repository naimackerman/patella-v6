"""Helpers for deterministic feature standardization."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-dimension mean/std scaling."""
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def transform_with_standardizer(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply pre-fit standardization."""
    features = np.asarray(features, dtype=np.float64)
    return (features - mean) / scale


def save_standardizer(path: str | Path, mean: np.ndarray, scale: np.ndarray) -> None:
    """Persist standardizer statistics as NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), mean=np.asarray(mean, dtype=np.float64), scale=np.asarray(scale, dtype=np.float64))


def load_standardizer(path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load standardizer statistics if they exist."""
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=False)
    return np.asarray(data["mean"], dtype=np.float64), np.asarray(data["scale"], dtype=np.float64)
