"""Helpers for computing balanced class weights."""

from __future__ import annotations

import numpy as np


def compute_balanced_class_weights(
    labels: np.ndarray | list[int],
    num_classes: int,
    power: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Compute inverse-frequency class weights for a 1D label array."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape {labels.shape}")
    if len(labels) == 0:
        raise ValueError("Cannot compute class weights from an empty label array.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = len(labels) / np.maximum(counts, 1.0)
    weights = np.power(weights, float(power))
    if normalize:
        weights = weights / weights.mean()
    return weights.astype(np.float32)


def compute_multitask_balanced_class_weights(
    labels: np.ndarray | list[list[int]],
    num_classes: int,
    power: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Compute inverse-frequency class weights for a 2D multitask label matrix."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels, got shape {labels.shape}")
    if labels.shape[0] == 0:
        raise ValueError("Cannot compute class weights from an empty label matrix.")

    return np.stack(
        [
            compute_balanced_class_weights(
                labels[:, site_idx],
                num_classes=num_classes,
                power=power,
                normalize=normalize,
            )
            for site_idx in range(labels.shape[1])
        ],
        axis=0,
    )
