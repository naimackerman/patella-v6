"""Weighted sampling utilities for class-imbalanced datasets."""

from typing import Union

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from src.utils.class_weights import compute_multitask_balanced_class_weights


def _normalize_positive_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize positive weights to mean 1.0."""
    weights = np.asarray(weights, dtype=np.float64)
    mean_value = float(weights.mean()) if weights.size else 1.0
    if mean_value <= 0:
        return np.ones_like(weights, dtype=np.float64)
    return weights / mean_value


def _cap_weights_by_median(
    weights: np.ndarray,
    max_weight_ratio_to_median: float | None = None,
) -> np.ndarray:
    """Clip very large sample weights relative to the median weight."""
    if max_weight_ratio_to_median is None:
        return weights
    ratio = float(max_weight_ratio_to_median)
    if ratio <= 0:
        raise ValueError(f"max_weight_ratio_to_median must be positive, got {ratio}")

    positive = weights[weights > 0]
    if positive.size == 0:
        return weights
    median_value = float(np.median(positive))
    if median_value <= 0:
        return weights
    return np.minimum(weights, median_value * ratio)


def _prepare_sample_weight_multipliers(
    sample_weight_multipliers: Union[np.ndarray, list, torch.Tensor, None],
    num_samples: int,
    power: float = 1.0,
    reduction: str = "mean",
) -> np.ndarray | None:
    """Validate and scale optional sample-level multipliers."""
    if sample_weight_multipliers is None:
        return None
    if isinstance(sample_weight_multipliers, torch.Tensor):
        sample_weight_multipliers = sample_weight_multipliers.numpy()
    multipliers = np.asarray(sample_weight_multipliers, dtype=np.float64)
    if multipliers.ndim == 2:
        if len(multipliers) != num_samples:
            raise ValueError(
                f"Expected {num_samples} rows of sample multipliers, got {len(multipliers)}"
            )
        reduction = str(reduction).lower()
        if reduction == "mean":
            multipliers = multipliers.mean(axis=1)
        elif reduction == "min":
            multipliers = multipliers.min(axis=1)
        elif reduction == "max":
            multipliers = multipliers.max(axis=1)
        else:
            raise ValueError(f"Unsupported multiplier reduction: {reduction}")
    elif multipliers.ndim != 1:
        raise ValueError(f"Expected 1D or 2D sample multipliers, got shape {multipliers.shape}")
    if len(multipliers) != num_samples:
        raise ValueError(
            f"Expected {num_samples} sample multipliers, got {len(multipliers)}"
        )
    multipliers = np.clip(multipliers, 1.0e-8, None)
    return np.power(multipliers, float(power))


def compute_multitask_sample_weights(
    labels: Union[np.ndarray, list, torch.Tensor],
    num_classes: int | None = None,
    strategy: str = "mean_class_balance",
    power: float = 1.0,
    normalize: bool = True,
    sample_weight_multipliers: Union[np.ndarray, list, torch.Tensor, None] = None,
    multiplier_power: float = 1.0,
    max_weight_ratio_to_median: float | None = None,
) -> np.ndarray:
    """Compute per-sample weights for a multitask label matrix.

    Args:
        labels: Array-like of shape (N, num_sites) with integer labels.
        num_classes: Number of classes per task. Defaults to inferred max + 1.
        strategy: How to aggregate per-site rarity for each sample.
            - ``mean_class_balance``: average inverse-frequency weight across sites
            - ``sum_class_balance``: sum inverse-frequency weights across sites
            - ``max_class_balance``: maximum inverse-frequency weight across sites
        power: Exponent applied to class weights before aggregation.
        normalize: Whether to normalize class weights within each site.
        sample_weight_multipliers: Optional confidence-derived multipliers with
            shape (N,) or (N, num_sites). When 2D, each site's rarity term is
            attenuated independently before aggregation.
        multiplier_power: Exponent applied to the optional sample multipliers.
        max_weight_ratio_to_median: Optional cap on final sample weights,
            relative to the median positive weight.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels for multitask sampling, got shape {labels.shape}")
    if len(labels) == 0:
        raise ValueError("Cannot compute multitask sample weights from an empty label matrix.")

    if num_classes is None:
        num_classes = int(labels.max()) + 1

    class_weights = compute_multitask_balanced_class_weights(
        labels,
        num_classes=num_classes,
        power=power,
        normalize=normalize,
    )
    per_site_weights = np.stack(
        [class_weights[site_idx][labels[:, site_idx]] for site_idx in range(labels.shape[1])],
        axis=1,
    )

    if sample_weight_multipliers is not None:
        if isinstance(sample_weight_multipliers, torch.Tensor):
            sample_weight_multipliers = sample_weight_multipliers.numpy()
        multipliers = np.asarray(sample_weight_multipliers, dtype=np.float64)
        if multipliers.ndim == 1:
            multipliers = _prepare_sample_weight_multipliers(
                multipliers,
                num_samples=len(labels),
                power=multiplier_power,
            )[:, None]
        elif multipliers.ndim == 2:
            if multipliers.shape != per_site_weights.shape:
                raise ValueError(
                    "Expected 2D sample multipliers to match label shape "
                    f"{per_site_weights.shape}, got {multipliers.shape}"
                )
            multipliers = np.power(np.clip(multipliers, 1.0e-8, None), float(multiplier_power))
        else:
            raise ValueError(
                f"Expected 1D or 2D sample multipliers, got shape {multipliers.shape}"
            )
        per_site_weights = per_site_weights * multipliers

    strategy = str(strategy).lower()
    if strategy == "mean_class_balance":
        sample_weights = per_site_weights.mean(axis=1)
    elif strategy == "sum_class_balance":
        sample_weights = per_site_weights.sum(axis=1)
    elif strategy == "max_class_balance":
        sample_weights = per_site_weights.max(axis=1)
    else:
        raise ValueError(f"Unsupported multitask sampling strategy: {strategy}")
    sample_weights = _cap_weights_by_median(
        sample_weights,
        max_weight_ratio_to_median=max_weight_ratio_to_median,
    )
    return _normalize_positive_weights(sample_weights)



def create_weighted_sampler(
    labels: Union[np.ndarray, list, torch.Tensor],
    num_samples: int = None,
    sample_weight_multipliers: Union[np.ndarray, list, torch.Tensor, None] = None,
    multiplier_power: float = 1.0,
    max_weight_ratio_to_median: float | None = None,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler with inverse-frequency weights.

    Args:
        labels: Array of integer class labels.
        num_samples: Number of samples per epoch. Defaults to len(labels).
        sample_weight_multipliers: Optional sample-level multipliers such as
            confidence weights. Multitask arrays are reduced across sites by
            their mean before being applied.
        multiplier_power: Exponent applied to the optional sample multipliers.
        max_weight_ratio_to_median: Optional cap on final sample weights,
            relative to the median positive weight.

    Returns:
        WeightedRandomSampler for use with DataLoader.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.asarray(labels, dtype=np.int64)

    num_classes = labels.max() + 1
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_weights = 1.0 / np.maximum(counts, 1)

    sample_weights = class_weights[labels]
    multipliers = _prepare_sample_weight_multipliers(
        sample_weight_multipliers,
        num_samples=len(sample_weights),
        power=multiplier_power,
    )
    if multipliers is not None:
        sample_weights = sample_weights * multipliers
    sample_weights = _cap_weights_by_median(
        sample_weights,
        max_weight_ratio_to_median=max_weight_ratio_to_median,
    )
    sample_weights = _normalize_positive_weights(sample_weights)
    sample_weights = torch.from_numpy(sample_weights).double()

    if num_samples is None:
        num_samples = len(labels)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


def create_multitask_weighted_sampler(
    labels: Union[np.ndarray, list, torch.Tensor],
    num_samples: int = None,
    num_classes: int | None = None,
    strategy: str = "mean_class_balance",
    power: float = 1.0,
    normalize: bool = True,
    sample_weight_multipliers: Union[np.ndarray, list, torch.Tensor, None] = None,
    multiplier_power: float = 1.0,
    max_weight_ratio_to_median: float | None = None,
) -> WeightedRandomSampler:
    """Create a sampler for multitask labels using aggregated per-site rarity."""
    sample_weights = compute_multitask_sample_weights(
        labels,
        num_classes=num_classes,
        strategy=strategy,
        power=power,
        normalize=normalize,
        sample_weight_multipliers=sample_weight_multipliers,
        multiplier_power=multiplier_power,
        max_weight_ratio_to_median=max_weight_ratio_to_median,
    )
    sample_weights = torch.from_numpy(sample_weights.astype(np.float64)).double()
    if num_samples is None:
        num_samples = len(sample_weights)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )
