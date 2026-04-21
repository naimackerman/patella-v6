import numpy as np

from src.utils.class_weights import (
    compute_balanced_class_weights,
    compute_multitask_balanced_class_weights,
)
from src.data.sampler import compute_multitask_sample_weights


def test_compute_balanced_class_weights_emphasizes_rare_classes():
    labels = np.array([0, 0, 0, 1, 2], dtype=np.int64)
    weights = compute_balanced_class_weights(labels, num_classes=3)

    assert weights.shape == (3,)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[2] > weights[1] > weights[0]


def test_compute_multitask_balanced_class_weights_is_site_specific():
    labels = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 1],
        ],
        dtype=np.int64,
    )

    weights = compute_multitask_balanced_class_weights(labels, num_classes=3)

    assert weights.shape == (2, 3)
    assert np.isclose(weights[0].mean(), 1.0)
    assert np.isclose(weights[1].mean(), 1.0)
    assert weights[0, 2] > weights[0, 1] > weights[0, 0]
    assert weights[1, 0] > weights[1, 1]


def test_compute_multitask_sample_weights_can_emphasize_jointly_rare_examples():
    labels = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 1],
        ],
        dtype=np.int64,
    )

    weights = compute_multitask_sample_weights(labels, num_classes=3, strategy="mean_class_balance")

    assert weights.shape == (4,)
    assert weights[3] > weights[0]


def test_compute_multitask_sample_weights_can_downweight_low_confidence_rare_examples():
    labels = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 1],
            [2, 1],
        ],
        dtype=np.int64,
    )
    sample_weight_multipliers = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.25, 0.25],
        ],
        dtype=np.float32,
    )

    uncapped = compute_multitask_sample_weights(labels, num_classes=3, strategy="mean_class_balance")
    weighted = compute_multitask_sample_weights(
        labels,
        num_classes=3,
        strategy="mean_class_balance",
        sample_weight_multipliers=sample_weight_multipliers,
    )

    assert weighted.shape == (4,)
    assert weighted[3] < uncapped[3]


def test_compute_multitask_sample_weights_can_cap_extreme_values():
    labels = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [2, 2],
        ],
        dtype=np.int64,
    )

    weights = compute_multitask_sample_weights(
        labels,
        num_classes=3,
        strategy="max_class_balance",
        max_weight_ratio_to_median=1.2,
    )

    assert weights.shape == (4,)
    assert weights.max() <= np.median(weights) * 1.2 + 1.0e-6
