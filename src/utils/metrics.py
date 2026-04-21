"""Evaluation metrics for KOA analysis."""

from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import (
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Quadratic Weighted Kappa (primary KL grade metric)."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Cohen's Kappa (for osteophyte per-site grading)."""
    return cohen_kappa_score(y_true, y_pred)


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 3) -> Dict[str, float]:
    """Compute per-class and mean Dice coefficient for segmentation.

    Args:
        pred_mask: Predicted mask (H, W) with integer class labels.
        gt_mask: Ground truth mask (H, W).
        num_classes: Number of classes including background.

    Returns:
        Dict with per-class Dice and mean Dice.
    """
    dice_scores = {}
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred_mask == cls).astype(np.float32)
        gt_cls = (gt_mask == cls).astype(np.float32)
        intersection = (pred_cls * gt_cls).sum()
        union = pred_cls.sum() + gt_cls.sum()
        if union < 1e-6:
            dice_scores[f"dice_class_{cls}"] = 1.0 if intersection < 1e-6 else 0.0
        else:
            dice_scores[f"dice_class_{cls}"] = 2.0 * intersection / union

    fg_dices = [v for k, v in dice_scores.items() if k.startswith("dice_class")]
    dice_scores["dice_mean"] = np.mean(fg_dices) if fg_dices else 0.0
    return dice_scores


def hausdorff_95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance between binary masks."""
    pred_border = pred_mask.astype(bool)
    gt_border = gt_mask.astype(bool)

    if not pred_border.any() or not gt_border.any():
        return float("inf")

    # Distance from pred boundary to nearest gt boundary
    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute per-class F1, precision, recall."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if class_names is not None:
        labels = list(range(len(class_names)))
        names = class_names
    else:
        max_label = int(max(y_true.max(initial=0), y_pred.max(initial=0)))
        labels = list(range(max_label + 1))
        names = [str(i) for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {}
    for cls_name in names:
        if cls_name in report:
            metrics[f"f1_{cls_name}"] = report[cls_name]["f1-score"]
            metrics[f"precision_{cls_name}"] = report[cls_name]["precision"]
            metrics[f"recall_{cls_name}"] = report[cls_name]["recall"]
    metrics["accuracy"] = report.get("accuracy", 0.0)
    metrics["f1_macro"] = report["macro avg"]["f1-score"]
    metrics["f1_weighted"] = report["weighted avg"]["f1-score"]
    return metrics


def icc(measurements1: np.ndarray, measurements2: np.ndarray) -> float:
    """Compute ICC(2,1): two-way random, absolute agreement, single measurement.

    This follows the Shrout and Fleiss / McGraw and Wong formulation for two
    raters. It is the appropriate reliability statistic for comparing predicted
    mJSW against ground-truth mJSW while treating both as interchangeable
    measurements of the same quantity.
    """
    x = np.asarray(measurements1, dtype=np.float64).reshape(-1)
    y = np.asarray(measurements2, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("ICC inputs must have the same shape.")

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    n = x.size
    k = 2
    if n < 2:
        return 0.0

    ratings = np.stack([x, y], axis=1)

    mean_targets = ratings.mean(axis=1, keepdims=True)
    mean_raters = ratings.mean(axis=0, keepdims=True)
    grand_mean = float(ratings.mean())

    ss_rows = float(k * np.square(mean_targets - grand_mean).sum())
    ss_cols = float(n * np.square(mean_raters - grand_mean).sum())
    ss_error = float(np.square(ratings - mean_targets - mean_raters + grand_mean).sum())

    ms_rows = ss_rows / max(n - 1, 1)
    ms_cols = ss_cols / max(k - 1, 1)
    ms_error = ss_error / max((n - 1) * (k - 1), 1)

    denominator = ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    if abs(denominator) < 1e-12:
        return 1.0 if np.allclose(x, y, atol=1e-8, rtol=1e-8) else 0.0

    icc_val = (ms_rows - ms_error) / denominator
    return float(np.clip(icc_val, -1.0, 1.0))
