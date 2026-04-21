"""Pseudo-label generation for semi-supervised learning.

Implements the three-tier labeling strategy:
- Tier 1: KL-grade-derived conservative pseudo-labels
- Tier 3: Self-training with high-confidence model predictions
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def generate_tier1_osteophyte_labels(
    image_ids: List[str],
    kl_grades: np.ndarray,
) -> pd.DataFrame:
    """Generate Tier 1 pseudo-labels for osteophyte grading from KL grades.

    KL 0-1: All 4 sites labeled as grade 0 (strong negatives).
    KL 2: Labeled as ambiguous (excluded from pseudo-labels).
    KL 3-4: Labeled as "present" (grade >= 1, exact grade unknown).

    Returns:
        DataFrame with columns: image_id, osp_mf, osp_lf, osp_mt, osp_lt, pseudo_label
    """
    records = []
    for img_id, kl in zip(image_ids, kl_grades):
        if kl <= 1:
            # Strong negatives: no osteophytes expected
            records.append({
                "image_id": img_id,
                "osp_mf": 0, "osp_lf": 0, "osp_mt": 0, "osp_lt": 0,
                "pseudo_label": True,
            })
        elif kl >= 3:
            # Conservative positives: osteophytes present but exact grade unknown
            records.append({
                "image_id": img_id,
                "osp_mf": 1, "osp_lf": 1, "osp_mt": 1, "osp_lt": 1,
                "pseudo_label": True,
            })
        # KL 2 excluded (ambiguous)

    return pd.DataFrame(records)


def generate_tier1_sclerosis_labels(
    image_ids: List[str],
    kl_grades: np.ndarray,
) -> pd.DataFrame:
    """Generate Tier 1 pseudo-labels for sclerosis from KL grades.

    KL 0-1: Grade 0 (none) both sides.
    KL 4: Grade >= 1 (present, exact grade unknown).
    """
    records = []
    for img_id, kl in zip(image_ids, kl_grades):
        if kl <= 1:
            # Strong negatives: no sclerosis expected
            records.append({
                "image_id": img_id,
                "scl_medial": 0, "scl_lateral": 0,
                "pseudo_label": True,
            })
        elif kl >= 3:
            # Conservative positives: sclerosis likely present
            records.append({
                "image_id": img_id,
                "scl_medial": 1, "scl_lateral": 1,
                "pseudo_label": True,
            })
        # KL 2 excluded (ambiguous)

    return pd.DataFrame(records)


def generate_tier3_labels(
    model: torch.nn.Module,
    dataloader: DataLoader,
    threshold: float = 0.90,
    device: torch.device = None,
) -> List[Dict]:
    """Generate Tier 3 pseudo-labels from high-confidence model predictions.

    Args:
        model: Trained classification model.
        dataloader: DataLoader for unlabeled data.
        threshold: Minimum softmax probability to accept as pseudo-label.
        device: Device to run inference on.

    Returns:
        List of dicts with keys: image_id, predicted_label, confidence.
    """
    model.eval()
    pseudo_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0]
            paths = batch[2] if len(batch) > 2 else [f"img_{i}" for i in range(len(images))]

            if device:
                images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = probs.max(dim=1)

            for prob, pred, path in zip(max_probs.cpu(), preds.cpu(), paths):
                if prob.item() >= threshold:
                    pseudo_labels.append({
                        "image_id": path if isinstance(path, str) else str(path),
                        "predicted_label": int(pred.item()),
                        "confidence": float(prob.item()),
                    })

    return pseudo_labels
