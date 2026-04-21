"""Compatibility ordinal losses for legacy training scripts.

The original reproduce scripts import from ``src.losses.coral_loss``.
This repo later consolidated the actively used implementations into
``focal_ordinal_ce.py`` and ``ordinal_ce.py``. These wrappers preserve the
legacy import path and constructor names without changing the reproduce CLI.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.focal_ordinal_ce import FocalOrdinalCrossEntropyLoss
from src.losses.ordinal_ce import OrdinalCrossEntropyLoss


class OrdinalFocalLoss(FocalOrdinalCrossEntropyLoss):
    """Backward-compatible alias for focal ordinal cross-entropy."""


class DistanceWeightedCE(nn.Module):
    """Cross-entropy with an expected ordinal-distance penalty.

    This is a compatibility wrapper for older experiments that referenced
    ``DistanceWeightedCE``. The model still emits standard multi-class logits.
    """

    def __init__(
        self,
        num_classes: int = 4,
        distance_power: float = 2.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.distance_power = float(distance_power)
        self.class_weight = weight

        distances = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                distances[i, j] = abs(i - j) ** self.distance_power
        self.register_buffer("distances", distances)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = class_weight if class_weight is not None else self.class_weight
        if weight is not None:
            weight = weight.to(logits.device)

        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction="none")
        probs = F.softmax(logits, dim=1)
        distance_penalty = (probs * self.distances[targets]).sum(dim=1)
        loss = ce_loss + distance_penalty

        if reduction == "none":
            return loss
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return loss.mean()


class CORALLoss(OrdinalCrossEntropyLoss):
    """Backward-compatible ordinal CE stand-in for the legacy CORAL import.

    The current reproduce models emit 4-class logits rather than K-1 threshold
    logits, so this wrapper keeps the old script runnable with a stable
    ordinally-aware objective on the existing output head shape.
    """

    def __init__(self, num_classes: int = 4, weight: torch.Tensor | None = None):
        super().__init__(num_classes=num_classes, weight=weight, ordinal_weight=0.5)
