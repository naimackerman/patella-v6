"""Ordinally regularized cross-entropy for graded classification tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCrossEntropyLoss(nn.Module):
    """Cross-entropy with an expected-distance ordinal penalty.

    This keeps the target class anchored by standard cross-entropy while
    discouraging probability mass on distant grades more than adjacent grades.
    """

    def __init__(
        self,
        num_classes: int = 4,
        weight: torch.Tensor = None,
        ordinal_weight: float = 0.25,
    ):
        super().__init__()
        self.num_classes = num_classes
        distances = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                distances[i, j] = abs(i - j)
        self.register_buffer("distances", distances)
        self.class_weight = weight
        self.ordinal_weight = float(ordinal_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ordinal-weighted cross-entropy loss.

        Args:
            logits: (B, C) raw logits.
            targets: (B,) integer class labels.
            reduction: "mean" or "none".
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=(
                class_weight.to(logits.device)
                if class_weight is not None
                else self.class_weight.to(logits.device) if self.class_weight is not None else None
            ),
            reduction="none",
        )
        probs = F.softmax(logits, dim=1)
        distance_penalty = (probs * self.distances[targets]).sum(dim=1)
        loss = ce_loss + self.ordinal_weight * distance_penalty

        if reduction == "none":
            return loss
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return loss.mean()
