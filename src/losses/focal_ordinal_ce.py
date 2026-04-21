"""Focal ordinal cross-entropy for imbalanced graded classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalOrdinalCrossEntropyLoss(nn.Module):
    """Cross-entropy with focal weighting and an ordinal distance penalty."""

    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ordinal_weight: float = 0.5,
        class_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ordinal_weight = float(ordinal_weight)
        self.class_weight = class_weight

        distances = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                distances[i, j] = abs(i - j)
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

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            reduction="none",
        )
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1.0e-8)
        focal_loss = self.alpha * torch.pow(1.0 - pt, self.gamma) * ce_loss
        distance_penalty = (probs * self.distances[targets]).sum(dim=1)
        loss = focal_loss + self.ordinal_weight * distance_penalty

        if reduction == "none":
            return loss
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return loss.mean()
