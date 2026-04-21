"""CORN (Conditional Ordinal Regression Network) loss for ordinal classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CORNLoss(nn.Module):
    """Conditional Ordinal Regression Network loss.

    Decomposes ordinal classification into K-1 binary classification tasks:
    P(Y > k | Y >= k) for k = 0, 1, ..., K-2.

    Reference: Shi et al., "CORN - Conditional Ordinal Regression for Neural Networks"
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute CORN loss.

        Args:
            logits: (B, K-1) binary logits for each ordinal threshold.
            targets: (B,) integer class labels in [0, K-1].
        """
        num_thresholds = self.num_classes - 1
        loss = torch.zeros(1, device=logits.device)

        for k in range(num_thresholds):
            # Only consider samples with Y >= k
            mask = targets >= k
            if mask.sum() == 0:
                continue

            # Binary label: Y > k given Y >= k
            binary_labels = (targets[mask] > k).float()
            binary_logits = logits[mask, k]

            task_loss = F.binary_cross_entropy_with_logits(binary_logits, binary_labels)
            loss = loss + task_loss

        return loss / num_thresholds
