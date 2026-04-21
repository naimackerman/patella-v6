"""Gaussian-smoothed soft-label cross-entropy loss for ordinal classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLabelCELoss(nn.Module):
    """Cross-entropy with Gaussian-smoothed ordinal targets.

    Instead of one-hot targets, uses Gaussian-smoothed distributions that
    respect the ordinal nature of KL grades.

    For example, KL grade 2 produces target: [0.027, 0.242, 0.461, 0.242, 0.027]
    """

    def __init__(self, num_classes: int = 5, sigma: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma

        # Precompute soft label matrix
        grades = torch.arange(num_classes, dtype=torch.float32)
        soft_labels = torch.zeros(num_classes, num_classes)
        for k in range(num_classes):
            dists = -((grades - k) ** 2) / (2 * sigma ** 2)
            soft_labels[k] = F.softmax(dists, dim=0)
        self.register_buffer("soft_labels", soft_labels)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute soft-label cross-entropy.

        Args:
            logits: (B, C) raw model logits.
            targets: (B,) integer class labels.
        """
        soft_targets = self.soft_labels[targets]  # (B, C)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1)
        return loss.mean()
