"""DiceCE loss for joint space segmentation."""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss as MonaiDiceCELoss


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for multi-class segmentation.

    Wraps MONAI's DiceCELoss with softmax activation and class weighting.
    """

    def __init__(self, num_classes: int = 3, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        # MONAI's current API expects `weight`, not `ce_weight`. The repo only
        # needs a scalar balance between CE and Dice here, so keep class weights
        # uniform and use lambda_ce/lambda_dice to control contribution.
        self.loss_fn = MonaiDiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            weight=torch.ones(num_classes, dtype=torch.float32),
            lambda_dice=dice_weight,
            lambda_ce=ce_weight,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute DiceCE loss.

        Args:
            logits: (B, C, H, W) raw logits from segmentation model.
            targets: (B, 1, H, W) or (B, H, W) integer class labels.
        """
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        return self.loss_fn(logits, targets)
