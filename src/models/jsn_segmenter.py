"""U-Net++ segmentation model for joint space segmentation."""

import segmentation_models_pytorch as smp
import torch.nn as nn
from omegaconf import DictConfig


def create_jsn_segmenter(cfg: DictConfig) -> nn.Module:
    """Create U-Net++ model with EfficientNet-B4 encoder.

    Args:
        cfg: Model config with encoder_name, encoder_weights, in_channels, classes.

    Returns:
        U-Net++ model outputting raw logits (no activation).
    """
    model = smp.UnetPlusPlus(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
        activation=None,
    )

    # Enable gradient checkpointing on encoder for M4 memory savings
    if cfg.get("gradient_checkpointing", False):
        if hasattr(model.encoder, "set_grad_checkpointing"):
            model.encoder.set_grad_checkpointing(True)

    return model
