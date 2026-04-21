"""Grad-CAM extraction for CNN-based models."""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMExtractor:
    """Generic Grad-CAM heatmap extractor for any CNN backbone."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model (must have a sequential-like structure).
            target_layer: The convolutional layer to extract CAMs from.
        """
        self.model = model
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def extract_heatmap(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for input image.

        Args:
            image: (1, C, H, W) input tensor.
            target_class: Class index to generate CAM for. None = predicted class.

        Returns:
            (H, W) normalized heatmap in [0, 1].
        """
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        grayscale_cam = self.cam(input_tensor=image, targets=targets)
        return grayscale_cam[0]

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay heatmap on original image.

        Args:
            heatmap: (H, W) normalized heatmap in [0, 1].
            image: (H, W) or (H, W, 3) original image (uint8).
            alpha: Transparency of heatmap overlay.

        Returns:
            (H, W, 3) overlay image (uint8).
        """
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), colormap
        )
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
