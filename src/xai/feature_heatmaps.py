"""Feature-level Grad-CAM helpers for osteophyte and sclerosis models."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.xai.gradcam import GradCAMExtractor


class _OsteophyteSiteWrapper(nn.Module):
    def __init__(self, model: nn.Module, site: str):
        super().__init__()
        self.model = model
        self.site = site

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.forward_single(image, self.site)


class _SclerosisVisionWrapper(nn.Module):
    def __init__(self, model: nn.Module, texture_features: torch.Tensor):
        super().__init__()
        self.model = model
        self.texture_features = texture_features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image, self.texture_features)


def extract_osteophyte_gradcam(
    model: nn.Module,
    site: str,
    image_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Generate a Grad-CAM heatmap for one osteophyte ROI."""
    target_layer = _find_last_conv_layer(model.backbone)
    if target_layer is None:
        return None
    wrapper = _OsteophyteSiteWrapper(model, site).eval()
    extractor = GradCAMExtractor(wrapper, target_layer)
    return extractor.extract_heatmap(image_tensor, target_class=target_class)


def extract_sclerosis_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    texture_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Generate a Grad-CAM heatmap for one sclerosis ROI."""
    if getattr(model, "cnn", None) is None:
        return None
    target_layer = _find_last_conv_layer(model.cnn)
    if target_layer is None:
        return None
    wrapper = _SclerosisVisionWrapper(model, texture_tensor).eval()
    extractor = GradCAMExtractor(wrapper, target_layer)
    return extractor.extract_heatmap(image_tensor, target_class=target_class)


def compose_heatmap_canvas(
    image_shape: Tuple[int, int],
    heatmap_items: Iterable[Tuple[np.ndarray, Tuple[int, int, int, int]]],
) -> Optional[np.ndarray]:
    """Project ROI heatmaps back into a full-image heatmap canvas."""
    canvas = np.zeros(image_shape[:2], dtype=np.float32)
    used = False
    for heatmap, box in heatmap_items:
        if heatmap is None or box is None:
            continue
        x1, y1, x2, y2 = map(int, box)
        if x2 <= x1 or y2 <= y1:
            continue
        resized = cv2.resize(heatmap.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], resized)
        used = True
    if not used:
        return None
    max_val = float(canvas.max())
    if max_val > 0:
        canvas /= max_val
    return canvas


def _find_last_conv_layer(module: nn.Module) -> Optional[nn.Module]:
    conv_layers = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
    return conv_layers[-1] if conv_layers else None
