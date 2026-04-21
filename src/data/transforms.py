"""Image preprocessing and augmentation pipelines using Albumentations."""

import cv2
import numpy as np
import albumentations as A
from omegaconf import DictConfig


def _histogram_clip(image: np.ndarray, low_pct: int = 5, high_pct: int = 99, **kwargs) -> np.ndarray:
    """Clip pixel values to percentile range and rescale to [0, 255]."""
    low = np.percentile(image, low_pct)
    high = np.percentile(image, high_pct)
    if high - low < 1e-6:
        return image
    image = np.clip(image, low, high)
    image = ((image - low) / (high - low) * 255).astype(np.uint8)
    return image


class _HistogramClip(A.ImageOnlyTransform):
    """Picklable Albumentations transform for histogram percentile clipping.

    Using a proper class (not a lambda) so multiprocessing workers can pickle it.
    """

    def __init__(self, low_pct: int = 5, high_pct: int = 99, always_apply: bool = True, p: float = 1.0):
        super().__init__(p=p)
        self.low_pct = low_pct
        self.high_pct = high_pct

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        return _histogram_clip(image, self.low_pct, self.high_pct)

    def get_transform_init_args_names(self):
        return ("low_pct", "high_pct")


def get_train_transforms(cfg: DictConfig) -> A.Compose:
    """Training transforms: CLAHE, histogram clip, augmentation, normalize."""
    prep = cfg.preprocessing
    aug = prep.augmentation
    transforms = []

    if hasattr(prep, "clahe") and prep.clahe is not None:
        transforms.append(A.CLAHE(
            clip_limit=prep.clahe.clip_limit,
            tile_grid_size=tuple(prep.clahe.tile_grid_size),
            p=1.0,
        ))

    if hasattr(prep, "histogram_clip") and prep.histogram_clip is not None:
        transforms.append(_HistogramClip(
            low_pct=prep.histogram_clip.low_percentile,
            high_pct=prep.histogram_clip.high_percentile,
            p=1.0,
        ))

    transforms.extend([
        A.Rotate(limit=aug.rotation_limit, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.HorizontalFlip(p=aug.horizontal_flip_p),
        A.RandomBrightnessContrast(
            brightness_limit=aug.brightness_limit,
            contrast_limit=aug.contrast_limit,
            p=0.5,
        ),
        A.Affine(
            scale=tuple(aug.scale_range),
            translate_percent=aug.translate_pct,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        A.Normalize(
            mean=prep.normalize.mean,
            std=prep.normalize.std,
            max_pixel_value=255.0,
        ),
    ])
    return A.Compose(transforms)


def get_eval_transforms(cfg: DictConfig) -> A.Compose:
    """Evaluation transforms: CLAHE, histogram clip, normalize (no augmentation)."""
    prep = cfg.preprocessing
    transforms = []

    if hasattr(prep, "clahe") and prep.clahe is not None:
        transforms.append(A.CLAHE(
            clip_limit=prep.clahe.clip_limit,
            tile_grid_size=tuple(prep.clahe.tile_grid_size),
            p=1.0,
        ))

    if hasattr(prep, "histogram_clip") and prep.histogram_clip is not None:
        transforms.append(_HistogramClip(
            low_pct=prep.histogram_clip.low_percentile,
            high_pct=prep.histogram_clip.high_percentile,
            p=1.0,
        ))

    transforms.append(A.Normalize(
        mean=prep.normalize.mean,
        std=prep.normalize.std,
        max_pixel_value=255.0,
    ))
    return A.Compose(transforms)


def get_segmentation_transforms(cfg: DictConfig, is_train: bool = True) -> A.Compose:
    """Segmentation transforms with joint image+mask augmentation."""
    prep = cfg.preprocessing
    aug = prep.augmentation
    seg_flip_p = float(getattr(aug, "segmentation_horizontal_flip_p", 0.0))

    base_transforms = [
        A.CLAHE(
            clip_limit=prep.clahe.clip_limit,
            tile_grid_size=tuple(prep.clahe.tile_grid_size),
            p=1.0,
        ),
        _HistogramClip(
            low_pct=prep.histogram_clip.low_percentile,
            high_pct=prep.histogram_clip.high_percentile,
            p=1.0,
        ),
    ]

    if is_train:
        base_transforms.extend([
            A.Rotate(limit=aug.rotation_limit, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.HorizontalFlip(p=seg_flip_p),
            A.RandomBrightnessContrast(
                brightness_limit=aug.brightness_limit,
                contrast_limit=aug.contrast_limit,
                p=0.5,
            ),
            A.Affine(
                scale=tuple(aug.scale_range),
                translate_percent=aug.translate_pct,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
        ])

    base_transforms.append(
        A.Normalize(
            mean=prep.normalize.mean,
            std=prep.normalize.std,
            max_pixel_value=255.0,
        )
    )

    return A.Compose(base_transforms)
