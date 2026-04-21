"""KL grade classification dataset loading from folder structure."""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from src.data.image_validation import can_read_grayscale_image, read_grayscale_image


def _normalize_feature_matrix(
    feature_matrix: np.ndarray,
    normalizer_stats_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Apply z-score normalization when normalizer stats are available."""
    if normalizer_stats_path is None:
        return feature_matrix

    stats_path = Path(normalizer_stats_path)
    if not stats_path.exists():
        return feature_matrix

    stats = np.load(stats_path)
    mean = stats["mean"]
    std = stats["std"]
    std = np.where(std < 1.0e-8, 1.0, std)
    return (feature_matrix - mean) / std


class KLDataset(Dataset):
    """Dataset for KL grade classification from folder-structured images.

    Expected structure: {root}/{split}/{grade}/image.png
    where grade is 0, 1, 2, 3, or 4.
    """

    NUM_CLASSES = 5
    GRADE_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        self.labels = []
        self.skipped_samples = []

        split_dir = self.root / split
        for grade in range(self.NUM_CLASSES):
            grade_dir = split_dir / str(grade)
            if not grade_dir.exists():
                continue
            for img_path in sorted(grade_dir.glob("*.png")):
                if can_read_grayscale_image(img_path):
                    self.samples.append(img_path)
                    self.labels.append(grade)
                else:
                    self.skipped_samples.append(img_path)

        self.labels = np.array(self.labels, dtype=np.int64)
        if self.skipped_samples:
            preview = ", ".join(path.name for path in self.skipped_samples[:5])
            print(
                f"KLDataset skipped {len(self.skipped_samples)} unreadable images "
                f"under {split_dir}. Examples: {preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        label = self.labels[idx]

        image = read_grayscale_image(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Convert to tensor: (H, W) -> (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0).float()

        return image, label, str(img_path)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights."""
        counts = np.bincount(self.labels, minlength=self.NUM_CLASSES).astype(np.float64)
        total = counts.sum()
        weights = total / (self.NUM_CLASSES * counts)
        # Normalize so minimum weight is 1.0
        weights = weights / weights.min()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return class_weights[self.labels]

    @staticmethod
    def parse_laterality(filename: str) -> str:
        """Parse knee laterality (L/R) from filename like '9001695L.png'."""
        stem = Path(filename).stem
        if stem.endswith("L"):
            return "left"
        elif stem.endswith("R"):
            return "right"
        return "unknown"

    @staticmethod
    def parse_patient_id(filename: str) -> str:
        """Extract patient ID from filename like '9001695L.png'."""
        stem = Path(filename).stem
        return stem.rstrip("LR")


class KLHybridDataset(Dataset):
    """Dataset for hybrid ConvNeXt + feature vector training.

    Joins KLDataset images with pre-extracted 50-dim feature vectors.
    Returns (image_tensor, feature_vector, label).
    """

    def __init__(
        self,
        root: str,
        split: str,
        features_npz: str,
        transform: Optional[A.Compose] = None,
        normalize_features: bool = True,
        normalizer_stats_path: Optional[str | Path] = None,
    ):
        self.image_ds = KLDataset(root, split, transform)

        data = np.load(features_npz)
        feat_ids = list(data["image_ids"])
        feat_matrix = data["features"].astype(np.float32)
        stats_path = normalizer_stats_path
        if stats_path is None and normalize_features:
            stats_path = Path(features_npz).with_name("normalizer_stats.npz")
        if normalize_features:
            feat_matrix = _normalize_feature_matrix(feat_matrix, stats_path).astype(np.float32)
        self.feature_map = {img_id: feat_matrix[i] for i, img_id in enumerate(feat_ids)}
        self.feat_dim = feat_matrix.shape[1]

    def __len__(self) -> int:
        return len(self.image_ds)

    def __getitem__(self, idx: int):
        image, label, path = self.image_ds[idx]
        image_id = Path(path).stem
        features = self.feature_map.get(image_id, np.zeros(self.feat_dim, dtype=np.float32))
        return image, torch.tensor(features, dtype=torch.float32), label

    @property
    def labels(self):
        return self.image_ds.labels
