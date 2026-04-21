"""Dataset for sclerosis classification with dual input (image + texture features)."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class SclerosisDataset(Dataset):
    """Dataset returning (ROI image, texture feature vector, sclerosis grade).

    Supports the hybrid CNN + texture MLP architecture.
    """

    def __init__(
        self,
        roi_paths: list,
        texture_features: np.ndarray,
        side_ids: np.ndarray,
        grades: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        transform: Optional[A.Compose] = None,
        target_size: Optional[int] = None,
    ):
        """
        Args:
            roi_paths: List of paths to subchondral ROI patch images.
            texture_features: (N, D) precomputed texture feature vectors.
            side_ids: (N,) compartment ids (0=medial, 1=lateral).
            grades: (N,) integer sclerosis grades (0=none, 1=mild, 2=significant).
            transform: Albumentations transform for the ROI image.
        """
        assert len(roi_paths) == len(texture_features) == len(side_ids) == len(grades)
        self.roi_paths = roi_paths
        self.texture_features = torch.tensor(texture_features, dtype=torch.float32)
        self.side_ids = torch.tensor(side_ids, dtype=torch.long)
        self.grades = torch.tensor(grades, dtype=torch.long)
        if sample_weights is None:
            sample_weights = np.ones(len(roi_paths), dtype=np.float32)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self.transform = transform
        self.target_size = int(target_size) if target_size is not None else None

    def __len__(self) -> int:
        return len(self.roi_paths)

    def __getitem__(self, idx: int):
        roi_path = self.roi_paths[idx]
        texture_vec = self.texture_features[idx]
        side_id = self.side_ids[idx]
        grade = self.grades[idx]

        image = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            fallback_size = self.target_size or 64
            image = np.zeros((fallback_size, fallback_size), dtype=np.uint8)
        elif self.target_size is not None and image.shape != (self.target_size, self.target_size):
            image = cv2.resize(
                image,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        image = torch.from_numpy(image).unsqueeze(0).float()
        sample_weight = self.sample_weights[idx]
        return image, texture_vec, side_id, grade, sample_weight
