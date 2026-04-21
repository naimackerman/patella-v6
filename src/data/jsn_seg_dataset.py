"""Dataset for joint space segmentation (image + mask pairs)."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from src.data.image_validation import can_read_grayscale_image, can_read_mask, read_grayscale_image


class JSNSegDataset(Dataset):
    """Dataset loading image + 3-class segmentation mask pairs.

    Mask classes: 0=background, 1=medial joint space, 2=lateral joint space.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[A.Compose] = None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Match images to masks by filename, preserving nested grade directories.
        self.samples = []
        self.skipped_samples = []
        for img_path in sorted(self.image_dir.rglob("*.png")):
            relative_path = img_path.relative_to(self.image_dir)
            candidates = [
                self.mask_dir / relative_path,
                self.mask_dir / img_path.name,
                (self.mask_dir / relative_path).with_suffix(".npy"),
                (self.mask_dir / img_path.name).with_suffix(".npy"),
            ]
            mask_path = next((candidate for candidate in candidates if candidate.exists()), None)
            if mask_path is not None:
                if can_read_grayscale_image(img_path) and can_read_mask(mask_path):
                    self.samples.append((img_path, mask_path))
                else:
                    self.skipped_samples.append((img_path, mask_path))

        if self.skipped_samples:
            preview = ", ".join(str(img_path.name) for img_path, _ in self.skipped_samples[:5])
            print(
                f"JSNSegDataset skipped {len(self.skipped_samples)} unreadable image/mask pairs "
                f"under {self.image_dir}. Examples: {preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        # Load grayscale image
        image = read_grayscale_image(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Load mask
        if mask_path.suffix == ".npy":
            mask = np.load(str(mask_path)).astype(np.uint8)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # Apply transforms jointly to image and mask
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).long()

        return image, mask, str(img_path)
