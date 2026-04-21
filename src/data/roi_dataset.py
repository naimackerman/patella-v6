"""Dataset for per-ROI or multi-ROI osteophyte grading patches."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A

from src.utils.annotation_confidence import confidence_at_least, confidence_weight, normalize_confidence


class ROIDataset(Dataset):
    """Dataset for single-site or multi-head osteophyte grading."""

    SITES = ["medial_femur", "lateral_femur", "medial_tibia", "lateral_tibia"]
    SITE_ABBREV = {"medial_femur": "mf", "lateral_femur": "lf",
                   "medial_tibia": "mt", "lateral_tibia": "lt"}

    def __init__(
        self,
        roi_dir: str,
        labels_csv: str,
        site: Optional[str],
        transform: Optional[A.Compose] = None,
        allowed_label_sources: Optional[list[str]] = None,
        min_confidence: str = "low",
        confidence_weights: Optional[dict[str, float]] = None,
    ):
        """
        Args:
            roi_dir: Directory containing ROI patches named {image_id}_{site}.png
            labels_csv: CSV with columns: image_id, osp_mf, osp_lf, osp_mt, osp_lt
            site: One of SITES or None for multi-head training.
            transform: Albumentations transform pipeline.
        """
        self.roi_dir = Path(roi_dir)
        self.site = site
        self.transform = transform
        self.min_confidence = min_confidence
        self.confidence_weights = confidence_weights or {"low": 0.5, "medium": 0.75, "high": 1.0}

        df = pd.read_csv(labels_csv)
        self.samples = []
        self.labels = []
        self.sample_weights = []
        self.confidences = []
        self.sample_keys = []

        if site is not None:
            abbrev = self.SITE_ABBREV[site]
            label_col = f"osp_{abbrev}"
            confidence_col = f"confidence_{abbrev}"

            for _, row in df.iterrows():
                image_id = str(row["image_id"]).replace(".png", "")
                roi_path = self.roi_dir / f"{image_id}_{site}.png"
                source_col = f"label_source_{abbrev}"
                source_value = row.get(source_col, row.get("label_source", None))
                if allowed_label_sources is not None and source_value not in allowed_label_sources:
                    continue
                confidence_value = normalize_confidence(row.get(confidence_col, "high"))
                if not confidence_at_least(confidence_value, self.min_confidence):
                    continue
                if roi_path.exists() and pd.notna(row[label_col]):
                    self.samples.append(roi_path)
                    self.labels.append(int(row[label_col]))
                    self.sample_weights.append(confidence_weight(confidence_value, self.confidence_weights))
                    self.confidences.append(confidence_value)
                    self.sample_keys.append(image_id)
        else:
            for _, row in df.iterrows():
                image_id = str(row["image_id"]).replace(".png", "")
                roi_paths = []
                labels = []
                sample_weights = []
                confidences = []
                valid = True
                for current_site in self.SITES:
                    abbrev = self.SITE_ABBREV[current_site]
                    label_col = f"osp_{abbrev}"
                    confidence_col = f"confidence_{abbrev}"
                    source_col = f"label_source_{abbrev}"
                    source_value = row.get(source_col, row.get("label_source", None))
                    roi_path = self.roi_dir / f"{image_id}_{current_site}.png"

                    if allowed_label_sources is not None and source_value not in allowed_label_sources:
                        valid = False
                        break
                    if not roi_path.exists() or not pd.notna(row[label_col]):
                        valid = False
                        break

                    confidence_value = normalize_confidence(row.get(confidence_col, "high"))
                    if not confidence_at_least(confidence_value, self.min_confidence):
                        valid = False
                        break

                    roi_paths.append(roi_path)
                    labels.append(int(row[label_col]))
                    sample_weights.append(confidence_weight(confidence_value, self.confidence_weights))
                    confidences.append(confidence_value)

                if valid:
                    self.samples.append(tuple(roi_paths))
                    self.labels.append(labels)
                    self.sample_weights.append(sample_weights)
                    self.confidences.append(confidences)
                    self.sample_keys.append(image_id)

        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.sample_weights = np.asarray(self.sample_weights, dtype=np.float32)
        if self.labels.ndim == 1:
            self.sampling_labels = self.labels
        else:
            # Bias sampling toward images with more severe osteophyte burden.
            self.sampling_labels = self.labels.max(axis=1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.site is not None:
            img_path = self.samples[idx]
            label = self.labels[idx]

            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load ROI: {img_path}")

            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]

            image = torch.from_numpy(image).unsqueeze(0).float()
            sample_weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
            return image, label, str(img_path), sample_weight

        roi_tensors = []
        for img_path in self.samples[idx]:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load ROI: {img_path}")
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed["image"]
            roi_tensors.append(torch.from_numpy(image).unsqueeze(0).float())

        image_stack = torch.stack(roi_tensors, dim=0)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        sample_weight = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return image_stack, labels, self.sample_keys[idx], sample_weight
