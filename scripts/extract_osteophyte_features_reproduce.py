"""Export osteophyte feature vectors from the HOW_TO_REPRODUCE model checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from src.models.osteophyte_grader import OsteophyteGrader
from src.utils.device import get_device


class _HistogramClip(A.ImageOnlyTransform):
    """Histogram percentile clipping used by the reproduce configs."""

    def __init__(self, low_pct: int = 5, high_pct: int = 99, always_apply: bool = True, p: float = 1.0):
        super().__init__(p=p)
        self.low_pct = low_pct
        self.high_pct = high_pct

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        low = np.percentile(image, self.low_pct)
        high = np.percentile(image, self.high_pct)
        if high - low < 1e-6:
            return image
        image = np.clip(image, low, high)
        return ((image - low) / (high - low) * 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("low_pct", "high_pct")


def get_eval_transforms(config) -> A.Compose:
    """Evaluation transforms matching the reproduce testing path."""
    prep = config.preprocessing
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


def load_model(checkpoint_path: Path, config):
    """Load the reproduce osteophyte checkpoint."""
    device = get_device()
    model = OsteophyteGrader(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("criterion.") or key.startswith("class_weights"):
            continue
        if key.startswith("model."):
            cleaned_state_dict[key[6:]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    return model, device


def _roi_tensor(roi_path: Path, transform: A.Compose, device) -> torch.Tensor:
    image = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load ROI: {roi_path}")
    transformed = transform(image=image)
    return torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(device)


def _feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([
        features["osp_grade_mf"],
        features["osp_grade_lf"],
        features["osp_grade_mt"],
        features["osp_grade_lt"],
        features["osp_sum"],
        features["osp_max"],
        features["osp_medial_sum"],
        features["osp_lateral_sum"],
        features["osp_femoral_sum"],
        features["osp_tibial_sum"],
    ], dtype=np.float64)


def export_split(model, device, transform: A.Compose, split_dir: Path) -> dict[str, np.ndarray]:
    """Export the standard 10-dim osteophyte features for one split."""
    feature_map: dict[str, np.ndarray] = {}
    mf_files = sorted(split_dir.glob("*_medial_femur.png"))

    for mf_path in tqdm(mf_files, desc=f"Export {split_dir.name}"):
        image_id = mf_path.name[:-len("_medial_femur.png")]
        roi_tensors = {}
        missing_site = False
        for site in OsteophyteGrader.SITES:
            roi_path = split_dir / f"{image_id}_{site}.png"
            if not roi_path.exists():
                missing_site = True
                break
            roi_tensors[site] = _roi_tensor(roi_path, transform, device)
        if missing_site:
            continue

        with torch.no_grad():
            logits_by_site = model(
                roi_tensors["medial_femur"],
                roi_tensors["lateral_femur"],
                roi_tensors["medial_tibia"],
                roi_tensors["lateral_tibia"],
            )
        feature_map[image_id] = _feature_vector(model.extract_osteophyte_features(logits_by_site))

    return feature_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/training/osteophyte_clahe_fullimage_ordinal.yaml",
        help="Reproduce config used to train/test the osteophyte model.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path for the reproduce-model evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="features/osteophyte",
        help="Directory where split feature NPZ files will be written.",
    )
    parser.add_argument(
        "--roi-dir",
        default=None,
        help="Optional ROI directory override. Defaults to config.data.roi_dir.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    roi_dir = Path(args.roi_dir) if args.roi_dir else Path(config.data.roi_dir)
    if not roi_dir.exists():
        raise FileNotFoundError(f"ROI directory not found: {roi_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_model(checkpoint_path, config)
    transform = get_eval_transforms(config)

    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"ROI dir: {roi_dir}")
    print(f"Output dir: {output_dir}")

    for split in ("train", "val", "test"):
        split_dir = roi_dir / split
        if not split_dir.exists():
            continue
        feature_map = export_split(model, device, transform, split_dir)
        output_path = output_dir / f"{split}_osteophyte_features.npz"
        np.savez(str(output_path), **feature_map)
        print(f"Saved {split}: {len(feature_map)} images -> {output_path}")


if __name__ == "__main__":
    main()
