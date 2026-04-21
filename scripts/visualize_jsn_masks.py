"""Create human-viewable JSN mask images and overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils.manifest_paths import resolve_manifest_path


VISIBLE_VALUES = {
    0: 0,
    1: 160,  # medial
    2: 255,  # lateral
}

MASK_COLORS = {
    1: np.array([60, 200, 60], dtype=np.uint8),   # green
    2: np.array([60, 80, 230], dtype=np.uint8),   # red-ish in BGR output
}


def _load_image_lookup(annotation_dir: Path, project_root: Path, data_root: Path) -> dict[str, str]:
    candidates = [
        annotation_dir / "manifests" / "jsn_contour_manifest.csv",
        annotation_dir / "packages" / "jsn_contours" / "jsn_contour_manifest.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if {"image_id", "path"}.issubset(df.columns):
                return {
                    str(row["image_id"]).replace(".png", ""): str(
                        resolve_manifest_path(row["path"], project_root=project_root, data_root=data_root)
                    )
                    for _, row in df.iterrows()
                }
    return {}


def _make_visible_mask(mask: np.ndarray) -> np.ndarray:
    visible = np.zeros_like(mask, dtype=np.uint8)
    for value, mapped in VISIBLE_VALUES.items():
        visible[mask == value] = mapped
    return visible


def _make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    for cls, color in MASK_COLORS.items():
        overlay[mask == cls] = color
    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
    blended[mask == 0] = base[mask == 0]
    return blended


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    annotation_dir = Path(cfg.annotation_dir)
    project_root = Path(cfg.project_root)
    data_root = Path(cfg.data.root)
    mask_root = annotation_dir / "jsn_masks"
    visible_root = annotation_dir / "jsn_masks_vis"
    overlay_root = annotation_dir / "jsn_mask_overlays"

    image_lookup = _load_image_lookup(annotation_dir, project_root=project_root, data_root=data_root)
    total = 0

    for split in ("train", "val", "test"):
        split_dir = mask_root / split
        if not split_dir.exists():
            continue

        (visible_root / split).mkdir(parents=True, exist_ok=True)
        (overlay_root / split).mkdir(parents=True, exist_ok=True)

        for mask_path in sorted(split_dir.glob("*.png")):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            image_id = mask_path.stem
            visible = _make_visible_mask(mask)
            cv2.imwrite(str(visible_root / split / mask_path.name), visible)

            src_path = image_lookup.get(image_id)
            if src_path:
                image = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
                if image is not None and image.shape == mask.shape:
                    overlay = _make_overlay(image, mask)
                    cv2.imwrite(str(overlay_root / split / mask_path.name), overlay)

            total += 1

    print(f"Saved visible JSN masks to {visible_root}")
    print(f"Saved JSN overlays to {overlay_root}")
    print(f"Processed {total} masks")


if __name__ == "__main__":
    main()
