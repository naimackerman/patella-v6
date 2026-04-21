"""Convert annotation formats (CVAT COCO JSON to masks, CSV parsing)."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


def contours_to_mask(
    femoral_pts: np.ndarray,
    tibial_pts: np.ndarray,
    img_shape: Tuple[int, int],
    midline_x: int = None,
    is_left: bool = False,
) -> np.ndarray:
    """Create 3-class segmentation mask from femoral and tibial contour points.

    Args:
        femoral_pts: (N, 2) array of (x, y) points for femoral surface.
        tibial_pts: (M, 2) array of (x, y) points for tibial surface.
        img_shape: (H, W) shape of the output mask.
        midline_x: X coordinate separating medial/lateral. Defaults to image center.
        is_left: Whether the image is a left-knee view. For left knees the
            medial compartment is on the image left; for right knees it is on
            the image right.

    Returns:
        (H, W) mask with 0=background, 1=medial JS, 2=lateral JS.
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if midline_x is None:
        midline_x = w // 2

    # Interpolate contours to cover all x positions
    fem_x = femoral_pts[:, 0]
    fem_y = femoral_pts[:, 1]
    tib_x = tibial_pts[:, 0]
    tib_y = tibial_pts[:, 1]

    x_min = int(max(fem_x.min(), tib_x.min()))
    x_max = int(min(fem_x.max(), tib_x.max()))

    for x in range(max(0, x_min), min(w, x_max + 1)):
        # Interpolate femoral y at this x
        fem_dists = np.abs(fem_x - x)
        fem_nearest = np.argsort(fem_dists)[:3]
        weights = 1.0 / (fem_dists[fem_nearest] + 1e-6)
        y_fem = int(np.average(fem_y[fem_nearest], weights=weights))

        # Interpolate tibial y at this x
        tib_dists = np.abs(tib_x - x)
        tib_nearest = np.argsort(tib_dists)[:3]
        weights = 1.0 / (tib_dists[tib_nearest] + 1e-6)
        y_tib = int(np.average(tib_y[tib_nearest], weights=weights))

        # Anatomical compartment assignment depends on knee laterality.
        # Right knee: medial on image right, lateral on image left.
        # Left knee: medial on image left, lateral on image right.
        compartment = 1 if ((x < midline_x) if is_left else (x >= midline_x)) else 2
        # Preserve bone-on-bone contact as a one-pixel column so severe KL3/KL4
        # cases are not erased into pure background when the contours touch.
        if y_tib <= y_fem:
            y0 = int(np.clip(y_fem, 0, h - 1))
            mask[y0:y0 + 1, x] = compartment
        else:
            mask[y_fem:y_tib, x] = compartment

    return mask


def infer_is_left_from_name(name: str) -> bool:
    """Infer laterality from an image filename/stem."""
    stem = Path(str(name)).stem.upper()
    return stem.endswith("L")


def cvat_coco_to_masks(
    coco_json_path: str,
    output_dir: str,
    img_shape: Tuple[int, int] = (224, 224),
) -> List[str]:
    """Convert CVAT COCO-format polyline annotations to segmentation masks.

    Expects annotations with categories: 'femoral_surface' and 'tibial_surface'.

    Args:
        coco_json_path: Path to COCO JSON file exported from CVAT.
        output_dir: Directory to save generated mask PNG files.
        img_shape: (H, W) image dimensions.

    Returns:
        List of generated mask file paths.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build category map
    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Build image map. Roboflow exports may append an `.rf...` suffix to
    # `file_name` while preserving the original source name in `extra.name`.
    img_map = {}
    for img in coco["images"]:
        file_name = img.get("file_name", "")
        extra = img.get("extra", {}) or {}
        original_name = extra.get("name")
        img_map[img["id"]] = original_name or file_name

    # Group annotations by image
    annotations_by_image: Dict[int, Dict[str, np.ndarray]] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_name = cat_map[ann["category_id"]]

        # Parse polyline/polygon points
        if "segmentation" in ann and ann["segmentation"]:
            points = np.array(ann["segmentation"][0]).reshape(-1, 2)
        elif "keypoints" in ann:
            kp = np.array(ann["keypoints"]).reshape(-1, 3)
            points = kp[:, :2]
        else:
            continue

        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = {}
        annotations_by_image[img_id][cat_name] = points

    generated = []
    for img_id, contours in annotations_by_image.items():
        if "femoral_surface" not in contours or "tibial_surface" not in contours:
            continue

        image_name = img_map[img_id]
        mask = contours_to_mask(
            contours["femoral_surface"],
            contours["tibial_surface"],
            img_shape,
            is_left=infer_is_left_from_name(image_name),
        )

        filename = Path(image_name).stem + ".png"
        mask_path = output_dir / filename
        cv2.imwrite(str(mask_path), mask)
        generated.append(str(mask_path))

    return generated


def load_osteophyte_labels(csv_path: str) -> pd.DataFrame:
    """Load osteophyte grading labels from CSV.

    Expected columns: image_id, osp_mf, osp_lf, osp_mt, osp_lt,
    confidence_mf, confidence_lf, confidence_mt, confidence_lt, notes
    """
    df = pd.read_csv(csv_path)
    required = ["image_id", "osp_mf", "osp_lf", "osp_mt", "osp_lt"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


def load_sclerosis_labels(csv_path: str) -> pd.DataFrame:
    """Load sclerosis grading labels from CSV.

    Expected columns: image_id, scl_medial, scl_lateral
    """
    df = pd.read_csv(csv_path)
    required = ["image_id", "scl_medial", "scl_lateral"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df
