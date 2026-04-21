"""Load reviewed ROI box annotations from CSV or COCO exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from src.models.roi_detector import ROI_CLASSES


def build_image_lookup(data_root: str | Path) -> Dict[str, Dict[str, str]]:
    """Map image ids to split/path records from the dataset tree."""
    data_root = Path(data_root)
    lookup: Dict[str, Dict[str, str]] = {}
    for split in ("train", "val", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for image_path in split_dir.rglob("*.png"):
            lookup[image_path.stem] = {
                "split": split,
                "path": str(image_path),
            }
    return lookup


def load_roi_annotations(
    annotation_dir: str | Path,
    data_root: str | Path,
    class_names: Iterable[str] = ROI_CLASSES,
) -> pd.DataFrame:
    """Load reviewed ROI box annotations and normalize them to a shared schema."""
    annotation_dir = Path(annotation_dir)
    reviewed_dir = annotation_dir / "reviewed"
    lookup = build_image_lookup(data_root)
    class_names = list(class_names)

    csv_candidates = [
        reviewed_dir / "roi_boxes.csv",
        reviewed_dir / "roi_annotations.csv",
    ]
    for csv_path in csv_candidates:
        if csv_path.exists():
            return _load_csv_annotations(csv_path, lookup, class_names)

    coco_candidates = [
        reviewed_dir / "roi_cvat_export.json",
        reviewed_dir / "roi_boxes_coco.json",
    ]
    for coco_path in coco_candidates:
        if coco_path.exists():
            return _load_coco_annotations(coco_path, lookup, class_names)

    raise FileNotFoundError(
        f"No reviewed ROI annotations found in {reviewed_dir}. "
        "Expected roi_boxes.csv, roi_annotations.csv, roi_cvat_export.json, or roi_boxes_coco.json."
    )


def _load_csv_annotations(
    csv_path: Path,
    lookup: Dict[str, Dict[str, str]],
    class_names: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        if "file_name" in df.columns:
            df["image_id"] = df["file_name"].astype(str).map(lambda value: Path(value).stem)
        elif "path" in df.columns:
            df["image_id"] = df["path"].astype(str).map(lambda value: Path(value).stem)
        else:
            raise ValueError(f"{csv_path} must contain image_id, file_name, or path.")

    if "class_name" not in df.columns:
        if "label" in df.columns:
            df["class_name"] = df["label"]
        elif "class_id" in df.columns:
            df["class_name"] = df["class_id"].map(lambda value: class_names[int(value)])
        else:
            raise ValueError(f"{csv_path} must contain class_name/label/class_id.")

    if {"x1", "y1", "x2", "y2"}.issubset(df.columns):
        pass
    elif {"x", "y", "w", "h"}.issubset(df.columns):
        df["x1"] = df["x"]
        df["y1"] = df["y"]
        df["x2"] = df["x"] + df["w"]
        df["y2"] = df["y"] + df["h"]
    else:
        raise ValueError(f"{csv_path} must contain either x1/y1/x2/y2 or x/y/w/h.")

    return _finalize_roi_annotation_df(df, lookup, class_names)


def _load_coco_annotations(
    coco_path: Path,
    lookup: Dict[str, Dict[str, str]],
    class_names: list[str],
) -> pd.DataFrame:
    with open(coco_path, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    category_map = {item["id"]: item["name"] for item in coco.get("categories", [])}
    image_map = {item["id"]: item for item in coco.get("images", [])}
    rows = []
    for ann in coco.get("annotations", []):
        image_info = image_map.get(ann["image_id"])
        if image_info is None:
            continue
        file_name = str(image_info.get("file_name", ""))
        image_id = Path(file_name).stem
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        rows.append({
            "image_id": image_id,
            "file_name": file_name,
            "class_name": category_map.get(ann["category_id"], f"class_{ann['category_id']}"),
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
        })

    df = pd.DataFrame(rows)
    return _finalize_roi_annotation_df(df, lookup, class_names)


def _finalize_roi_annotation_df(
    df: pd.DataFrame,
    lookup: Dict[str, Dict[str, str]],
    class_names: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str).str.replace(".png", "", regex=False)
    df["class_name"] = df["class_name"].astype(str)
    df = df[df["class_name"].isin(class_names)].copy()
    if df.empty:
        raise ValueError("Reviewed ROI annotation table contains no rows for the expected ROI classes.")

    if "split" not in df.columns:
        df["split"] = df["image_id"].map(lambda image_id: lookup.get(image_id, {}).get("split"))
    if "path" not in df.columns:
        df["path"] = df["image_id"].map(lambda image_id: lookup.get(image_id, {}).get("path"))

    missing_split = df["split"].isna()
    missing_path = df["path"].isna()
    if missing_split.any() or missing_path.any():
        unresolved = df.loc[missing_split | missing_path, "image_id"].astype(str).unique().tolist()
        raise ValueError(
            "Could not resolve split/path for ROI annotations: "
            + ", ".join(unresolved[:20])
        )

    df["class_id"] = df["class_name"].map(class_names.index)
    numeric_cols = ["x1", "y1", "x2", "y2"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    return df[["image_id", "split", "path", "class_name", "class_id", "x1", "y1", "x2", "y2"]].sort_values(
        ["split", "image_id", "class_id"]
    )
