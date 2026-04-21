#!/usr/bin/env python3
"""Retry failed sclerosis extraction rows and merge them into existing outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.timed_image_reader import TimedImageReader
from src.features.bootstrap_heuristics import (
    build_sclerosis_features,
    extract_geometric_subchondral_roi,
)
from src.features.subchondral_roi import extract_subchondral_roi_with_boxes_and_source
from src.features.texture_features import extract_all_texture_features
from src.utils.annotation_confidence import normalize_confidence
from src.utils.annotation_paths import resolve_annotation_csv, select_label_subset


def _resolve_image_path(data_root: Path, split: str, image_id: str) -> Path | None:
    matches = list((data_root / split).rglob(f"{image_id}.png"))
    if not matches:
        return None
    return matches[0]


def _load_label_map(annotation_dir: Path, label_mode: str, allow_bootstrap_fallback: bool) -> dict[str, dict]:
    label_csv = resolve_annotation_csv(
        annotation_dir,
        "sclerosis_labels",
        mode=label_mode,
        allow_bootstrap_fallback=allow_bootstrap_fallback,
    )
    label_df = pd.read_csv(label_csv)
    label_df, subset_mode = select_label_subset(
        label_df,
        mode=label_mode,
        allow_bootstrap_fallback=allow_bootstrap_fallback,
    )
    print(f"Loaded sclerosis labels from {label_csv} ({len(label_df)} rows, mode={subset_mode})")
    label_map: dict[str, dict] = {}
    for _, row in label_df.iterrows():
        image_id = str(row["image_id"]).replace(".png", "")
        label_map[image_id] = row.to_dict()
    return label_map


def _load_existing_feature_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _save_feature_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **payload)


def _merge_retry_rows(existing_npz_path: Path, retry_rows: list[dict]) -> None:
    existing = np.load(existing_npz_path, allow_pickle=True)
    image_ids_existing = np.asarray(existing["image_ids"]).astype(str)
    base_ids_existing = np.asarray([value.rsplit("_", 1)[0] for value in image_ids_existing], dtype=object)
    retry_base_ids = {row["base_image_id"] for row in retry_rows}
    keep_mask = ~np.isin(base_ids_existing, list(retry_base_ids))

    merged = {}
    for key in existing.files:
        merged[key] = np.asarray(existing[key])[keep_mask]
    if "roi_sources" not in merged:
        merged["roi_sources"] = np.full(int(keep_mask.sum()), "unknown", dtype=object)

    append_map = {
        "roi_paths": np.asarray([row["roi_path"] for row in retry_rows], dtype=object),
        "texture_features": np.asarray([row["texture_features"] for row in retry_rows], dtype=np.float64),
        "side_ids": np.asarray([row["side_id"] for row in retry_rows], dtype=np.int64),
        "grades": np.asarray([row["grade"] for row in retry_rows], dtype=np.int64),
        "image_ids": np.asarray([row["image_id"] for row in retry_rows], dtype=object),
        "label_sources": np.asarray([row["label_source"] for row in retry_rows], dtype=object),
        "confidence_levels": np.asarray([row["confidence_level"] for row in retry_rows], dtype=object),
        "roi_sources": np.asarray([row["roi_source"] for row in retry_rows], dtype=object),
    }

    for key, values in append_map.items():
        existing_values = np.asarray(merged[key])
        if existing_values.dtype == object or values.dtype == object:
            merged[key] = np.concatenate([existing_values.astype(object), values.astype(object)], axis=0)
        else:
            merged[key] = np.concatenate([existing_values, values], axis=0)

    np.savez(str(existing_npz_path), **merged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--failure-csv",
        default="features/sclerosis/sclerosis_extraction_failures.csv",
        help="CSV listing failed sclerosis extraction rows.",
    )
    parser.add_argument(
        "--remaining-failure-csv",
        default="features/sclerosis/sclerosis_extraction_failures_remaining.csv",
        help="Where to write rows that still fail after retry.",
    )
    parser.add_argument(
        "--data-root",
        default="KneeXrayData/ClsKLData/kneeKL224",
        help="Root knee image directory containing train/val/test grade folders.",
    )
    parser.add_argument(
        "--feature-dir",
        default="features",
        help="Feature root directory containing jsn/ and sclerosis/ outputs.",
    )
    parser.add_argument(
        "--sclerosis-dir",
        default=None,
        help="Optional explicit sclerosis artifact directory. Defaults to <feature-dir>/sclerosis.",
    )
    parser.add_argument(
        "--annotation-dir",
        default="annotations",
        help="Annotation directory containing reviewed sclerosis labels.",
    )
    parser.add_argument(
        "--label-mode",
        default="manual",
        choices=["manual", "expanded", "auto", "bootstrap"],
        help="Label subset mode for assigning sclerosis grades.",
    )
    parser.add_argument(
        "--allow-bootstrap-fallback",
        action="store_true",
        help="Allow bootstrap labels if reviewed/expanded labels are unavailable.",
    )
    parser.add_argument(
        "--image-read-timeout-seconds",
        type=float,
        default=5.0,
        help="Per-image load timeout in seconds.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    failure_csv = (project_root / args.failure_csv).resolve()
    remaining_failure_csv = (project_root / args.remaining_failure_csv).resolve()
    data_root = (project_root / args.data_root).resolve()
    feature_dir = (project_root / args.feature_dir).resolve()
    annotation_dir = (project_root / args.annotation_dir).resolve()

    if not failure_csv.exists():
        raise FileNotFoundError(f"Failure CSV not found: {failure_csv}")

    preprocessing_cfg = OmegaConf.load(project_root / "configs" / "preprocessing" / "default.yaml")
    scl_cfg = preprocessing_cfg.sclerosis_roi
    roi_min_depth_px = int(getattr(scl_cfg, "min_depth_px", 20))
    roi_depth_fraction = float(getattr(scl_cfg, "depth_fraction", 0.12))
    roi_medial_depth_fraction = float(getattr(scl_cfg, "medial_depth_fraction", roi_depth_fraction))
    roi_lateral_depth_fraction = float(getattr(scl_cfg, "lateral_depth_fraction", roi_depth_fraction))
    roi_offset_pct = float(getattr(scl_cfg, "offset_pct", 0.10))
    roi_medial_offset_pct = float(getattr(scl_cfg, "medial_offset_pct", roi_offset_pct))
    roi_lateral_offset_pct = float(getattr(scl_cfg, "lateral_offset_pct", roi_offset_pct))
    roi_medial_inner_offset_pct = getattr(scl_cfg, "medial_inner_offset_pct", None)
    roi_medial_outer_offset_pct = getattr(scl_cfg, "medial_outer_offset_pct", None)
    roi_lateral_inner_offset_pct = getattr(scl_cfg, "lateral_inner_offset_pct", None)
    roi_lateral_outer_offset_pct = getattr(scl_cfg, "lateral_outer_offset_pct", None)
    roi_surface_offset_fraction = float(getattr(scl_cfg, "surface_offset_fraction", 0.015))
    roi_surface_smoothing_window = int(getattr(scl_cfg, "surface_smoothing_window", 5))
    roi_output_size = int(getattr(scl_cfg, "output_size", 96))

    jsn_mask_dir = feature_dir / "jsn" / "masks"
    output_dir = (project_root / args.sclerosis_dir).resolve() if args.sclerosis_dir else (feature_dir / "sclerosis")
    label_map = _load_label_map(annotation_dir, args.label_mode, args.allow_bootstrap_fallback)

    failures = list(csv.DictReader(failure_csv.open()))
    print(f"Retrying {len(failures)} sclerosis failures from {failure_csv}")

    retry_rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    retry_image_features_by_split: dict[str, dict[str, dict]] = {"train": {}, "val": {}, "test": {}}
    remaining_failures: list[dict] = []

    with TimedImageReader(args.image_read_timeout_seconds) as image_reader:
        for row in failures:
            image_id = row["image_id"]
            split = row["split"]
            image_path = _resolve_image_path(data_root, split, image_id)
            if image_path is None:
                remaining_failures.append({"image_id": image_id, "split": split, "reason": "image_not_found"})
                continue

            image = image_reader.read(image_path)
            if image is None:
                remaining_failures.append({"image_id": image_id, "split": split, "reason": "image_load_failed"})
                continue

            mask_path = jsn_mask_dir / f"{image_id}_mask.npy"
            if mask_path.exists():
                try:
                    jsn_mask = np.load(str(mask_path))
                except Exception:
                    jsn_mask = np.zeros_like(image, dtype=np.uint8)
            else:
                jsn_mask = np.zeros_like(image, dtype=np.uint8)

            is_left = image_id.upper().endswith("L")
            try:
                medial_roi, lateral_roi, _, _, roi_source = extract_subchondral_roi_with_boxes_and_source(
                    jsn_mask,
                    image,
                    depth_px=roi_min_depth_px,
                    offset_pct=roi_offset_pct,
                    medial_offset_pct=roi_medial_offset_pct,
                    lateral_offset_pct=roi_lateral_offset_pct,
                    medial_depth_fraction=roi_medial_depth_fraction,
                    lateral_depth_fraction=roi_lateral_depth_fraction,
                    medial_inner_offset_pct=roi_medial_inner_offset_pct,
                    medial_outer_offset_pct=roi_medial_outer_offset_pct,
                    lateral_inner_offset_pct=roi_lateral_inner_offset_pct,
                    lateral_outer_offset_pct=roi_lateral_outer_offset_pct,
                    is_left=is_left,
                    depth_fraction=roi_depth_fraction,
                    surface_offset_fraction=roi_surface_offset_fraction,
                    surface_smoothing_window=roi_surface_smoothing_window,
                    output_size=roi_output_size,
                )
            except Exception:
                medial_roi, lateral_roi = None, None
                roi_source = "jsn_exception"

            if medial_roi is None or lateral_roi is None:
                medial_roi, lateral_roi = extract_geometric_subchondral_roi(image, is_left=is_left)
                roi_source = "geometric_fallback"
            if medial_roi is None or lateral_roi is None:
                remaining_failures.append({"image_id": image_id, "split": split, "reason": "subchondral_roi_failed"})
                continue

            if medial_roi.shape != (roi_output_size, roi_output_size):
                medial_roi = cv2.resize(medial_roi, (roi_output_size, roi_output_size), interpolation=cv2.INTER_LINEAR)
            if lateral_roi.shape != (roi_output_size, roi_output_size):
                lateral_roi = cv2.resize(lateral_roi, (roi_output_size, roi_output_size), interpolation=cv2.INTER_LINEAR)

            label_row = label_map.get(image_id)
            predicted_grade_by_side = {}
            for side_name, roi in (("medial", medial_roi), ("lateral", lateral_roi)):
                side_id = 0 if side_name == "medial" else 1
                roi_save_path = output_dir / "patches" / split / f"{image_id}_{side_name}.png"
                roi_save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(roi_save_path), roi)

                tex_feats = extract_all_texture_features(roi)
                label_key = f"scl_{side_name}"
                label_source_key = f"label_source_{side_name}"
                confidence_key = "scl_confidence_med" if side_name == "medial" else "scl_confidence_lat"
                has_manual_label = (
                    label_row is not None
                    and label_key in label_row
                    and pd.notna(label_row[label_key])
                    and str(label_row[label_key]).strip() != ""
                )
                if has_manual_label:
                    grade = int(label_row[label_key])
                    label_source = str(label_row.get(label_source_key, label_row.get("label_source", "manual_review")))
                    confidence_level = normalize_confidence(label_row.get(confidence_key, "high"))
                else:
                    feature_dict = build_sclerosis_features(
                        medial_roi if side_name == "medial" else None,
                        lateral_roi if side_name == "lateral" else None,
                    )
                    grade = int(feature_dict[f"scl_grade_{side_name}"])
                    label_source = "heuristic_image_only"
                    confidence_level = "high"

                retry_rows_by_split[split].append({
                    "base_image_id": image_id,
                    "image_id": f"{image_id}_{side_name}",
                    "roi_path": str(roi_save_path),
                    "texture_features": np.asarray(tex_feats, dtype=np.float64),
                    "side_id": side_id,
                    "grade": grade,
                    "label_source": label_source,
                    "confidence_level": confidence_level,
                    "roi_source": roi_source,
                })
                predicted_grade_by_side[side_name] = grade

            image_level_feature_dict = build_sclerosis_features(medial_roi, lateral_roi)
            image_level_feature_dict["scl_grade_medial"] = predicted_grade_by_side["medial"]
            image_level_feature_dict["scl_grade_lateral"] = predicted_grade_by_side["lateral"]
            retry_image_features_by_split[split][image_id] = image_level_feature_dict

    for split in ("train", "val", "test"):
        if not retry_rows_by_split[split]:
            continue

        dataset_npz_path = output_dir / f"{split}_sclerosis_data.npz"
        _merge_retry_rows(dataset_npz_path, retry_rows_by_split[split])

        scalar_features = _load_existing_feature_npz(output_dir / f"{split}_sclerosis_features.npz")
        lbp_histograms = _load_existing_feature_npz(output_dir / f"{split}_sclerosis_lbp_histograms.npz")
        for image_id, feat_dict in retry_image_features_by_split[split].items():
            scalar_features[image_id] = np.array([
                feat_dict["scl_grade_medial"],
                feat_dict["scl_grade_lateral"],
                feat_dict["scl_intensity_medial"],
                feat_dict["scl_intensity_lateral"],
                feat_dict["scl_fractal_dim_med"],
                feat_dict["scl_fractal_dim_lat"],
                feat_dict["scl_glcm_contrast_med"],
                feat_dict["scl_glcm_dissimilarity_med"],
                feat_dict["scl_glcm_homogeneity_med"],
                feat_dict["scl_glcm_energy_med"],
                feat_dict["scl_glcm_correlation_med"],
                feat_dict["scl_glcm_contrast_lat"],
                feat_dict["scl_glcm_dissimilarity_lat"],
                feat_dict["scl_glcm_homogeneity_lat"],
                feat_dict["scl_glcm_energy_lat"],
                feat_dict["scl_glcm_correlation_lat"],
                feat_dict["scl_lbp_entropy_med"],
                feat_dict["scl_lbp_entropy_lat"],
            ], dtype=np.float64)
            lbp_histograms[image_id] = np.concatenate([
                np.asarray(feat_dict.get("scl_lbp_hist_med", np.zeros(54)), dtype=np.float64),
                np.asarray(feat_dict.get("scl_lbp_hist_lat", np.zeros(54)), dtype=np.float64),
            ])
        _save_feature_npz(output_dir / f"{split}_sclerosis_features.npz", scalar_features)
        _save_feature_npz(output_dir / f"{split}_sclerosis_lbp_histograms.npz", lbp_histograms)

    remaining_failure_csv.parent.mkdir(parents=True, exist_ok=True)
    with remaining_failure_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", "reason"])
        writer.writeheader()
        writer.writerows(remaining_failures)

    processed_count = sum(len(values) for values in retry_rows_by_split.values()) // 2
    print(f"Recovered {processed_count} failed sclerosis images.")
    print(f"Remaining failures: {len(remaining_failures)}")
    print(f"Remaining failure log: {remaining_failure_csv}")


if __name__ == "__main__":
    main()
