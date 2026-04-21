"""Prepare annotation packages from manifests and heuristic suggestions."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils.manifest_paths import resolve_manifest_path


FEATURE_REVIEW_COLUMNS = [
    "image_id",
    "split",
    "kl_grade",
    "path",
    "local_image_path",
    "suggestion_osp_mf",
    "suggestion_osp_lf",
    "suggestion_osp_mt",
    "suggestion_osp_lt",
    "suggestion_scl_medial",
    "suggestion_scl_lateral",
    "final_osp_mf",
    "final_osp_lf",
    "final_osp_mt",
    "final_osp_lt",
    "final_scl_medial",
    "final_scl_lateral",
    "confidence_mf",
    "confidence_lf",
    "confidence_mt",
    "confidence_lt",
    "scl_confidence_med",
    "scl_confidence_lat",
    "notes",
]


def _copy_images_enabled() -> bool:
    value = str(os.getenv("ANNOTATION_PACKAGE_COPY_IMAGES", "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def _package_image_filename(row, src: Path) -> str:
    image_id = str(getattr(row, "image_id", src.stem)).replace(".png", "")
    split = str(getattr(row, "split", "unknown"))
    kl_grade = str(getattr(row, "kl_grade", "na"))
    return f"{split}_{kl_grade}_{image_id}{src.suffix}"


def _histogram_clip(image: np.ndarray, low_pct: int = 5, high_pct: int = 99) -> np.ndarray:
    low = np.percentile(image, low_pct)
    high = np.percentile(image, high_pct)
    if high - low < 1e-6:
        return image.astype(np.uint8)
    clipped = np.clip(image, low, high)
    return ((clipped - low) / (high - low) * 255).astype(np.uint8)


def _preprocess_annotation_image(
    src: Path,
    dst: Path,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    low_pct: int = 5,
    high_pct: int = 99,
):
    image = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image for preprocessing: {src}")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    proc = clahe.apply(image)
    proc = _histogram_clip(proc, low_pct=low_pct, high_pct=high_pct)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), proc)


def _resolve_local_source_paths(
    rows: pd.DataFrame,
    *,
    project_root: Path,
    data_root: Path,
) -> list[str]:
    paths: list[str] = []
    for row in rows.itertuples(index=False):
        src = resolve_manifest_path(row.path, project_root=project_root, data_root=data_root)
        if not src.exists():
            raise FileNotFoundError(f"Manifest entry points to a missing image: {row.path} -> {src}")
        paths.append(str(src))
    return paths


def _prepare_feature_review_package(
    annotation_dir: Path,
    package_dir: Path,
    *,
    project_root: Path,
    data_root: Path,
    copy_images: bool,
):
    manifest_path = annotation_dir / "manifests" / "feature_grading_manifest.csv"
    osp_path = annotation_dir / "osteophyte_labels.csv"
    scl_path = annotation_dir / "sclerosis_labels.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not osp_path.exists() or not scl_path.exists():
        raise FileNotFoundError("Run bootstrap_pseudo_labels.py before preparing the annotation workspace.")

    manifest = pd.read_csv(manifest_path)
    osp = pd.read_csv(osp_path)[[
        "image_id",
        "osp_mf", "osp_lf", "osp_mt", "osp_lt",
        "confidence_mf", "confidence_lf", "confidence_mt", "confidence_lt",
    ]]
    scl = pd.read_csv(scl_path)[[
        "image_id",
        "scl_medial", "scl_lateral",
        "scl_confidence_med", "scl_confidence_lat",
    ]]

    review = manifest.merge(osp, on="image_id", how="left").merge(scl, on="image_id", how="left")
    review = review.rename(columns={
        "osp_mf": "suggestion_osp_mf",
        "osp_lf": "suggestion_osp_lf",
        "osp_mt": "suggestion_osp_mt",
        "osp_lt": "suggestion_osp_lt",
        "scl_medial": "suggestion_scl_medial",
        "scl_lateral": "suggestion_scl_lateral",
    })

    image_dir = package_dir / "feature_grading" / "images"
    if copy_images:
        _reset_dir(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
    else:
        _reset_dir(image_dir)

    resolved_paths = _resolve_local_source_paths(review, project_root=project_root, data_root=data_root)
    if copy_images:
        local_paths = []
        for row, resolved_path in zip(review.itertuples(index=False), resolved_paths):
            src = Path(resolved_path)
            dst = image_dir / _package_image_filename(row, src)
            _link_or_copy(src, dst)
            local_paths.append(str(dst))
    else:
        local_paths = resolved_paths
    review["local_image_path"] = local_paths

    for col in FEATURE_REVIEW_COLUMNS:
        if col not in review.columns:
            review[col] = ""

    review = review[FEATURE_REVIEW_COLUMNS]
    out_csv = package_dir / "feature_grading" / "feature_review_template.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(out_csv, index=False)

    inter_rater = annotation_dir / "manifests" / "inter_rater_subset.csv"
    if inter_rater.exists():
        shutil.copy2(inter_rater, package_dir / "feature_grading" / "inter_rater_subset.csv")

    return out_csv, len(review)


def _prepare_jsn_package(
    annotation_dir: Path,
    package_dir: Path,
    preprocessing_cfg,
    *,
    project_root: Path,
    data_root: Path,
    copy_images: bool,
):
    manifest_path = annotation_dir / "manifests" / "jsn_contour_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing JSN manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    image_dir = package_dir / "jsn_contours" / "images"
    preprocessed_image_dir = package_dir / "jsn_contours" / "images_preprocessed"
    prefixed_image_dir = package_dir / "jsn_contours" / "images_split_grade"

    if copy_images:
        for path in (image_dir, preprocessed_image_dir, prefixed_image_dir):
            _reset_dir(path)
            path.mkdir(parents=True, exist_ok=True)
    else:
        for path in (image_dir, preprocessed_image_dir, prefixed_image_dir):
            _reset_dir(path)

    resolved_paths = _resolve_local_source_paths(manifest, project_root=project_root, data_root=data_root)
    local_paths: list[str] = []
    preprocessed_local_paths: list[str] = []
    prefixed_local_paths: list[str] = []

    for row, resolved_path in zip(manifest.itertuples(index=False), resolved_paths):
        src = Path(resolved_path)
        local_paths.append(str(src))

        if copy_images:
            base_name = _package_image_filename(row, src)
            dst = image_dir / base_name
            _link_or_copy(src, dst)
            local_paths[-1] = str(dst)

            preprocessed_dst = preprocessed_image_dir / base_name
            _preprocess_annotation_image(
                src,
                preprocessed_dst,
                clip_limit=getattr(preprocessing_cfg.clahe, "clip_limit", 3.0),
                tile_grid_size=tuple(getattr(preprocessing_cfg.clahe, "tile_grid_size", [8, 8])),
                low_pct=getattr(preprocessing_cfg.histogram_clip, "low_percentile", 5),
                high_pct=getattr(preprocessing_cfg.histogram_clip, "high_percentile", 99),
            )
            preprocessed_local_paths.append(str(preprocessed_dst))

            prefixed_dst = prefixed_image_dir / base_name
            shutil.copy2(src, prefixed_dst)
            prefixed_local_paths.append(str(prefixed_dst))
        else:
            preprocessed_local_paths.append("")
            prefixed_local_paths.append("")

    manifest["local_image_path"] = local_paths
    manifest["preprocessed_local_image_path"] = preprocessed_local_paths
    manifest["prefixed_local_image_path"] = prefixed_local_paths

    package_root = package_dir / "jsn_contours"
    out_csv = package_root / "jsn_contour_manifest.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_csv, index=False)
    manifest[["image_id", "split", "kl_grade", "path", "local_image_path"]].to_csv(
        package_root / "jsn_contour_image_map.csv",
        index=False,
    )

    readme_lines = [
        "# JSN Annotation Package",
        "",
        "Use `jsn_contour_manifest.csv` as the source table for annotation import.",
        "The `path` column is repo-local and portable; `local_image_path` resolves to this machine.",
        "",
    ]
    if copy_images:
        manifest[["image_id", "split", "kl_grade", "path", "prefixed_local_image_path"]].to_csv(
            package_root / "jsn_contour_prefixed_image_map.csv",
            index=False,
        )
        manifest[["image_id", "split", "kl_grade", "path", "preprocessed_local_image_path"]].to_csv(
            package_root / "jsn_contour_preprocessed_image_map.csv",
            index=False,
        )
        readme_lines.extend([
            "Copied annotation folders are available when `ANNOTATION_PACKAGE_COPY_IMAGES=1`:",
            "- `images/`",
            "- `images_preprocessed/`",
            "- `images_split_grade/`",
            "- copied filenames use `split_grade_imageid.png` to avoid collisions",
            "",
        ])
    else:
        for stale_map in (
            package_root / "jsn_contour_prefixed_image_map.csv",
            package_root / "jsn_contour_preprocessed_image_map.csv",
        ):
            if stale_map.exists():
                stale_map.unlink()
        readme_lines.extend([
            "No duplicate image folders were created.",
            "If you later want copied package images, rerun with `ANNOTATION_PACKAGE_COPY_IMAGES=1`.",
            "",
        ])

    readme_lines.extend([
        "Trace two polylines per image:",
        "- `femoral_surface`",
        "- `tibial_surface`",
        "",
        "Export the reviewed result as COCO JSON and place it at:",
        "`annotations/reviewed/jsn_cvat_export.json`",
    ])
    (package_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    return out_csv, len(manifest)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    annotation_dir = Path(cfg.annotation_dir)
    package_dir = annotation_dir / "packages"
    package_dir.mkdir(parents=True, exist_ok=True)

    copy_images = _copy_images_enabled()
    project_root = Path(cfg.project_root)
    data_root = Path(cfg.data.root)

    feature_csv, feature_count = _prepare_feature_review_package(
        annotation_dir,
        package_dir,
        project_root=project_root,
        data_root=data_root,
        copy_images=copy_images,
    )
    jsn_csv, jsn_count = _prepare_jsn_package(
        annotation_dir,
        package_dir,
        cfg.preprocessing,
        project_root=project_root,
        data_root=data_root,
        copy_images=copy_images,
    )

    print("Prepared annotation workspace:")
    print(f"  Feature grading package: {feature_csv} ({feature_count} images)")
    print(f"  JSN contour package:     {jsn_csv} ({jsn_count} images)")
    print(f"  Root package dir:        {package_dir}")
    print(f"  Copied image folders:    {'enabled' if copy_images else 'disabled'}")


if __name__ == "__main__":
    main()
