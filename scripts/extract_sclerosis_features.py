"""Extract sclerosis features from subchondral ROIs using JSN masks + texture analysis."""

import csv
from collections import Counter
from pathlib import Path

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.timed_image_reader import TimedImageReader
from src.data.transforms import get_eval_transforms
from src.features.bootstrap_heuristics import (
    build_sclerosis_features,
    extract_geometric_subchondral_roi,
)
from src.features.subchondral_roi import extract_subchondral_roi_with_boxes_and_source
from src.features.texture_features import extract_all_texture_features
from src.models.sclerosis_classifier import SclerosisClassifier
from src.utils.annotation_paths import resolve_annotation_csv, select_label_subset
from src.utils.annotation_confidence import normalize_confidence
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_device, clear_memory
from src.utils.seed import seed_everything
from src.utils.feature_scaling import load_standardizer, transform_with_standardizer


SEPARATE_STRATEGIES = {"separate", "separate_by_side", "per_side", "side_specific"}
PROGRESS_FIELDNAMES = [
    "image_id",
    "side",
    "roi_path",
    "side_id",
    "grade",
    "label_source",
    "confidence_level",
    "roi_source",
]
FAILURE_FIELDNAMES = [
    "image_id",
    "split",
    "reason",
    "path",
    "reader_status",
    "exists",
    "is_file",
    "size_bytes",
    "detail",
]


def _path_diagnostics(path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "path": "",
            "exists": "",
            "is_file": "",
            "size_bytes": "",
        }

    exists = path.exists()
    is_file = path.is_file()
    size_bytes: int | str = ""
    if exists:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = ""
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": size_bytes,
    }


def _write_failure(
    writer: csv.DictWriter,
    handle,
    image_id: str,
    split: str,
    reason: str,
    path: Path | None = None,
    reader_status: str = "",
    detail: str = "",
) -> None:
    writer.writerow({
        "image_id": image_id,
        "split": split,
        "reason": reason,
        "reader_status": reader_status,
        "detail": detail,
        **_path_diagnostics(path),
    })
    handle.flush()


def _strategy_is_separate(cfg: DictConfig, checkpoint_root: Path) -> bool:
    strategy = str(getattr(cfg.training, "sclerosis_strategy", "separate")).lower()
    if strategy in SEPARATE_STRATEGIES:
        return True
    if strategy == "shared":
        return False
    return (checkpoint_root / "sclerosis_medial").exists() and (checkpoint_root / "sclerosis_lateral").exists()


def _load_classifier_entry(
    cfg: DictConfig,
    checkpoint_path: Path,
    standardizer_path: Path | None,
    device,
):
    from omegaconf import OmegaConf
    model_cfg = cfg.get("model", None)
    if model_cfg is None:
        model_cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "configs" / "model" / "sclerosis_hybrid.yaml")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)

    # Detect checkpoint architecture: shared classifier vs side-specific heads
    has_shared_classifier = any(k.startswith("classifier.") for k in state_dict)
    has_side_heads = any(k.startswith("medial_head.") for k in state_dict)
    if has_shared_classifier and not has_side_heads:
        model_cfg = OmegaConf.merge(model_cfg, {"use_side_specific_heads": False})
    elif has_side_heads and not has_shared_classifier:
        model_cfg = OmegaConf.merge(model_cfg, {"use_side_specific_heads": True})

    classifier = SclerosisClassifier(model_cfg)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()
    texture_standardizer = load_standardizer(standardizer_path) if standardizer_path is not None else None
    return {
        "model": classifier,
        "standardizer": texture_standardizer,
        "checkpoint_path": str(checkpoint_path),
    }


def _resolve_classifier_bundle(cfg: DictConfig, output_dir: Path, device) -> dict[str, dict]:
    checkpoint_root = Path(cfg.checkpoint_dir)
    checkpoint_path = getattr(cfg, "checkpoint_path", None)
    checkpoint_monitor = getattr(cfg, "checkpoint_monitor", None)
    if checkpoint_monitor in (None, "", "null"):
        checkpoint_monitor = getattr(cfg.training, "sclerosis_primary_monitor", "val_f1_macro")
    checkpoint_mode = getattr(cfg, "checkpoint_mode", None)
    if checkpoint_mode in (None, "", "null"):
        checkpoint_mode = getattr(cfg.training, "sclerosis_primary_mode", "max")

    bundle: dict[str, dict] = {}
    if checkpoint_path:
        ckpt_path = Path(str(checkpoint_path))
        try:
            bundle["shared"] = _load_classifier_entry(
                cfg,
                ckpt_path,
                output_dir / "texture_standardizer.npz",
                device,
            )
        except RuntimeError as exc:
            print(
                "Warning: skipping optional sclerosis classifier load because the "
                f"checkpoint is incompatible with the current model definition: {exc}"
            )
        return bundle

    if _strategy_is_separate(cfg, checkpoint_root):
        for side_name in ("medial", "lateral"):
            ckpt_dir = checkpoint_root / f"sclerosis_{side_name}"
            ckpt_path = find_best_lightning_checkpoint(
                ckpt_dir,
                monitor=str(checkpoint_monitor),
                mode=str(checkpoint_mode),
            ) if ckpt_dir.exists() else None
            if ckpt_path is None:
                continue
            try:
                bundle[side_name] = _load_classifier_entry(
                    cfg,
                    ckpt_path,
                    output_dir / f"texture_standardizer_{side_name}.npz",
                    device,
                )
            except RuntimeError as exc:
                print(
                    f"Warning: skipping optional sclerosis {side_name} classifier load because "
                    f"the checkpoint is incompatible with the current model definition: {exc}"
                )
        if bundle:
            return bundle

    ckpt_dir = checkpoint_root / "sclerosis"
    ckpt_path = find_best_lightning_checkpoint(
        ckpt_dir,
        monitor=str(checkpoint_monitor),
        mode=str(checkpoint_mode),
    ) if ckpt_dir.exists() else None
    if ckpt_path is None:
        return bundle
    try:
        bundle["shared"] = _load_classifier_entry(
            cfg,
            ckpt_path,
            output_dir / "texture_standardizer.npz",
            device,
        )
    except RuntimeError as exc:
        print(
            "Warning: skipping optional sclerosis classifier load because the "
            f"checkpoint is incompatible with the current model definition: {exc}"
        )
    return bundle


def _load_roi_patch(path: Path, output_size: int) -> np.ndarray | None:
    roi = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if roi is None or roi.size == 0:
        return None
    if roi.shape != (output_size, output_size):
        roi = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return roi


def _extract_patch_pairs(patch_dir: Path) -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    if not patch_dir.exists():
        return pairs

    for patch_path in sorted(patch_dir.glob("*.png")):
        stem = patch_path.stem
        if stem.endswith("_medial"):
            image_id = stem[:-7]
            side = "medial"
        elif stem.endswith("_lateral"):
            image_id = stem[:-8]
            side = "lateral"
        else:
            continue
        pairs.setdefault(image_id, {})[side] = patch_path
    return pairs


def _load_progress_lookup(progress_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not progress_path.exists():
        return {}

    lookup: dict[tuple[str, str], dict[str, str]] = {}
    with progress_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            side = str(row.get("side", "")).strip()
            if not image_id or side not in {"medial", "lateral"}:
                continue
            lookup[(image_id, side)] = row
    return lookup


def _write_progress_snapshot(progress_path: Path, progress_rows: list[dict[str, object]]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDNAMES)
        writer.writeheader()
        for row in progress_rows:
            writer.writerow(row)


def _resolve_grade_metadata(
    side: str,
    roi: np.ndarray,
    medial_roi: np.ndarray,
    lateral_roi: np.ndarray,
    label_row: dict | None,
    classifier_bundle: dict[str, dict],
    transform,
    device,
) -> tuple[int, str, str]:
    side_id = 0 if side == "medial" else 1
    label_key = f"scl_{side}"
    label_source_key = f"label_source_{side}"
    confidence_key = "scl_confidence_med" if side == "medial" else "scl_confidence_lat"
    has_manual_label = (
        label_row is not None
        and label_key in label_row
        and pd.notna(label_row[label_key])
        and str(label_row[label_key]).strip() != ""
    )
    if has_manual_label:
        grade = int(label_row[label_key])
        label_source_value = str(label_row.get(label_source_key, label_row.get("label_source", "manual_review")))
        confidence_value = normalize_confidence(label_row.get(confidence_key, "high"))
        return grade, label_source_value, confidence_value

    classifier_entry = classifier_bundle.get(side) or classifier_bundle.get("shared")
    if classifier_entry is not None:
        transformed = transform(image=roi)
        roi_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(device)
        tex_array = np.asarray(extract_all_texture_features(roi), dtype=np.float64)
        if classifier_entry["standardizer"] is not None:
            tex_array = transform_with_standardizer(
                tex_array[None, :],
                classifier_entry["standardizer"][0],
                classifier_entry["standardizer"][1],
            )[0]
        tex_tensor = torch.tensor(tex_array, dtype=torch.float32).unsqueeze(0).to(device)
        side_tensor = torch.tensor([side_id], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = classifier_entry["model"](roi_tensor, tex_tensor, side_tensor)
            grade = logits.argmax(dim=1).item()
        return int(grade), "model_prediction", "high"

    feature_dict = build_sclerosis_features(
        medial_roi if side == "medial" else None,
        lateral_roi if side == "lateral" else None,
    )
    grade = feature_dict[f"scl_grade_{side}"]
    return int(grade), "heuristic_image_only", "high"


def _append_image_records(
    state: dict[str, object],
    output_dir: Path,
    split: str,
    image_id: str,
    medial_roi: np.ndarray,
    lateral_roi: np.ndarray,
    roi_source: str,
    label_map: dict[str, dict],
    classifier_bundle: dict[str, dict],
    transform,
    device,
    progress_lookup: dict[tuple[str, str], dict[str, str]] | None = None,
    save_patches: bool = True,
) -> list[dict[str, object]]:
    label_row = label_map.get(image_id)
    image_level_feature_dict = build_sclerosis_features(medial_roi, lateral_roi)
    current_progress_rows: list[dict[str, object]] = []
    predicted_grade_by_side: dict[str, int] = {}

    for side, roi in (("medial", medial_roi), ("lateral", lateral_roi)):
        side_id = 0 if side == "medial" else 1
        roi_save_path = output_dir / "patches" / split / f"{image_id}_{side}.png"
        roi_save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_patches:
            cv2.imwrite(str(roi_save_path), roi)

        tex_feats = extract_all_texture_features(roi)
        progress_row = None if progress_lookup is None else progress_lookup.get((image_id, side))
        grade = None
        label_source_value = None
        confidence_value = "high"
        current_roi_source = roi_source

        if progress_row is not None:
            try:
                grade = int(progress_row["grade"])
                label_source_value = str(progress_row.get("label_source", "") or "manual_review")
                confidence_value = normalize_confidence(progress_row.get("confidence_level", "high"))
                current_roi_source = str(progress_row.get("roi_source", "") or roi_source)
            except (TypeError, ValueError):
                grade = None

        if grade is None:
            grade, label_source_value, confidence_value = _resolve_grade_metadata(
                side,
                roi,
                medial_roi,
                lateral_roi,
                label_row,
                classifier_bundle,
                transform,
                device,
            )

        state["texture_vectors"].append(np.asarray(tex_feats, dtype=np.float64))
        state["roi_paths"].append(str(roi_save_path))
        state["side_ids"].append(side_id)
        state["image_ids"].append(f"{image_id}_{side}")
        state["roi_sources"].append(current_roi_source)
        state["grades"].append(int(grade))
        state["label_sources"].append(label_source_value)
        state["confidence_levels"].append(confidence_value)
        predicted_grade_by_side[side] = int(grade)

        progress_row_out = {
            "image_id": image_id,
            "side": side,
            "roi_path": str(roi_save_path),
            "side_id": side_id,
            "grade": int(grade),
            "label_source": label_source_value,
            "confidence_level": confidence_value,
            "roi_source": current_roi_source,
        }
        current_progress_rows.append(progress_row_out)

    if "medial" in predicted_grade_by_side:
        image_level_feature_dict["scl_grade_medial"] = predicted_grade_by_side["medial"]
    if "lateral" in predicted_grade_by_side:
        image_level_feature_dict["scl_grade_lateral"] = predicted_grade_by_side["lateral"]
    state["image_level_features"][image_id] = image_level_feature_dict
    state["processed_image_ids"].add(image_id)
    state["progress_rows"].extend(current_progress_rows)
    return current_progress_rows


def _empty_split_state() -> dict[str, object]:
    return {
        "roi_paths": [],
        "texture_vectors": [],
        "side_ids": [],
        "grades": [],
        "image_ids": [],
        "label_sources": [],
        "confidence_levels": [],
        "roi_sources": [],
        "image_level_features": {},
        "processed_image_ids": set(),
        "progress_rows": [],
    }


def _restore_split_state_from_patches(
    output_dir: Path,
    split: str,
    label_map: dict[str, dict],
    classifier_bundle: dict[str, dict],
    transform,
    device,
    roi_output_size: int,
    use_progress_metadata: bool = True,
) -> tuple[dict[str, object], dict[str, int], Path]:
    state = _empty_split_state()
    progress_path = output_dir / f"{split}_sclerosis_progress.csv"
    progress_lookup = _load_progress_lookup(progress_path) if use_progress_metadata else {}
    patch_pairs = _extract_patch_pairs(output_dir / "patches" / split)
    reused_from_progress = 0
    reused_from_patch_only = 0

    for image_id, side_paths in sorted(patch_pairs.items()):
        if "medial" not in side_paths or "lateral" not in side_paths:
            continue

        medial_roi = _load_roi_patch(side_paths["medial"], roi_output_size)
        lateral_roi = _load_roi_patch(side_paths["lateral"], roi_output_size)
        if medial_roi is None or lateral_roi is None:
            continue

        has_progress_rows = (
            (image_id, "medial") in progress_lookup
            and (image_id, "lateral") in progress_lookup
        )
        _append_image_records(
            state,
            output_dir=output_dir,
            split=split,
            image_id=image_id,
            medial_roi=medial_roi,
            lateral_roi=lateral_roi,
            roi_source="resume_existing_patch",
            label_map=label_map,
            classifier_bundle=classifier_bundle,
            transform=transform,
            device=device,
            progress_lookup=progress_lookup,
            save_patches=False,
        )
        if has_progress_rows:
            reused_from_progress += 1
        else:
            reused_from_patch_only += 1

    if state["progress_rows"] and (not progress_path.exists() or reused_from_patch_only > 0):
        _write_progress_snapshot(progress_path, state["progress_rows"])

    stats = {
        "reused_images": len(state["processed_image_ids"]),
        "from_progress": reused_from_progress,
        "from_patch_only": reused_from_patch_only,
    }
    return state, stats, progress_path


def _save_split_outputs(output_dir: Path, split: str, state: dict[str, object]) -> None:
    np.savez(
        str(output_dir / f"{split}_sclerosis_data.npz"),
        roi_paths=np.asarray(state["roi_paths"], dtype=object),
        texture_features=np.asarray(state["texture_vectors"], dtype=np.float64),
        side_ids=np.asarray(state["side_ids"], dtype=np.int64),
        grades=np.asarray(state["grades"], dtype=np.int64),
        image_ids=np.asarray(state["image_ids"], dtype=object),
        label_sources=np.asarray(state["label_sources"], dtype=object),
        confidence_levels=np.asarray(state["confidence_levels"], dtype=object),
        roi_sources=np.asarray(state["roi_sources"], dtype=object),
    )

    scalar_features = {}
    lbp_histograms = {}
    for image_id_key, feat_dict in state["image_level_features"].items():
        scalar_features[image_id_key] = np.array([
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
        lbp_histograms[image_id_key] = np.concatenate([
            np.asarray(feat_dict.get("scl_lbp_hist_med", np.zeros(54)), dtype=np.float64),
            np.asarray(feat_dict.get("scl_lbp_hist_lat", np.zeros(54)), dtype=np.float64),
        ])

    np.savez(str(output_dir / f"{split}_sclerosis_features.npz"), **scalar_features)
    np.savez(str(output_dir / f"{split}_sclerosis_lbp_histograms.npz"), **lbp_histograms)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()

    feature_dir = Path(cfg.feature_dir)
    jsn_mask_dir = feature_dir / "jsn" / "masks"
    output_dir = Path(str(getattr(cfg, "sclerosis_output_dir", feature_dir / "sclerosis")))
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log_path = output_dir / "sclerosis_extraction_failures.csv"
    sclerosis_roi_cfg = getattr(cfg.preprocessing, "sclerosis_roi", {})
    image_read_timeout_seconds = float(getattr(cfg.preprocessing, "image_read_timeout_seconds", 5.0))
    roi_min_depth_px = int(getattr(sclerosis_roi_cfg, "min_depth_px", 10))
    roi_depth_fraction = float(getattr(sclerosis_roi_cfg, "depth_fraction", 0.10))
    roi_medial_depth_fraction = float(getattr(sclerosis_roi_cfg, "medial_depth_fraction", roi_depth_fraction))
    roi_lateral_depth_fraction = float(getattr(sclerosis_roi_cfg, "lateral_depth_fraction", roi_depth_fraction))
    roi_offset_pct = float(getattr(sclerosis_roi_cfg, "offset_pct", 0.10))
    roi_medial_offset_pct = float(getattr(sclerosis_roi_cfg, "medial_offset_pct", roi_offset_pct))
    roi_lateral_offset_pct = float(getattr(sclerosis_roi_cfg, "lateral_offset_pct", roi_offset_pct))
    roi_medial_inner_offset_pct = getattr(sclerosis_roi_cfg, "medial_inner_offset_pct", None)
    roi_medial_outer_offset_pct = getattr(sclerosis_roi_cfg, "medial_outer_offset_pct", None)
    roi_lateral_inner_offset_pct = getattr(sclerosis_roi_cfg, "lateral_inner_offset_pct", None)
    roi_lateral_outer_offset_pct = getattr(sclerosis_roi_cfg, "lateral_outer_offset_pct", None)
    roi_surface_offset_fraction = float(getattr(sclerosis_roi_cfg, "surface_offset_fraction", 0.015))
    roi_surface_smoothing_window = int(getattr(sclerosis_roi_cfg, "surface_smoothing_window", 7))
    roi_output_size = int(getattr(sclerosis_roi_cfg, "output_size", 64))

    transform = get_eval_transforms(cfg)
    data_root = Path(cfg.data.root)
    label_mode = getattr(cfg.training, "label_mode", "manual")
    allow_bootstrap_fallback = bool(getattr(cfg.training, "allow_bootstrap_fallback", False))
    label_csv = resolve_annotation_csv(
        cfg.annotation_dir,
        "sclerosis_labels",
        mode=label_mode,
        allow_bootstrap_fallback=allow_bootstrap_fallback,
    )
    label_map = {}
    label_df = pd.read_csv(label_csv)
    label_df, subset_mode = select_label_subset(
        label_df,
        mode=label_mode,
        allow_bootstrap_fallback=allow_bootstrap_fallback,
    )
    for _, row in label_df.iterrows():
        image_id = str(row["image_id"]).replace(".png", "")
        label_map[image_id] = row.to_dict()
    print(f"Loaded sclerosis labels from {label_csv} ({len(label_map)} image rows, mode={subset_mode})")
    print(f"Sclerosis output dir: {output_dir}")

    classifier_bundle = {}
    if str(label_mode).lower() == "manual":
        print(
            "Manual sclerosis extraction: skipping optional classifier bundle; "
            "non-reviewed rows will keep heuristic_image_only metadata."
        )
    else:
        classifier_bundle = _resolve_classifier_bundle(cfg, output_dir, device)

    with TimedImageReader(image_read_timeout_seconds) as image_reader:
        with open(failure_log_path, "w", newline="", encoding="utf-8") as failure_handle:
            failure_writer = csv.DictWriter(
                failure_handle,
                fieldnames=FAILURE_FIELDNAMES,
            )
            failure_writer.writeheader()

            for split in ["train", "val", "test"]:
                split_dir = data_root / split
                if not split_dir.exists():
                    continue

                state, resume_stats, progress_path = _restore_split_state_from_patches(
                    output_dir=output_dir,
                    split=split,
                    label_map=label_map,
                    classifier_bundle=classifier_bundle,
                    transform=transform,
                    device=device,
                    roi_output_size=roi_output_size,
                    use_progress_metadata=str(label_mode).lower() != "manual",
                )
                processed_image_ids = state["processed_image_ids"]
                skipped_images = 0

                if resume_stats["reused_images"]:
                    print(
                        f"Resuming sclerosis {split}: reused {resume_stats['reused_images']} complete images "
                        f"({resume_stats['from_progress']} from progress metadata, "
                        f"{resume_stats['from_patch_only']} from patch-only recovery)"
                    )

                progress_exists = progress_path.exists()
                with progress_path.open("a", newline="", encoding="utf-8") as progress_handle:
                    progress_writer = csv.DictWriter(progress_handle, fieldnames=PROGRESS_FIELDNAMES)
                    if not progress_exists:
                        progress_writer.writeheader()
                        progress_handle.flush()

                    for grade_dir in sorted(split_dir.iterdir()):
                        if not grade_dir.is_dir():
                            continue

                        for img_path in tqdm(sorted(grade_dir.glob("*.png")),
                                             desc=f"Sclerosis {split}/{grade_dir.name}"):
                            image_id = img_path.stem
                            if image_id in processed_image_ids:
                                continue

                            image = image_reader.read(img_path)
                            if image is None:
                                _write_failure(
                                    failure_writer,
                                    failure_handle,
                                    image_id=image_id,
                                    split=split,
                                    reason="image_load_failed",
                                    path=img_path,
                                    reader_status=getattr(image_reader, "last_status", ""),
                                )
                                skipped_images += 1
                                continue

                            mask_path = jsn_mask_dir / f"{image_id}_mask.npy"
                            if mask_path.exists():
                                try:
                                    jsn_mask = np.load(str(mask_path))
                                except Exception as exc:
                                    _write_failure(
                                        failure_writer,
                                        failure_handle,
                                        image_id=image_id,
                                        split=split,
                                        reason="jsn_mask_load_failed",
                                        path=mask_path,
                                        detail=type(exc).__name__,
                                    )
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
                            except Exception as exc:
                                _write_failure(
                                    failure_writer,
                                    failure_handle,
                                    image_id=image_id,
                                    split=split,
                                    reason="subchondral_roi_failed",
                                    path=img_path,
                                    detail=type(exc).__name__,
                                )
                                medial_roi, lateral_roi = None, None
                                roi_source = "jsn_exception"
                            if medial_roi is None or lateral_roi is None:
                                medial_roi, lateral_roi = extract_geometric_subchondral_roi(image, is_left=is_left)
                                roi_source = "geometric_fallback"
                            if medial_roi is not None and medial_roi.shape != (roi_output_size, roi_output_size):
                                medial_roi = cv2.resize(
                                    medial_roi,
                                    (roi_output_size, roi_output_size),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            if lateral_roi is not None and lateral_roi.shape != (roi_output_size, roi_output_size):
                                lateral_roi = cv2.resize(
                                    lateral_roi,
                                    (roi_output_size, roi_output_size),
                                    interpolation=cv2.INTER_LINEAR,
                                )

                            current_progress_rows = _append_image_records(
                                state,
                                output_dir=output_dir,
                                split=split,
                                image_id=image_id,
                                medial_roi=medial_roi,
                                lateral_roi=lateral_roi,
                                roi_source=roi_source,
                                label_map=label_map,
                                classifier_bundle=classifier_bundle,
                                transform=transform,
                                device=device,
                                progress_lookup=None,
                                save_patches=True,
                            )
                            for row in current_progress_rows:
                                progress_writer.writerow(row)
                            progress_handle.flush()

                _save_split_outputs(output_dir, split, state)
                print(
                    f"Saved sclerosis data for {len(state['roi_paths'])} ROIs ({split}); "
                    f"skipped unreadable/failed images: {skipped_images}"
                )
                if state["roi_sources"]:
                    print(f"Sclerosis ROI sources ({split}): {dict(Counter(state['roi_sources']))}")
                clear_memory()

    print(f"Sclerosis extraction failure log: {failure_log_path}")


if __name__ == "__main__":
    main()
