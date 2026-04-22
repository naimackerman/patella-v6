"""Generate high-confidence pseudo-label expansions from trained stage models."""

from __future__ import annotations

from math import ceil
from pathlib import Path
import sys

import cv2
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.transforms import get_eval_transforms
from src.features.bootstrap_heuristics import ROI_SITES
from src.models.osteophyte_grader import OsteophyteGrader
from src.models.sclerosis_classifier import SclerosisClassifier
from src.utils.annotation_paths import MANUAL_SOURCES, resolve_annotation_csv
from src.utils.checkpoints import (
    extract_model_state_dict,
    find_best_lightning_checkpoint,
    load_checkpoint,
    resolve_osteophyte_checkpoint_paths,
)
from src.utils.device import get_device
from src.utils.feature_scaling import load_standardizer, transform_with_standardizer
from src.utils.seed import seed_everything
from src.utils.sclerosis_labels import apply_sclerosis_label_scheme_to_cfg


SITE_ABBREV = {
    "medial_femur": "mf",
    "lateral_femur": "lf",
    "medial_tibia": "mt",
    "lateral_tibia": "lt",
}

SEPARATE_STRATEGIES = {"separate", "separate_by_side", "per_side", "side_specific"}
OSTEOPHYTE_PSEUDO_COLUMNS = [
    "image_id",
    "split",
    "pseudo_label",
    "needs_review",
    "label_source",
    "confidence_mf",
    "osp_mf",
    "label_source_mf",
    "confidence_lf",
    "osp_lf",
    "label_source_lf",
    "confidence_mt",
    "osp_mt",
    "label_source_mt",
    "confidence_lt",
    "osp_lt",
    "label_source_lt",
]
SCLEROSIS_PSEUDO_COLUMNS = [
    "image_id",
    "split",
    "pseudo_label",
    "needs_review",
    "label_source",
    "scl_confidence_med",
    "scl_medial",
    "label_source_medial",
    "scl_confidence_lat",
    "scl_lateral",
    "label_source_lateral",
]


def _config_has_keys(model_cfg: DictConfig, required_keys: tuple[str, ...]) -> bool:
    if model_cfg is None:
        return False
    return all(key in model_cfg and model_cfg[key] is not None for key in required_keys)


def _resolve_osteophyte_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and _config_has_keys(cfg.model, ("backbone", "num_classes_per_head")):
        return cfg.model
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "se_resnet50.yaml")


def _resolve_sclerosis_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and _config_has_keys(cfg.model, ("cnn_backbone", "num_classes", "texture_feature_dim")):
        return cfg.model
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "sclerosis_hybrid.yaml")


def _resolve_binary_threshold(cfg: DictConfig) -> float | None:
    threshold = getattr(cfg.training, "sclerosis_binary_threshold", None)
    if threshold in (None, "", "null", "None"):
        return None
    return float(threshold)


def _load_manual_ids(annotation_dir: Path, stem: str) -> set[str]:
    label_path = resolve_annotation_csv(
        annotation_dir,
        stem,
        mode="manual",
        allow_bootstrap_fallback=False,
    )
    if not label_path.exists():
        return set()

    df = pd.read_csv(label_path)
    if "label_source" in df.columns:
        df = df[df["label_source"].isin(MANUAL_SOURCES)].copy()
    return {str(image_id).replace(".png", "") for image_id in df["image_id"].tolist()}


def _merge_expanded(manual_path: Path, pseudo_df: pd.DataFrame, output_path: Path):
    manual_df = pd.read_csv(manual_path) if manual_path.exists() else pd.DataFrame()
    if not manual_df.empty:
        manual_df["image_id"] = manual_df["image_id"].astype(str).str.replace(".png", "", regex=False)
        if "label_source" in manual_df.columns:
            manual_df = manual_df[manual_df["label_source"].isin(MANUAL_SOURCES)].copy()
    if not pseudo_df.empty:
        pseudo_df["image_id"] = pseudo_df["image_id"].astype(str).str.replace(".png", "", regex=False)

    if manual_df.empty:
        merged = pseudo_df
    elif pseudo_df.empty:
        merged = manual_df
    else:
        merged = pd.concat([
            manual_df,
            pseudo_df[~pseudo_df["image_id"].isin(set(manual_df["image_id"]))],
        ], ignore_index=True)
    merged.to_csv(output_path, index=False)
    return merged


def _resolve_allowed_roi_sources(cfg: DictConfig, split: str) -> list[str]:
    source_cfg = getattr(cfg.training, "sclerosis_roi_source_filter", {})
    values = getattr(source_cfg, split, [])
    return [str(item).strip() for item in list(values or []) if str(item).strip()]


def _iter_adaptive_thresholds(start: float, minimum: float, step: float) -> list[float]:
    start = float(start)
    minimum = float(minimum)
    step = float(step)
    if step <= 0:
        step = 0.05
    thresholds: list[float] = []
    current = start
    while current >= minimum - 1.0e-8:
        thresholds.append(round(current, 4))
        current -= step
    if not thresholds:
        thresholds = [round(minimum, 4)]
    elif thresholds[-1] > minimum:
        thresholds.append(round(minimum, 4))
    return thresholds


def _select_adaptive_dataframe(
    builder,
    base_threshold: float,
    min_threshold: float,
    step: float,
    target_rows: int,
) -> tuple[pd.DataFrame, float]:
    thresholds = _iter_adaptive_thresholds(base_threshold, min_threshold, step)
    selected_df = pd.DataFrame()
    selected_threshold = thresholds[-1]
    for threshold in thresholds:
        candidate_df = builder(float(threshold))
        selected_df = candidate_df
        selected_threshold = float(threshold)
        if target_rows <= 0 or len(candidate_df) >= target_rows:
            break
    return selected_df, selected_threshold


def _select_site_thresholds(
    candidates: list[dict[str, object]],
    base_threshold: float,
    min_threshold: float,
    step: float,
    target_rows_per_site: int,
) -> tuple[dict[str, float], dict[str, int]]:
    thresholds = _iter_adaptive_thresholds(base_threshold, min_threshold, step)
    selected_thresholds: dict[str, float] = {}
    selected_counts: dict[str, int] = {}
    for site in ROI_SITES:
        short = SITE_ABBREV[site]
        confidence_key = f"confidence_{short}"
        selected_threshold = thresholds[-1]
        selected_count = 0
        for threshold in thresholds:
            count = sum(
                1
                for candidate in candidates
                if confidence_key in candidate and float(candidate[confidence_key]) >= float(threshold)
            )
            selected_threshold = float(threshold)
            selected_count = int(count)
            if target_rows_per_site <= 0 or count >= target_rows_per_site:
                break
        selected_thresholds[site] = selected_threshold
        selected_counts[site] = selected_count
    return selected_thresholds, selected_counts


def _build_osteophyte_pseudo_df(
    candidates: list[dict[str, object]],
    threshold: float | None = None,
    thresholds_by_site: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    site_thresholds = thresholds_by_site or {site: float(threshold) for site in ROI_SITES}
    for candidate in candidates:
        row = {
            "image_id": candidate["image_id"],
            "split": "train",
            "pseudo_label": True,
            "needs_review": False,
            "label_source": "high_conf_model",
        }
        any_label = False
        for site in ROI_SITES:
            short = SITE_ABBREV[site]
            conf = float(candidate[f"confidence_{short}"])
            pred = int(candidate[f"pred_{short}"])
            row[f"confidence_{short}"] = conf
            if conf >= float(site_thresholds[site]):
                row[f"osp_{short}"] = pred
                row[f"label_source_{short}"] = "high_conf_model"
                any_label = True
            else:
                row[f"osp_{short}"] = np.nan
                row[f"label_source_{short}"] = "low_confidence_skip"
        if any_label:
            rows.append(row)
    return pd.DataFrame(rows, columns=OSTEOPHYTE_PSEUDO_COLUMNS)


def _build_sclerosis_pseudo_df(candidates: dict[str, dict[str, object]], threshold: float) -> pd.DataFrame:
    rows = []
    for image_id, candidate in candidates.items():
        row = {
            "image_id": image_id,
            "split": "train",
            "pseudo_label": True,
            "needs_review": False,
            "label_source": "high_conf_model",
        }
        any_label = False
        for side in ("medial", "lateral"):
            side_candidate = candidate.get(side)
            if side_candidate is None:
                continue
            conf = float(side_candidate["confidence"])
            pred = int(side_candidate["pred"])
            row[f"scl_confidence_{'med' if side == 'medial' else 'lat'}"] = conf
            if conf >= threshold:
                row[f"scl_{side}"] = pred
                row[f"label_source_{side}"] = "high_conf_model"
                any_label = True
        if any_label:
            rows.append(row)
    return pd.DataFrame(rows, columns=SCLEROSIS_PSEUDO_COLUMNS)


def _generate_osteophyte_pseudolabels(cfg: DictConfig, device) -> pd.DataFrame:
    annotation_dir = Path(cfg.annotation_dir)
    manual_ids = _load_manual_ids(annotation_dir, "osteophyte_labels")
    roi_root = Path(str(getattr(cfg, "osteophyte_roi_dir", Path(cfg.feature_dir) / "rois")))
    roi_dir = roi_root / "train"
    ckpt_dir = Path(cfg.checkpoint_dir) / "osteophyte"
    transform = get_eval_transforms(cfg)
    threshold = float(getattr(cfg.training, "pseudo_confidence_threshold_osteophyte", 0.90))
    min_threshold = float(getattr(cfg.training, "pseudo_confidence_threshold_osteophyte_min", threshold))
    threshold_step = float(getattr(cfg.training, "pseudo_confidence_threshold_step", 0.05))
    target_rows = int(getattr(cfg.training, "pseudo_target_rows_osteophyte", 250))
    target_rows_per_site = getattr(cfg.training, "pseudo_target_rows_osteophyte_per_site", None)
    if target_rows_per_site in (None, "", "null", "None"):
        target_rows_per_site = ceil(max(target_rows, 0) / max(len(ROI_SITES), 1))
    else:
        target_rows_per_site = int(target_rows_per_site)
    balance_by_site = bool(getattr(cfg.training, "pseudo_balance_osteophyte_by_site", True))

    if not ckpt_dir.exists() or not roi_dir.exists():
        return pd.DataFrame()

    override_cfg = getattr(cfg.training, "osteophyte_checkpoint_overrides", {})
    override_paths = {
        str(site_name): str(path_value)
        for site_name, path_value in dict(override_cfg).items()
        if path_value is not None
    } if override_cfg is not None else {}
    resolved_ckpts = resolve_osteophyte_checkpoint_paths(
        ckpt_dir,
        ROI_SITES,
        override_paths_by_site=override_paths,
    )
    if len(resolved_ckpts) != len(ROI_SITES):
        return pd.DataFrame()

    print(
        f"Osteophyte pseudo-labeling: scanning train ROI dir {roi_dir} "
        f"with {len(manual_ids)} manual IDs excluded.",
        flush=True,
    )
    for site in ROI_SITES:
        print(
            f"  teacher[{site}] = {resolved_ckpts[site]['path']}",
            flush=True,
        )

    model_cache: dict[str, OsteophyteGrader] = {}
    base_model_cfg = _resolve_osteophyte_model_cfg(cfg)

    def load_model(ckpt_path: Path) -> OsteophyteGrader:
        cache_key = str(ckpt_path)
        if cache_key not in model_cache:
            checkpoint = load_checkpoint(ckpt_path, map_location=device)
            state_dict = extract_model_state_dict(checkpoint)
            current_model_cfg = base_model_cfg
            has_old_heads = any(
                key.startswith("heads.") and key.count(".") == 2
                for key in state_dict
            )
            if has_old_heads:
                current_model_cfg = OmegaConf.merge(base_model_cfg, {"use_mlp_heads": False})
            model = OsteophyteGrader(current_model_cfg)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            model_cache[cache_key] = model
        return model_cache[cache_key]

    image_ids = sorted({
        path.name.rsplit("_", 2)[0]
        for path in roi_dir.glob("*.png")
        if any(path.name.endswith(f"_{site}.png") for site in ROI_SITES)
    })
    print(f"Osteophyte candidate train images: {len(image_ids)}", flush=True)
    candidates: list[dict[str, object]] = []
    for idx, image_id in enumerate(tqdm(image_ids, desc="Pseudo osteophyte", file=sys.stdout)):
        if image_id in manual_ids:
            continue

        candidate = {
            "image_id": image_id,
        }
        for site in ROI_SITES:
            roi_path = roi_dir / f"{image_id}_{site}.png"
            roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
            if roi is None:
                continue
            transformed = transform(image=roi)
            roi_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(device)
            model = load_model(Path(resolved_ckpts[site]["path"]))
            with torch.no_grad():
                logits = model.forward_single(roi_tensor, site)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, pred = probs.max(dim=0)
            short = SITE_ABBREV[site]
            candidate[f"confidence_{short}"] = float(conf.item())
            candidate[f"pred_{short}"] = int(pred.item())
        if all(f"confidence_{SITE_ABBREV[site]}" in candidate for site in ROI_SITES):
            candidates.append(candidate)
        if (idx + 1) % 500 == 0:
            print(
                f"Osteophyte pseudo-labeling progress: processed {idx + 1}/{len(image_ids)} images, "
                f"eligible candidates={len(candidates)}",
                flush=True,
            )

    if balance_by_site:
        thresholds_by_site, accepted_per_site = _select_site_thresholds(
            candidates,
            base_threshold=threshold,
            min_threshold=min_threshold,
            step=threshold_step,
            target_rows_per_site=target_rows_per_site,
        )
        pseudo_df = _build_osteophyte_pseudo_df(candidates, thresholds_by_site=thresholds_by_site)
        threshold_summary = ", ".join(
            f"{SITE_ABBREV[site]}={thresholds_by_site[site]:.2f}/{accepted_per_site[site]}"
            for site in ROI_SITES
        )
        print(
            "Osteophyte pseudo-labeling site thresholds "
            f"(target_per_site={target_rows_per_site}): {threshold_summary}",
            flush=True,
        )
        threshold_used = min(thresholds_by_site.values()) if thresholds_by_site else threshold
    else:
        pseudo_df, threshold_used = _select_adaptive_dataframe(
            lambda thr: _build_osteophyte_pseudo_df(candidates, thr),
            base_threshold=threshold,
            min_threshold=min_threshold,
            step=threshold_step,
            target_rows=target_rows,
        )
    out_dir = annotation_dir / "pseudo"
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_path = out_dir / "osteophyte_labels_high_conf.csv"
    pseudo_df.to_csv(pseudo_path, index=False)
    manual_path = resolve_annotation_csv(
        annotation_dir,
        "osteophyte_labels",
        mode="manual",
        allow_bootstrap_fallback=False,
    )
    merged = _merge_expanded(manual_path, pseudo_df, annotation_dir / "osteophyte_labels_expanded.csv")
    print(
        f"Saved osteophyte pseudo-labels: {pseudo_path} ({len(pseudo_df)} image rows, "
        f"threshold_used={threshold_used:.2f}, base_threshold={threshold:.2f})"
    )
    print(f"Saved expanded osteophyte labels: {annotation_dir / 'osteophyte_labels_expanded.csv'} ({len(merged)} rows)")
    return pseudo_df


def _generate_sclerosis_pseudolabels(cfg: DictConfig, device) -> pd.DataFrame:
    annotation_dir = Path(cfg.annotation_dir)
    manual_ids = _load_manual_ids(annotation_dir, "sclerosis_labels")
    sclerosis_dir = Path(str(getattr(cfg, "sclerosis_output_dir", Path(cfg.feature_dir) / "sclerosis")))
    scl_npz = sclerosis_dir / "train_sclerosis_data.npz"
    checkpoint_root = Path(cfg.checkpoint_dir)
    threshold = float(getattr(cfg.training, "pseudo_confidence_threshold_sclerosis", 0.90))
    min_threshold = float(getattr(cfg.training, "pseudo_confidence_threshold_sclerosis_min", threshold))
    threshold_step = float(getattr(cfg.training, "pseudo_confidence_threshold_step", 0.05))
    target_rows = int(getattr(cfg.training, "pseudo_target_rows_sclerosis", 250))
    transform = get_eval_transforms(cfg)

    if not scl_npz.exists():
        return pd.DataFrame()

    strategy = str(getattr(cfg.training, "sclerosis_strategy", "separate")).lower()
    explicit_shared_ckpt = getattr(cfg, "checkpoint_path", None)
    separate = strategy in SEPARATE_STRATEGIES or (
        (checkpoint_root / "sclerosis_medial").exists() and (checkpoint_root / "sclerosis_lateral").exists()
    )
    checkpoint_monitor = str(getattr(cfg.training, "sclerosis_primary_monitor", "val_f1_macro"))
    checkpoint_mode = str(getattr(cfg.training, "sclerosis_primary_mode", "max"))
    binary_threshold = _resolve_binary_threshold(cfg)
    base_model_cfg = _resolve_sclerosis_model_cfg(cfg)
    print(
        f"Sclerosis pseudo-labeling: scanning {scl_npz} with {len(manual_ids)} manual IDs excluded.",
        flush=True,
    )

    model_bundle: dict[str, dict] = {}
    if explicit_shared_ckpt not in (None, "", "null", "None"):
        ckpt_path = Path(str(explicit_shared_ckpt))
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        state_dict = extract_model_state_dict(checkpoint)
        current_model_cfg = base_model_cfg
        has_shared_classifier = any(k.startswith("classifier.") for k in state_dict)
        has_side_heads = any(k.startswith("medial_head.") for k in state_dict)
        if has_shared_classifier and not has_side_heads:
            current_model_cfg = OmegaConf.merge(base_model_cfg, {"use_side_specific_heads": False})
        elif has_side_heads and not has_shared_classifier:
            current_model_cfg = OmegaConf.merge(base_model_cfg, {"use_side_specific_heads": True})
        model = SclerosisClassifier(current_model_cfg)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        model_bundle["shared"] = {
            "model": model,
            "standardizer": load_standardizer(sclerosis_dir / "texture_standardizer.npz"),
        }
        print(f"  teacher[shared-explicit] = {ckpt_path}", flush=True)

    if separate and not model_bundle:
        for side_name in ("medial", "lateral"):
            ckpt_dir = checkpoint_root / f"sclerosis_{side_name}"
            ckpt_path = find_best_lightning_checkpoint(
                ckpt_dir,
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
            ) if ckpt_dir.exists() else None
            if ckpt_path is None:
                continue
            checkpoint = load_checkpoint(ckpt_path, map_location=device)
            state_dict = extract_model_state_dict(checkpoint)
            current_model_cfg = base_model_cfg
            has_shared_classifier = any(k.startswith("classifier.") for k in state_dict)
            has_side_heads = any(k.startswith("medial_head.") for k in state_dict)
            if has_shared_classifier and not has_side_heads:
                current_model_cfg = OmegaConf.merge(base_model_cfg, {"use_side_specific_heads": False})
            elif has_side_heads and not has_shared_classifier:
                current_model_cfg = OmegaConf.merge(base_model_cfg, {"use_side_specific_heads": True})
            model = SclerosisClassifier(current_model_cfg)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            model_bundle[side_name] = {
                "model": model,
                "standardizer": load_standardizer(sclerosis_dir / f"texture_standardizer_{side_name}.npz"),
            }
            print(f"  teacher[{side_name}] = {ckpt_path}", flush=True)
    if not model_bundle:
        ckpt_dir = checkpoint_root / "sclerosis"
        ckpt_path = find_best_lightning_checkpoint(
            ckpt_dir,
            monitor=checkpoint_monitor,
            mode=checkpoint_mode,
        ) if ckpt_dir.exists() else None
        if ckpt_path is None:
            return pd.DataFrame()
        checkpoint = load_checkpoint(ckpt_path, map_location=device)
        state_dict = extract_model_state_dict(checkpoint)
        current_model_cfg = base_model_cfg
        has_shared_classifier = any(k.startswith("classifier.") for k in state_dict)
        has_side_heads = any(k.startswith("medial_head.") for k in state_dict)
        if has_shared_classifier and not has_side_heads:
            current_model_cfg = OmegaConf.merge(base_model_cfg, {"use_side_specific_heads": False})
        elif has_side_heads and not has_shared_classifier:
            current_model_cfg = OmegaConf.merge(base_model_cfg, {"use_side_specific_heads": True})
        model = SclerosisClassifier(current_model_cfg)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        model_bundle["shared"] = {
            "model": model,
            "standardizer": load_standardizer(sclerosis_dir / "texture_standardizer.npz"),
        }
        print(f"  teacher[shared] = {ckpt_path}", flush=True)

    if not model_bundle:
        return pd.DataFrame()

    data = np.load(scl_npz, allow_pickle=True)
    if "side_ids" in data.files:
        side_ids = np.asarray(data["side_ids"], dtype=np.int64)
    else:
        side_ids = np.asarray([0 if str(image_id).endswith("_medial") else 1 for image_id in data["image_ids"]], dtype=np.int64)
    allowed_roi_sources = _resolve_allowed_roi_sources(cfg, "pseudo")
    roi_sources = None
    if "roi_sources" in data.files:
        roi_sources = np.asarray(data["roi_sources"]).astype(str)
    print(
        f"Sclerosis candidate ROI rows: {len(data['roi_paths'])}; "
        f"allowed_roi_sources={allowed_roi_sources or ['all']}",
        flush=True,
    )
    candidate_rows_by_image: dict[str, dict[str, object]] = {}
    for row_idx, (roi_path, texture_vec, image_id, side_id) in enumerate(tqdm(
        zip(data["roi_paths"], data["texture_features"], data["image_ids"], side_ids),
        total=len(data["roi_paths"]),
        desc="Pseudo sclerosis",
        file=sys.stdout,
    )):
        base_id, side = str(image_id).rsplit("_", 1)
        if base_id in manual_ids:
            continue
        if allowed_roi_sources and roi_sources is not None and roi_sources[row_idx] not in allowed_roi_sources:
            continue

        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if roi is None:
            continue
        transformed = transform(image=roi)
        roi_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(device)
        model_entry = model_bundle.get(side) or model_bundle.get("shared")
        if model_entry is None:
            continue
        tex_array = np.asarray(texture_vec, dtype=np.float64)
        if model_entry["standardizer"] is not None:
            tex_array = transform_with_standardizer(
                tex_array[None, :],
                model_entry["standardizer"][0],
                model_entry["standardizer"][1],
            )[0]
        tex_tensor = torch.tensor(tex_array, dtype=torch.float32).unsqueeze(0).to(device)
        side_tensor = torch.tensor([int(side_id)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model_entry["model"](roi_tensor, tex_tensor, side_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        if probs.numel() == 2 and binary_threshold is not None:
            present_prob = float(probs[1].item())
            pred_value = int(present_prob >= binary_threshold)
            conf_value = present_prob if pred_value == 1 else 1.0 - present_prob
        else:
            conf, pred = probs.max(dim=0)
            pred_value = int(pred.item())
            conf_value = float(conf.item())
        row = candidate_rows_by_image.setdefault(base_id, {})
        row[side] = {
            "pred": pred_value,
            "confidence": conf_value,
        }
        if (row_idx + 1) % 2000 == 0:
            print(
                f"Sclerosis pseudo-labeling progress: processed {row_idx + 1}/{len(data['roi_paths'])} ROI rows, "
                f"candidate images={len(candidate_rows_by_image)}",
                flush=True,
            )

    pseudo_df, threshold_used = _select_adaptive_dataframe(
        lambda thr: _build_sclerosis_pseudo_df(candidate_rows_by_image, thr),
        base_threshold=threshold,
        min_threshold=min_threshold,
        step=threshold_step,
        target_rows=target_rows,
    )
    out_dir = annotation_dir / "pseudo"
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_path = out_dir / "sclerosis_labels_high_conf.csv"
    pseudo_df.to_csv(pseudo_path, index=False)
    manual_path = resolve_annotation_csv(
        annotation_dir,
        "sclerosis_labels",
        mode="manual",
        allow_bootstrap_fallback=False,
    )
    merged = _merge_expanded(manual_path, pseudo_df, annotation_dir / "sclerosis_labels_expanded.csv")
    print(
        f"Saved sclerosis pseudo-labels: {pseudo_path} ({len(pseudo_df)} image rows, "
        f"threshold_used={threshold_used:.2f}, base_threshold={threshold:.2f})"
    )
    print(f"Saved expanded sclerosis labels: {annotation_dir / 'sclerosis_labels_expanded.csv'} ({len(merged)} rows)")
    return pseudo_df


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    label_scheme = str(getattr(cfg.training, "sclerosis_label_scheme", "severity"))
    cfg = apply_sclerosis_label_scheme_to_cfg(cfg, label_scheme)
    device = get_device()

    _generate_osteophyte_pseudolabels(cfg, device)
    _generate_sclerosis_pseudolabels(cfg, device)


if __name__ == "__main__":
    main()
