"""Tune binary sclerosis decision thresholds on validation and evaluate test."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.sclerosis_dataset import SclerosisDataset
from src.data.transforms import get_eval_transforms
from src.models.sclerosis_classifier import SclerosisClassifier
from src.utils.annotation_confidence import confidence_at_least, confidence_weight
from src.utils.annotation_paths import EXPANDED_SOURCES, MANUAL_SOURCES
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.utils.feature_scaling import load_standardizer, transform_with_standardizer
from src.utils.seed import seed_everything
from src.utils.sclerosis_labels import apply_sclerosis_label_scheme_to_cfg, map_sclerosis_grades


SIDE_NAMES = {0: "medial", 1: "lateral"}


def _data_has_key(data, key: str) -> bool:
    return key in data.keys() if isinstance(data, dict) else key in data.files


def _apply_binary_labels(data) -> dict[str, np.ndarray]:
    keys = data.keys() if isinstance(data, dict) else data.files
    mapped = {key: np.asarray(data[key]) for key in keys}
    mapped["grades"] = map_sclerosis_grades(mapped["grades"], "binary_present")
    return mapped


def _select_indices(data, label_mode: str, allow_bootstrap_fallback: bool) -> np.ndarray:
    if not _data_has_key(data, "label_sources"):
        if allow_bootstrap_fallback or label_mode in {"bootstrap", "all", "pseudo"}:
            return np.arange(len(data["grades"]))
        raise ValueError("Sclerosis threshold evaluation requires label_sources metadata.")

    sources = np.asarray(data["label_sources"]).astype(str)
    if label_mode == "manual":
        mask = np.isin(sources, list(MANUAL_SOURCES))
        if mask.any():
            return np.flatnonzero(mask)
        if allow_bootstrap_fallback:
            return np.arange(len(sources))
        raise ValueError("Manual sclerosis evaluation requested, but no manual rows are present.")
    if label_mode == "expanded":
        mask = np.isin(sources, list(EXPANDED_SOURCES))
        if mask.any():
            return np.flatnonzero(mask)
        if allow_bootstrap_fallback:
            return np.arange(len(sources))
        raise ValueError("Expanded sclerosis evaluation requested, but no expanded rows are present.")
    return np.arange(len(sources))


def _apply_confidence_policy(data, keep: np.ndarray, min_confidence: str, confidence_weights: dict[str, float]):
    if not _data_has_key(data, "confidence_levels"):
        return keep, np.ones(len(keep), dtype=np.float32)
    confidences = np.asarray(data["confidence_levels"]).astype(str)
    selected = []
    sample_weights = []
    for idx in keep.tolist():
        conf = confidences[idx]
        if not confidence_at_least(conf, min_confidence):
            continue
        selected.append(idx)
        sample_weights.append(confidence_weight(conf, confidence_weights))
    return np.asarray(selected, dtype=np.int64), np.asarray(sample_weights, dtype=np.float32)


def _extract_side_ids(data) -> np.ndarray:
    if _data_has_key(data, "side_ids"):
        return np.asarray(data["side_ids"], dtype=np.int64)
    image_ids = np.asarray(data["image_ids"]).astype(str)
    return np.asarray([0 if iid.endswith("_medial") else 1 for iid in image_ids], dtype=np.int64)


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "sclerosis_hybrid.yaml")


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    checkpoint_path = getattr(cfg, "checkpoint_path", None)
    if checkpoint_path:
        return Path(str(checkpoint_path))
    checkpoint_root = Path(cfg.checkpoint_dir) / "sclerosis"
    checkpoint_monitor = getattr(cfg, "checkpoint_monitor", None)
    if checkpoint_monitor in (None, "", "null"):
        checkpoint_monitor = getattr(cfg.training, "sclerosis_secondary_monitor", "val_auc_macro")
    checkpoint_mode = getattr(cfg, "checkpoint_mode", None)
    if checkpoint_mode in (None, "", "null"):
        checkpoint_mode = getattr(cfg.training, "sclerosis_secondary_mode", "max")
    ckpt_path = find_best_lightning_checkpoint(
        checkpoint_root,
        pattern="scl-*.ckpt",
        monitor=str(checkpoint_monitor),
        mode=str(checkpoint_mode),
    )
    if ckpt_path is None:
        raise FileNotFoundError(f"No sclerosis checkpoint found in {checkpoint_root}")
    return ckpt_path


def _load_model(model_cfg: DictConfig, checkpoint_path: Path, device: torch.device) -> SclerosisClassifier:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = extract_model_state_dict(checkpoint)
    current_model_cfg = model_cfg
    has_shared_classifier = any(k.startswith("classifier.") for k in state_dict)
    has_side_heads = any(k.startswith("medial_head.") for k in state_dict)
    if has_shared_classifier and not has_side_heads:
        current_model_cfg = OmegaConf.merge(model_cfg, {"use_side_specific_heads": False})
    elif has_side_heads and not has_shared_classifier:
        current_model_cfg = OmegaConf.merge(model_cfg, {"use_side_specific_heads": True})
    model = SclerosisClassifier(current_model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _predict_split(
    cfg: DictConfig,
    split: str,
    model: SclerosisClassifier,
    standardizer: tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scl_dir = Path(str(getattr(cfg, "sclerosis_output_dir", Path(cfg.feature_dir) / "sclerosis")))
    data = _apply_binary_labels(np.load(scl_dir / f"{split}_sclerosis_data.npz", allow_pickle=True))
    label_mode = str(getattr(cfg.training, "label_mode", "manual")).lower()
    allow_bootstrap_fallback = bool(getattr(cfg.training, "allow_bootstrap_fallback", False))
    confidence_cfg = getattr(cfg.training, "annotation_confidence", {})
    confidence_weights = {
        "low": float(confidence_cfg.get("weight_low", 0.5)),
        "medium": float(confidence_cfg.get("weight_medium", 0.75)),
        "high": float(confidence_cfg.get("weight_high", 1.0)),
    }
    keep = _select_indices(data, label_mode, allow_bootstrap_fallback)
    keep, sample_weights = _apply_confidence_policy(
        data,
        keep,
        min_confidence=str(confidence_cfg.get("min_eval", "low")),
        confidence_weights=confidence_weights,
    )
    side_ids = _extract_side_ids(data)[keep]
    texture = transform_with_standardizer(data["texture_features"][keep], standardizer[0], standardizer[1])
    target_roi_size = int(getattr(cfg.preprocessing.sclerosis_roi, "output_size", 96))
    dataset = SclerosisDataset(
        roi_paths=data["roi_paths"][keep].tolist(),
        texture_features=texture,
        side_ids=side_ids,
        grades=data["grades"][keep],
        sample_weights=sample_weights,
        transform=get_eval_transforms(cfg),
        target_size=target_roi_size,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=cfg.data.num_workers)
    probs_present = []
    targets = []
    sides = []
    with torch.no_grad():
        for images, texture_feats, side_batch, labels, _ in loader:
            logits = model(
                images.to(device),
                texture_feats.to(device),
                side_batch.to(device),
            )
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_present.extend(probs.cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
            sides.extend(side_batch.numpy().tolist())
    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(probs_present, dtype=np.float64),
        np.asarray(sides, dtype=np.int64),
    )


def _metrics_at_threshold(targets: np.ndarray, probs_present: np.ndarray, threshold: float) -> dict:
    preds = (probs_present >= float(threshold)).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        preds,
        labels=[0, 1],
        zero_division=0,
    )
    result = {
        "threshold": float(threshold),
        "num_samples": int(len(targets)),
        "accuracy": float(accuracy_score(targets, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1]).tolist(),
        "precision_none": float(precision[0]),
        "precision_present": float(precision[1]),
        "recall_none": float(recall[0]),
        "recall_present": float(recall[1]),
        "f1_none": float(f1[0]),
        "f1_present": float(f1[1]),
    }
    if len(np.unique(targets)) > 1:
        result["auc"] = float(roc_auc_score(targets, probs_present))
    return result


def _threshold_grid(probs_present: np.ndarray) -> np.ndarray:
    base = np.linspace(0.05, 0.95, 181, dtype=np.float64)
    candidates = np.unique(np.concatenate([base, probs_present]))
    return candidates[(candidates >= 0.01) & (candidates <= 0.99)]


def _select_threshold(targets: np.ndarray, probs_present: np.ndarray, objective: str) -> tuple[float, dict]:
    objective = str(objective or "f1_macro")
    best_threshold = 0.5
    best_metrics = _metrics_at_threshold(targets, probs_present, best_threshold)
    for threshold in _threshold_grid(probs_present):
        metrics = _metrics_at_threshold(targets, probs_present, float(threshold))
        score = float(metrics.get(objective, metrics["f1_macro"]))
        best_score = float(best_metrics.get(objective, best_metrics["f1_macro"]))
        if score > best_score or (score == best_score and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _side_specific_thresholds(
    targets: np.ndarray,
    probs_present: np.ndarray,
    side_ids: np.ndarray,
    objective: str,
) -> tuple[dict[str, float], dict[str, dict]]:
    thresholds = {}
    metrics = {}
    for side_value, side_name in SIDE_NAMES.items():
        mask = side_ids == side_value
        if not mask.any() or len(np.unique(targets[mask])) < 2:
            thresholds[side_name] = 0.5
            metrics[side_name] = _metrics_at_threshold(targets[mask], probs_present[mask], 0.5)
            continue
        threshold, side_metrics = _select_threshold(targets[mask], probs_present[mask], objective)
        thresholds[side_name] = threshold
        metrics[side_name] = side_metrics
    return thresholds, metrics


def _metrics_with_side_thresholds(
    targets: np.ndarray,
    probs_present: np.ndarray,
    side_ids: np.ndarray,
    thresholds: dict[str, float],
) -> dict:
    preds = np.zeros_like(targets)
    for side_value, side_name in SIDE_NAMES.items():
        mask = side_ids == side_value
        preds[mask] = (probs_present[mask] >= float(thresholds[side_name])).astype(np.int64)
    # Reuse metric function by passing binary predictions as 0/1 probabilities at threshold 0.5.
    metrics = _metrics_at_threshold(targets, preds.astype(np.float64), 0.5)
    metrics["thresholds"] = {name: float(value) for name, value in thresholds.items()}
    if len(np.unique(targets)) > 1:
        metrics["auc"] = float(roc_auc_score(targets, probs_present))
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    cfg = apply_sclerosis_label_scheme_to_cfg(cfg, "binary_present")
    device = get_device()
    model_cfg = _resolve_model_cfg(cfg)
    ckpt_path = _resolve_checkpoint_path(cfg)
    model = _load_model(model_cfg, ckpt_path, device)

    scl_dir = Path(str(getattr(cfg, "sclerosis_output_dir", Path(cfg.feature_dir) / "sclerosis")))
    standardizer = load_standardizer(scl_dir / "texture_standardizer.npz")
    if standardizer is None:
        raise FileNotFoundError(f"Missing texture standardizer in {scl_dir}")

    objective = str(getattr(cfg, "threshold_objective", "f1_macro"))
    val_targets, val_probs, val_sides = _predict_split(cfg, "val", model, standardizer, device)
    test_targets, test_probs, test_sides = _predict_split(cfg, "test", model, standardizer, device)

    global_threshold, val_global = _select_threshold(val_targets, val_probs, objective)
    side_thresholds, val_side = _side_specific_thresholds(val_targets, val_probs, val_sides, objective)

    result = {
        "checkpoint": str(ckpt_path),
        "label_scheme": "binary_present",
        "objective": objective,
        "class_names": ["none", "present"],
        "global_threshold": {
            "threshold": float(global_threshold),
            "val": val_global,
            "test": _metrics_at_threshold(test_targets, test_probs, global_threshold),
        },
        "side_specific_thresholds": {
            "thresholds": side_thresholds,
            "val_by_side": val_side,
            "val": _metrics_with_side_thresholds(val_targets, val_probs, val_sides, side_thresholds),
            "test": _metrics_with_side_thresholds(test_targets, test_probs, test_sides, side_thresholds),
        },
        "default_threshold_0_5": {
            "val": _metrics_at_threshold(val_targets, val_probs, 0.5),
            "test": _metrics_at_threshold(test_targets, test_probs, 0.5),
        },
    }
    out_path = Path(cfg.result_dir) / "sclerosis_threshold_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved threshold evaluation to {out_path}")


if __name__ == "__main__":
    main()
