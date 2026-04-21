"""Evaluate sclerosis checkpoints and report KL-grade correlation."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src.data.sclerosis_dataset import SclerosisDataset
from src.data.transforms import get_eval_transforms
from src.models.sclerosis_classifier import SclerosisClassifier
from src.utils.annotation_confidence import confidence_at_least, confidence_weight
from src.utils.annotation_paths import EXPANDED_SOURCES, MANUAL_SOURCES
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.utils.feature_scaling import fit_standardizer, load_standardizer, transform_with_standardizer
from src.utils.seed import seed_everything


SIDE_NAMES = {0: "medial", 1: "lateral"}
SEPARATE_STRATEGIES = {"separate", "separate_by_side", "per_side", "side_specific"}


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "sclerosis_hybrid.yaml")


def _build_kl_lookup(data_root: Path) -> dict[str, int]:
    lookup = {}
    for split in ("train", "val", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for grade_dir in split_dir.iterdir():
            if not grade_dir.is_dir():
                continue
            grade = int(grade_dir.name)
            for image_path in grade_dir.glob("*.png"):
                lookup[image_path.stem] = grade
    return lookup


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if not np.isfinite(value) else value


def _resolve_checkpoint_path(cfg: DictConfig, ckpt_dir: Path) -> Path:
    checkpoint_path = getattr(cfg, "checkpoint_path", None)
    if checkpoint_path:
        return Path(str(checkpoint_path))

    checkpoint_monitor = getattr(cfg, "checkpoint_monitor", None)
    if checkpoint_monitor in (None, "", "null"):
        checkpoint_monitor = getattr(cfg.training, "sclerosis_primary_monitor", "val_f1_macro")
    checkpoint_mode = getattr(cfg, "checkpoint_mode", None)
    if checkpoint_mode in (None, "", "null"):
        checkpoint_mode = getattr(cfg.training, "sclerosis_primary_mode", "max")
    ckpt_path = find_best_lightning_checkpoint(
        ckpt_dir,
        monitor=str(checkpoint_monitor),
        mode=str(checkpoint_mode),
    ) if ckpt_dir.exists() else None
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No trained sclerosis checkpoint found in {ckpt_dir} for monitor={checkpoint_monitor}"
        )
    return ckpt_path


def _extract_side_ids(data) -> np.ndarray:
    if "side_ids" in data.files:
        return np.asarray(data["side_ids"], dtype=np.int64)
    image_ids = np.asarray(data["image_ids"]).astype(str)
    return np.asarray([0 if iid.endswith("_medial") else 1 for iid in image_ids], dtype=np.int64)


def _select_indices(data, label_mode: str, allow_bootstrap_fallback: bool) -> np.ndarray:
    if "label_sources" not in data.files:
        if allow_bootstrap_fallback or label_mode in {"bootstrap", "all", "pseudo"}:
            return np.arange(len(data["grades"]))
        raise ValueError("Sclerosis evaluation requires label_sources metadata for manual/expanded mode.")

    sources = np.asarray(data["label_sources"]).astype(str)
    if label_mode == "manual":
        mask = np.isin(sources, list(MANUAL_SOURCES))
        if mask.any():
            return np.flatnonzero(mask)
        if allow_bootstrap_fallback:
            return np.arange(len(sources))
        raise ValueError("Manual sclerosis evaluation requested, but no reviewed/manual labels are present.")
    if label_mode == "expanded":
        mask = np.isin(sources, list(EXPANDED_SOURCES))
        if mask.any():
            return np.flatnonzero(mask)
        mask = np.isin(sources, list(MANUAL_SOURCES))
        if mask.any():
            return np.flatnonzero(mask)
        if allow_bootstrap_fallback:
            return np.arange(len(sources))
        raise ValueError("Expanded sclerosis evaluation requested, but no reviewed/high-confidence labels are present.")
    return np.arange(len(sources))


def _apply_confidence_policy(
    data,
    keep: np.ndarray,
    min_confidence: str,
    confidence_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    if "confidence_levels" not in data.files:
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


def _apply_roi_source_filter(
    data,
    keep: np.ndarray,
    sample_weights: np.ndarray,
    allowed_sources,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = [str(item).strip() for item in list(allowed_sources or []) if str(item).strip()]
    if not allowed or "roi_sources" not in data.files:
        return keep, sample_weights

    roi_sources = np.asarray(data["roi_sources"]).astype(str)
    mask = np.isin(roi_sources[keep], allowed)
    return keep[mask], sample_weights[mask]


def _strategy_is_separate(cfg: DictConfig, checkpoint_root: Path) -> bool:
    strategy = str(getattr(cfg.training, "sclerosis_strategy", "separate")).lower()
    if strategy in SEPARATE_STRATEGIES:
        return True
    if strategy == "shared":
        return False
    return (checkpoint_root / "sclerosis_medial").exists() and (checkpoint_root / "sclerosis_lateral").exists()


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


def _fit_texture_baseline(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(features, labels)
    return model


def _compute_metrics(targets: np.ndarray, preds: np.ndarray, probs: np.ndarray, kl_grades: np.ndarray) -> dict:
    metrics = {
        "num_samples": int(len(targets)),
        "accuracy": float(accuracy_score(targets, preds)),
        "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1, 2]).tolist(),
        "kl_correlation": _safe_correlation(preds.astype(np.float64), kl_grades.astype(np.float64)),
    }
    if len(np.unique(targets)) > 1:
        try:
            metrics["auc_macro"] = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return metrics


def _collect_model_predictions(
    model: SclerosisClassifier,
    dataset: SclerosisDataset,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    preds = []
    probs = []
    targets = []
    with torch.no_grad():
        for images, texture_feats, side_ids_batch, labels, _ in loader:
            images = images.to(device)
            texture_feats = texture_feats.to(device)
            side_ids_batch = side_ids_batch.to(device)
            logits = model(images, texture_feats, side_ids_batch)
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
            targets.extend(labels.tolist())
    return (
        np.asarray(preds, dtype=np.int64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(targets, dtype=np.int64),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)
    device = get_device()

    checkpoint_root = Path(cfg.checkpoint_dir)
    separate = _strategy_is_separate(cfg, checkpoint_root)
    kl_lookup = _build_kl_lookup(Path(cfg.data.root))
    transform = get_eval_transforms(cfg)
    target_roi_size = int(getattr(cfg.preprocessing.sclerosis_roi, "output_size", 96))
    result = {
        "checkpoint_strategy": "separate" if separate else "shared",
        "checkpoints": {},
        "hybrid": {},
        "hybrid_by_side": {},
        "texture_only": {},
        "texture_only_by_side": {},
    }

    label_mode = str(getattr(cfg.training, "label_mode", "manual")).lower()
    allow_bootstrap_fallback = bool(getattr(cfg.training, "allow_bootstrap_fallback", False))
    confidence_cfg = getattr(cfg.training, "annotation_confidence", {})
    min_eval_confidence = str(confidence_cfg.get("min_eval", "low"))
    confidence_weights = {
        "low": float(confidence_cfg.get("weight_low", 0.5)),
        "medium": float(confidence_cfg.get("weight_medium", 0.75)),
        "high": float(confidence_cfg.get("weight_high", 1.0)),
    }

    scl_dir = Path(str(getattr(cfg, "sclerosis_output_dir", Path(cfg.feature_dir) / "sclerosis")))
    train_npz = np.load(scl_dir / "train_sclerosis_data.npz", allow_pickle=True)
    train_keep = _select_indices(train_npz, label_mode, allow_bootstrap_fallback)
    train_keep, _ = _apply_confidence_policy(train_npz, train_keep, min_eval_confidence, confidence_weights)
    train_keep, _ = _apply_roi_source_filter(
        train_npz,
        train_keep,
        np.ones(len(train_keep), dtype=np.float32),
        allowed_sources=getattr(getattr(cfg.training, "sclerosis_roi_source_filter", {}), "train", []),
    )
    train_side_ids_all = _extract_side_ids(train_npz)

    bundles: dict[str, dict] = {}
    texture_models: dict[str, LogisticRegression] = {}

    if separate:
        side_checkpoint_overrides = {
            "medial": getattr(cfg, "checkpoint_path_medial", None),
            "lateral": getattr(cfg, "checkpoint_path_lateral", None),
        }
        for side_name, side_id in (("medial", 0), ("lateral", 1)):
            side_keep = train_keep[train_side_ids_all[train_keep] == side_id]
            if len(side_keep) == 0:
                continue
            ckpt_dir = checkpoint_root / f"sclerosis_{side_name}"
            if side_checkpoint_overrides[side_name]:
                ckpt_path = Path(str(side_checkpoint_overrides[side_name]))
            else:
                ckpt_path = _resolve_checkpoint_path(cfg, ckpt_dir)
            standardizer = load_standardizer(scl_dir / f"texture_standardizer_{side_name}.npz")
            if standardizer is None:
                standardizer = fit_standardizer(train_npz["texture_features"][side_keep])
            mean, scale = standardizer
            scaled_texture = transform_with_standardizer(train_npz["texture_features"][side_keep], mean, scale)
            bundles[side_name] = {
                "model": _load_model(model_cfg, ckpt_path, device),
                "standardizer": (mean, scale),
                "side_id": side_id,
            }
            texture_models[side_name] = _fit_texture_baseline(scaled_texture, train_npz["grades"][side_keep])
            result["checkpoints"][side_name] = str(ckpt_path)
    else:
        ckpt_path = _resolve_checkpoint_path(cfg, checkpoint_root / "sclerosis")
        standardizer = load_standardizer(scl_dir / "texture_standardizer.npz")
        if standardizer is None:
            standardizer = fit_standardizer(train_npz["texture_features"][train_keep])
        mean, scale = standardizer
        scaled_texture = transform_with_standardizer(train_npz["texture_features"][train_keep], mean, scale)
        bundles["shared"] = {
            "model": _load_model(model_cfg, ckpt_path, device),
            "standardizer": (mean, scale),
        }
        train_side_onehot = np.eye(2, dtype=np.float64)[train_side_ids_all[train_keep]]
        texture_models["shared"] = _fit_texture_baseline(
            np.concatenate([scaled_texture, train_side_onehot], axis=1),
            train_npz["grades"][train_keep],
        )
        result["checkpoints"]["shared"] = str(ckpt_path)

    for split in ("val", "test"):
        data = np.load(scl_dir / f"{split}_sclerosis_data.npz", allow_pickle=True)
        keep = _select_indices(data, label_mode, allow_bootstrap_fallback)
        keep, sample_weights = _apply_confidence_policy(data, keep, min_eval_confidence, confidence_weights)
        keep, sample_weights = _apply_roi_source_filter(
            data,
            keep,
            sample_weights,
            allowed_sources=getattr(getattr(cfg.training, "sclerosis_roi_source_filter", {}), "eval", []),
        )
        side_ids_all = _extract_side_ids(data)

        hybrid_by_side = {}
        texture_by_side = {}
        hybrid_preds_all = []
        hybrid_probs_all = []
        hybrid_targets_all = []
        hybrid_kl_all = []
        texture_preds_all = []
        texture_probs_all = []
        texture_targets_all = []
        texture_kl_all = []

        if separate:
            run_defs = [
                ("medial", 0),
                ("lateral", 1),
            ]
        else:
            run_defs = [("shared", None)]

        for run_name, side_id in run_defs:
            if separate:
                side_keep = keep[side_ids_all[keep] == side_id]
                if len(side_keep) == 0:
                    continue
                side_sample_weights = sample_weights[side_ids_all[keep] == side_id]
                side_name = SIDE_NAMES[side_id]
                if side_name not in bundles or side_name not in texture_models:
                    continue
                bundle = bundles[side_name]
                standardizer = bundle["standardizer"]
                scaled_texture = transform_with_standardizer(data["texture_features"][side_keep], *standardizer)
                dataset = SclerosisDataset(
                    roi_paths=data["roi_paths"][side_keep].tolist(),
                    texture_features=scaled_texture,
                    side_ids=side_ids_all[side_keep],
                    grades=data["grades"][side_keep],
                    sample_weights=side_sample_weights,
                    transform=transform,
                    target_size=target_roi_size,
                )
                preds, probs, targets = _collect_model_predictions(bundle["model"], dataset, device, cfg.data.num_workers)
                base_image_ids = [str(item).rsplit("_", 1)[0] for item in data["image_ids"][side_keep]]
                kl_grades = np.asarray([kl_lookup.get(image_id, 0) for image_id in base_image_ids], dtype=np.float64)
                hybrid_by_side[side_name] = _compute_metrics(targets, preds, probs, kl_grades)
                hybrid_preds_all.append(preds)
                hybrid_probs_all.append(probs)
                hybrid_targets_all.append(targets)
                hybrid_kl_all.append(kl_grades)

                texture_input = scaled_texture
                texture_probs = texture_models[side_name].predict_proba(texture_input)
                texture_preds = texture_models[side_name].predict(texture_input)
                texture_by_side[side_name] = _compute_metrics(targets, texture_preds, texture_probs, kl_grades)
                texture_preds_all.append(texture_preds.astype(np.int64))
                texture_probs_all.append(np.asarray(texture_probs, dtype=np.float64))
                texture_targets_all.append(targets)
                texture_kl_all.append(kl_grades)
            else:
                bundle = bundles["shared"]
                standardizer = bundle["standardizer"]
                scaled_texture = transform_with_standardizer(data["texture_features"][keep], *standardizer)
                dataset = SclerosisDataset(
                    roi_paths=data["roi_paths"][keep].tolist(),
                    texture_features=scaled_texture,
                    side_ids=side_ids_all[keep],
                    grades=data["grades"][keep],
                    sample_weights=sample_weights,
                    transform=transform,
                    target_size=target_roi_size,
                )
                preds, probs, targets = _collect_model_predictions(bundle["model"], dataset, device, cfg.data.num_workers)
                base_image_ids = [str(item).rsplit("_", 1)[0] for item in data["image_ids"][keep]]
                kl_grades = np.asarray([kl_lookup.get(image_id, 0) for image_id in base_image_ids], dtype=np.float64)
                hybrid_preds_all.append(preds)
                hybrid_probs_all.append(probs)
                hybrid_targets_all.append(targets)
                hybrid_kl_all.append(kl_grades)

                side_onehot = np.eye(2, dtype=np.float64)[side_ids_all[keep]]
                texture_input = np.concatenate([scaled_texture, side_onehot], axis=1)
                texture_probs = texture_models["shared"].predict_proba(texture_input)
                texture_preds = texture_models["shared"].predict(texture_input)
                texture_preds_all.append(texture_preds.astype(np.int64))
                texture_probs_all.append(np.asarray(texture_probs, dtype=np.float64))
                texture_targets_all.append(targets)
                texture_kl_all.append(kl_grades)

                for side_value, side_name in SIDE_NAMES.items():
                    mask = side_ids_all[keep] == side_value
                    if not mask.any():
                        continue
                    hybrid_by_side[side_name] = _compute_metrics(
                        targets[mask],
                        preds[mask],
                        probs[mask],
                        kl_grades[mask],
                    )
                    texture_by_side[side_name] = _compute_metrics(
                        targets[mask],
                        texture_preds[mask],
                        texture_probs[mask],
                        kl_grades[mask],
                    )

        if hybrid_targets_all:
            result["hybrid"][split] = _compute_metrics(
                np.concatenate(hybrid_targets_all),
                np.concatenate(hybrid_preds_all),
                np.concatenate(hybrid_probs_all),
                np.concatenate(hybrid_kl_all),
            )
        if texture_targets_all:
            result["texture_only"][split] = _compute_metrics(
                np.concatenate(texture_targets_all),
                np.concatenate(texture_preds_all),
                np.concatenate(texture_probs_all),
                np.concatenate(texture_kl_all),
            )
        if hybrid_by_side:
            result["hybrid_by_side"][split] = hybrid_by_side
        if texture_by_side:
            result["texture_only_by_side"][split] = texture_by_side

    out_path = Path(cfg.result_dir) / "sclerosis_evaluation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved sclerosis evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
