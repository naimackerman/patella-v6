"""Stable metrics-only evaluation for KL Path B (hybrid ConvNeXt + features)."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src.data.kl_dataset import KLHybridDataset
from src.data.transforms import get_eval_transforms
from src.models.kl_hybrid import HybridKLClassifier
from src.utils.checkpoints import checkpoint_score, extract_model_state_dict, load_checkpoint
from src.utils.device import clear_memory, get_device
from src.utils.metrics import per_class_metrics, quadratic_weighted_kappa
from src.utils.seed import seed_everything


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    model_cfg_path = Path(__file__).resolve().parents[1] / "configs" / "model" / "convnext_hybrid.yaml"
    return OmegaConf.load(model_cfg_path)


def _model_cfg_from_checkpoint(cfg: DictConfig, checkpoint: dict) -> DictConfig:
    base_cfg = OmegaConf.create(OmegaConf.to_container(_resolve_model_cfg(cfg), resolve=True))
    checkpoint_hparams = checkpoint.get("hyper_parameters", {})
    checkpoint_model_cfg = checkpoint_hparams.get("model")
    if checkpoint_model_cfg:
        base_cfg = OmegaConf.merge(base_cfg, checkpoint_model_cfg)
    # Evaluation should use checkpoint weights only, not fetch fresh pretrained weights.
    base_cfg.pretrained = False
    return base_cfg


def _checkpoint_feature_dim(checkpoint: dict) -> int | None:
    model_cfg = checkpoint.get("hyper_parameters", {}).get("model")
    if model_cfg is None:
        return None
    value = (
        model_cfg.get("feature_dim")
        if isinstance(model_cfg, dict)
        else getattr(model_cfg, "feature_dim", None)
    )
    return int(value) if value is not None else None


def _find_best_compatible_checkpoint(
    ckpt_dir: Path,
    monitor: str,
    mode: str,
    expected_feature_dim: int | None,
) -> Path | None:
    candidates = []
    for ckpt_path in sorted(ckpt_dir.glob("*.ckpt")):
        score = checkpoint_score(ckpt_path, monitor=monitor)
        if score == float("-inf"):
            continue
        if expected_feature_dim is not None:
            checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
            feature_dim = _checkpoint_feature_dim(checkpoint)
            if feature_dim is not None and feature_dim != expected_feature_dim:
                continue
        candidates.append((score, ckpt_path))
    if not candidates:
        return None
    reverse = str(mode).lower() != "min"
    candidates.sort(key=lambda item: item[0], reverse=reverse)
    return candidates[0][1]


def _resolve_hybrid_checkpoint(cfg: DictConfig, expected_feature_dim: int | None = None) -> Path | None:
    explicit_ckpt = getattr(cfg, "checkpoint_path", None)
    if explicit_ckpt:
        ckpt_path = Path(str(explicit_ckpt))
        return ckpt_path if ckpt_path.exists() else None

    ckpt_dir = Path(cfg.checkpoint_dir) / "kl_hybrid"
    if not ckpt_dir.exists():
        return None

    monitor = str(getattr(cfg, "checkpoint_monitor", "val_qwk"))
    mode = str(getattr(cfg, "checkpoint_mode", "max"))
    monitor_candidates = []
    if monitor not in {"", "null", "None"}:
        monitor_candidates.append(monitor)
    for candidate in ("val_qwk", "val/qwk"):
        if candidate not in monitor_candidates:
            monitor_candidates.append(candidate)

    for candidate in monitor_candidates:
        ckpt_path = _find_best_compatible_checkpoint(
            ckpt_dir,
            monitor=candidate,
            mode=mode,
            expected_feature_dim=expected_feature_dim,
        )
        if ckpt_path is not None and ckpt_path.exists():
            return ckpt_path
    return None


def _evaluate_split(
    model: HybridKLClassifier,
    dataset: KLHybridDataset,
    class_names: list[str],
    device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, features, labels in loader:
            images = images.to(device)
            features = features.to(device)

            logits = model(images, features)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())

    preds = np.asarray(all_preds, dtype=np.int64)
    probs = np.vstack(all_probs) if all_probs else np.zeros((0, len(class_names)), dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)

    metrics = {
        "num_samples": int(labels.shape[0]),
        "qwk": float(quadratic_weighted_kappa(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
    }
    metrics.update(per_class_metrics(labels, preds, class_names))

    if len(np.unique(labels)) > 1 and probs.size > 0:
        try:
            metrics["auc_macro"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except ValueError:
            pass

    payload = {
        "image_ids": np.asarray([Path(path).stem for path in dataset.image_ds.samples], dtype=object),
        "y_true": labels,
        "y_pred": preds,
        "y_prob": probs,
    }
    return metrics, payload


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()

    feature_dir = Path(cfg.feature_dir) / "aggregated"
    val_features_npz = np.load(feature_dir / "val_features.npz")
    expected_feature_dim = int(val_features_npz["features"].shape[1])
    ckpt_path = _resolve_hybrid_checkpoint(cfg, expected_feature_dim=expected_feature_dim)

    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("No KL hybrid checkpoint found for evaluation.")

    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    model_cfg = _model_cfg_from_checkpoint(cfg, checkpoint)
    if int(model_cfg.feature_dim) != expected_feature_dim:
        raise ValueError(
            f"Hybrid checkpoint feature_dim={model_cfg.feature_dim} does not match "
            f"current aggregated feature_dim={expected_feature_dim}: {ckpt_path}"
        )
    model = HybridKLClassifier(model_cfg)
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.to(device)
    model.eval()

    result_dir = Path(cfg.result_dir) / "kl_hybrid_evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    transform = get_eval_transforms(cfg)

    summary = {"checkpoint": str(ckpt_path)}
    for split in ("val", "test"):
        features_npz = feature_dir / f"{split}_features.npz"
        dataset = KLHybridDataset(
            cfg.data.root,
            split,
            str(features_npz),
            transform,
        )
        metrics, payload = _evaluate_split(
            model,
            dataset,
            cfg.data.class_names,
            device,
            batch_size=4,
            num_workers=cfg.data.num_workers,
        )
        summary[split] = metrics
        np.savez(
            str(result_dir / f"hybrid_{split}_predictions.npz"),
            **payload,
        )

    out_path = result_dir / "kl_hybrid_evaluation.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved KL hybrid evaluation summary to {out_path}")
    clear_memory()


if __name__ == "__main__":
    main()
