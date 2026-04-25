"""Inference-time feature ablation for KL Path B.

This evaluates the trained ConvNeXt + feature-fusion checkpoint while masking or
permuting the normalized 50-dimensional feature vector at inference time. It is
an occlusion/sensitivity analysis, not a replacement for training an image-only
control model.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from scripts.evaluate_kl_hybrid_metrics import (
    _checkpoint_feature_dim,
    _model_cfg_from_checkpoint,
    _resolve_hybrid_checkpoint,
)
from src.data.kl_dataset import KLHybridDataset
from src.data.transforms import get_eval_transforms
from src.models.kl_hybrid import HybridKLClassifier
from src.utils.checkpoints import extract_model_state_dict, load_checkpoint
from src.utils.device import clear_memory, get_device
from src.utils.metrics import per_class_metrics, quadratic_weighted_kappa
from src.utils.seed import seed_everything


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    intervention: str
    feature_group: str


FEATURE_GROUPS = {
    "all": (0, 50),
    "jsn": (0, 22),
    "osteophyte": (22, 32),
    "sclerosis": (32, 50),
}

ABLATIONS = [
    AblationSpec("full", "Original images + original 50-dim feature vector", "none", "all"),
    AblationSpec(
        "zero_all",
        "Images + feature vector replaced by post-normalization zeros",
        "zero",
        "all",
    ),
    AblationSpec(
        "permute_all",
        "Images + feature vector permuted across samples",
        "permute",
        "all",
    ),
    AblationSpec("zero_jsn", "Images + JSN dimensions zeroed", "zero", "jsn"),
    AblationSpec("permute_jsn", "Images + JSN dimensions permuted", "permute", "jsn"),
    AblationSpec("zero_osteophyte", "Images + osteophyte dimensions zeroed", "zero", "osteophyte"),
    AblationSpec(
        "permute_osteophyte",
        "Images + osteophyte dimensions permuted",
        "permute",
        "osteophyte",
    ),
    AblationSpec("zero_sclerosis", "Images + sclerosis dimensions zeroed", "zero", "sclerosis"),
    AblationSpec(
        "permute_sclerosis",
        "Images + sclerosis dimensions permuted",
        "permute",
        "sclerosis",
    ),
]


def _feature_slice(feature_group: str, feature_dim: int) -> slice:
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {feature_group}")
    start, end = FEATURE_GROUPS[feature_group]
    if start >= feature_dim:
        raise ValueError(f"Feature group {feature_group} starts beyond feature_dim={feature_dim}")
    return slice(start, min(end, feature_dim))


def _ordered_feature_matrix(dataset: KLHybridDataset) -> np.ndarray:
    rows = []
    for img_path in dataset.image_ds.samples:
        image_id = Path(img_path).stem
        rows.append(dataset.feature_map.get(image_id, np.zeros(dataset.feat_dim, dtype=np.float32)))
    return np.stack(rows).astype(np.float32)


def _classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> dict:
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
    return metrics


def _ablate_features(
    features: torch.Tensor,
    spec: AblationSpec,
    feature_matrix: torch.Tensor,
    batch_indices: np.ndarray,
    permutations: dict[str, np.ndarray],
) -> torch.Tensor:
    if spec.intervention == "none":
        return features

    feature_dim = int(features.shape[1])
    feature_slice = _feature_slice(spec.feature_group, feature_dim)
    ablated = features.clone()

    if spec.intervention == "zero":
        ablated[:, feature_slice] = 0.0
    elif spec.intervention == "permute":
        permuted_indices = permutations[spec.feature_group][batch_indices]
        source = feature_matrix[torch.as_tensor(permuted_indices, device=feature_matrix.device)]
        ablated[:, feature_slice] = source[:, feature_slice]
    else:
        raise ValueError(f"Unknown intervention: {spec.intervention}")

    return ablated


def _evaluate_split_ablation(
    model: HybridKLClassifier,
    dataset: KLHybridDataset,
    class_names: list[str],
    device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, dict], dict[str, dict[str, np.ndarray]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    feature_matrix = torch.from_numpy(_ordered_feature_matrix(dataset)).to(device)
    rng = np.random.default_rng(seed)
    permutations = {
        group: rng.permutation(len(dataset))
        for group in FEATURE_GROUPS
    }

    labels_all = []
    preds_by_ablation = {spec.name: [] for spec in ABLATIONS}
    probs_by_ablation = {spec.name: [] for spec in ABLATIONS}

    offset = 0
    with torch.no_grad():
        for images, features, labels in loader:
            batch_size_actual = int(labels.shape[0])
            batch_indices = np.arange(offset, offset + batch_size_actual)
            offset += batch_size_actual

            images = images.to(device)
            features = features.to(device)

            image_features = model.image_encoder(images)
            for spec in ABLATIONS:
                ablated_features = _ablate_features(
                    features,
                    spec,
                    feature_matrix,
                    batch_indices,
                    permutations,
                )
                encoded_features = model.feature_encoder(ablated_features)
                fused = torch.cat([image_features, encoded_features], dim=1)
                logits = model.classifier(fused)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                preds_by_ablation[spec.name].extend(preds.cpu().numpy().tolist())
                probs_by_ablation[spec.name].append(probs.cpu().numpy())

            labels_all.extend(labels.numpy().tolist())

    labels_np = np.asarray(labels_all, dtype=np.int64)
    image_ids = np.asarray([Path(path).stem for path in dataset.image_ds.samples], dtype=object)

    metrics_by_ablation = {}
    payload_by_ablation = {}
    for spec in ABLATIONS:
        preds = np.asarray(preds_by_ablation[spec.name], dtype=np.int64)
        probs = np.vstack(probs_by_ablation[spec.name])
        metrics = _classification_metrics(labels_np, preds, probs, class_names)
        metrics.update(
            {
                "description": spec.description,
                "intervention": spec.intervention,
                "feature_group": spec.feature_group,
                "n_features_changed": int(_feature_slice(spec.feature_group, dataset.feat_dim).stop - _feature_slice(spec.feature_group, dataset.feat_dim).start)
                if spec.intervention != "none"
                else 0,
            }
        )
        metrics_by_ablation[spec.name] = metrics
        payload_by_ablation[spec.name] = {
            "image_ids": image_ids,
            "y_true": labels_np,
            "y_pred": preds,
            "y_prob": probs,
        }

    full_metrics = metrics_by_ablation["full"]
    for name, metrics in metrics_by_ablation.items():
        metrics["delta_qwk_vs_full"] = float(metrics["qwk"] - full_metrics["qwk"])
        metrics["delta_accuracy_vs_full"] = float(metrics["accuracy"] - full_metrics["accuracy"])

    return metrics_by_ablation, payload_by_ablation


def _write_csv(summary: dict, out_path: Path, splits: list[str]) -> None:
    fieldnames = [
        "split",
        "ablation",
        "description",
        "intervention",
        "feature_group",
        "n_features_changed",
        "qwk",
        "delta_qwk_vs_full",
        "accuracy",
        "delta_accuracy_vs_full",
        "f1_macro",
        "auc_macro",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in splits:
            for spec in ABLATIONS:
                metrics = summary[split][spec.name]
                writer.writerow(
                    {
                        "split": split,
                        "ablation": spec.name,
                        "description": metrics["description"],
                        "intervention": metrics["intervention"],
                        "feature_group": metrics["feature_group"],
                        "n_features_changed": metrics["n_features_changed"],
                        "qwk": metrics["qwk"],
                        "delta_qwk_vs_full": metrics["delta_qwk_vs_full"],
                        "accuracy": metrics["accuracy"],
                        "delta_accuracy_vs_full": metrics["delta_accuracy_vs_full"],
                        "f1_macro": metrics.get("f1_macro"),
                        "auc_macro": metrics.get("auc_macro"),
                    }
                )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()

    feature_dir = Path(cfg.feature_dir) / "aggregated"
    val_features_npz = np.load(feature_dir / "val_features.npz")
    expected_feature_dim = int(val_features_npz["features"].shape[1])
    ckpt_path = _resolve_hybrid_checkpoint(cfg, expected_feature_dim=expected_feature_dim)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError("No KL hybrid checkpoint found for feature ablation.")

    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    checkpoint_feature_dim = _checkpoint_feature_dim(checkpoint)
    model_cfg = _model_cfg_from_checkpoint(cfg, checkpoint)
    if checkpoint_feature_dim is not None and checkpoint_feature_dim != expected_feature_dim:
        raise ValueError(
            f"Hybrid checkpoint feature_dim={checkpoint_feature_dim} does not match "
            f"current aggregated feature_dim={expected_feature_dim}: {ckpt_path}"
        )
    model = HybridKLClassifier(model_cfg)
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.to(device)
    model.eval()

    result_dir = Path(cfg.result_dir) / "kl_hybrid_feature_ablation"
    result_dir.mkdir(parents=True, exist_ok=True)
    transform = get_eval_transforms(cfg)
    eval_batch_size = int(getattr(cfg, "eval_batch_size", 4))
    split_override = str(getattr(cfg, "ablation_splits", "val,test"))
    splits = [item.strip() for item in split_override.split(",") if item.strip()]
    if not splits:
        raise ValueError("ablation_splits must include at least one split.")

    summary = {
        "checkpoint": str(ckpt_path),
        "feature_dim": expected_feature_dim,
        "note": (
            "Zero ablations are applied after training-set z-score normalization, "
            "so zero_all is equivalent to replacing raw features with the training-set mean. "
            "Permutation ablations preserve the feature distribution but break image-feature alignment."
        ),
    }
    for split in splits:
        print(f"Evaluating split={split} with batch_size={eval_batch_size}")
        dataset = KLHybridDataset(
            cfg.data.root,
            split,
            str(feature_dir / f"{split}_features.npz"),
            transform,
        )
        metrics_by_ablation, payload_by_ablation = _evaluate_split_ablation(
            model,
            dataset,
            cfg.data.class_names,
            device,
            batch_size=eval_batch_size,
            num_workers=cfg.data.num_workers,
            seed=int(cfg.seed) + (0 if split == "val" else 1000),
        )
        summary[split] = metrics_by_ablation
        for ablation_name, payload in payload_by_ablation.items():
            np.savez(
                str(result_dir / f"hybrid_{split}_{ablation_name}_predictions.npz"),
                **payload,
            )

    json_path = result_dir / "kl_hybrid_feature_ablation.json"
    csv_path = result_dir / "kl_hybrid_feature_ablation.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(summary, csv_path, splits)
    print(json.dumps(summary, indent=2))
    print(f"Saved KL hybrid feature ablation summary to {json_path}")
    print(f"Saved KL hybrid feature ablation table to {csv_path}")
    clear_memory()


if __name__ == "__main__":
    main()
