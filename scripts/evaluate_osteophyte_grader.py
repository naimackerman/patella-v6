"""Evaluate per-site osteophyte grading checkpoints on reviewed labels."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from src.data.roi_dataset import ROIDataset
from src.data.transforms import get_eval_transforms
from src.models.osteophyte_grader import OsteophyteGrader
from src.utils.annotation_paths import EXPANDED_SOURCES, MANUAL_SOURCES, resolve_annotation_csv, select_label_subset
from src.utils.checkpoints import extract_model_state_dict, load_checkpoint, resolve_osteophyte_checkpoint_paths
from src.utils.device import get_device
from src.utils.seed import seed_everything


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "se_resnet50.yaml")


def _site_metrics(model_cfg: DictConfig, targets: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    metrics = {
        "num_samples": int(len(targets)),
        "kappa": float(cohen_kappa_score(targets, preds, weights="quadratic")),
        "balanced_accuracy": float(balanced_accuracy_score(targets, preds)),
        "confusion_matrix": confusion_matrix(targets, preds, labels=list(range(model_cfg.num_classes_per_head))).tolist(),
    }
    if len(np.unique(targets)) > 1:
        try:
            metrics["auc_macro"] = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)
    device = get_device()

    label_mode = getattr(cfg.training, "label_mode", "manual")
    allow_bootstrap_fallback = bool(getattr(cfg.training, "allow_bootstrap_fallback", False))
    labels_csv = resolve_annotation_csv(
        cfg.annotation_dir,
        "osteophyte_labels",
        mode=label_mode,
        allow_bootstrap_fallback=allow_bootstrap_fallback,
    )
    labels_df = pd.read_csv(labels_csv)
    _, source_mode = select_label_subset(labels_df, mode=label_mode, allow_bootstrap_fallback=allow_bootstrap_fallback)
    confidence_cfg = getattr(cfg.training, "annotation_confidence", {})
    min_eval_confidence = str(confidence_cfg.get("min_eval", "low"))
    confidence_weights = {
        "low": float(confidence_cfg.get("weight_low", 0.5)),
        "medium": float(confidence_cfg.get("weight_medium", 0.75)),
        "high": float(confidence_cfg.get("weight_high", 1.0)),
    }
    allowed_sources = None
    if source_mode.startswith("manual_only"):
        allowed_sources = sorted(MANUAL_SOURCES)
    elif source_mode == "manual_plus_high_confidence":
        allowed_sources = sorted(EXPANDED_SOURCES)

    roi_dir = Path(str(getattr(cfg, "osteophyte_roi_dir", Path(cfg.feature_dir) / "rois")))
    ckpt_dir = Path(cfg.checkpoint_dir) / "osteophyte"
    transform = get_eval_transforms(cfg)
    site_summaries = {}
    override_cfg = getattr(cfg.training, "osteophyte_checkpoint_overrides", {})
    selection_cfg = getattr(cfg.training, "osteophyte_checkpoint_selection", {})
    override_paths = {
        str(site_name): str(path_value)
        for site_name, path_value in dict(override_cfg).items()
        if path_value is not None
    } if override_cfg is not None else {}
    prefer_refined = bool(selection_cfg.get("prefer_refined", True))
    force_multitask_sites = [str(site_name) for site_name in selection_cfg.get("force_multitask_sites", [])]
    force_refined_sites = [str(site_name) for site_name in selection_cfg.get("force_refined_sites", [])]
    resolved_ckpts = resolve_osteophyte_checkpoint_paths(
        ckpt_dir,
        OsteophyteGrader.SITES,
        prefer_refined=prefer_refined,
        force_multitask_sites=force_multitask_sites,
        force_refined_sites=force_refined_sites,
        override_paths_by_site=override_paths,
    )
    model_cache: dict[str, OsteophyteGrader] = {}

    def load_model(ckpt_path: Path) -> OsteophyteGrader:
        cache_key = str(ckpt_path)
        if cache_key not in model_cache:
            checkpoint = load_checkpoint(ckpt_path, map_location=device)
            state_dict = extract_model_state_dict(checkpoint)
            current_model_cfg = model_cfg
            has_old_heads = any(
                key.startswith("heads.") and key.count(".") == 2
                for key in state_dict
            )
            if has_old_heads:
                current_model_cfg = OmegaConf.merge(model_cfg, {"use_mlp_heads": False})
            model = OsteophyteGrader(current_model_cfg)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            model_cache[cache_key] = model
        return model_cache[cache_key]

    for site in OsteophyteGrader.SITES:
        ckpt_meta = resolved_ckpts.get(site)
        if ckpt_meta is None:
            continue
        ckpt_path = Path(ckpt_meta["path"])
        model = load_model(ckpt_path)

        split_metrics = {}
        for split in ("val", "test"):
            dataset = ROIDataset(
                str(roi_dir / split),
                str(labels_csv),
                site,
                transform,
                allowed_label_sources=allowed_sources,
                min_confidence=min_eval_confidence,
                confidence_weights=confidence_weights,
            )
            if len(dataset) == 0:
                continue
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=cfg.data.num_workers)
            preds = []
            probs = []
            targets = []
            with torch.no_grad():
                for images, labels, _, _ in loader:
                    images = images.to(device)
                    logits = model.forward_single(images, site)
                    preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                    probs.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
                    targets.extend(labels.tolist())
            current_preds = np.asarray(preds, dtype=np.int64)
            current_probs = np.asarray(probs, dtype=np.float64)
            current_targets = np.asarray(targets, dtype=np.int64)
            split_metrics[split] = _site_metrics(model_cfg, current_targets, current_preds, current_probs)
        if split_metrics:
            site_summaries[site] = {
                "checkpoint": str(ckpt_path),
                "checkpoint_mode": str(ckpt_meta["mode"]),
                **split_metrics,
            }

    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / "osteophyte_evaluation.json"
    out_path.write_text(json.dumps(site_summaries, indent=2), encoding="utf-8")
    print(json.dumps(site_summaries, indent=2))
    print(f"Saved osteophyte evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
