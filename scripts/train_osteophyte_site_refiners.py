"""Refine per-site osteophyte models from the best multitask checkpoint."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from src.data.roi_dataset import ROIDataset
from src.data.sampler import create_weighted_sampler
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.osteophyte_grader import OsteophyteGrader
from src.modules.osteophyte_module import OsteophyteModule
from src.utils.annotation_paths import EXPANDED_SOURCES, MANUAL_SOURCES, resolve_annotation_csv, select_label_subset
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.class_weights import compute_balanced_class_weights
from src.utils.device import clear_memory, get_accelerator, get_device
from src.utils.lightning import build_loggers
from src.utils.seed import seed_everything


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _ensure_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg
    model_cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "se_resnet50.yaml")
    with open_dict(cfg):
        cfg.model = model_cfg
    return cfg


def _apply_osteophyte_training_profile(cfg: DictConfig) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    profile_cfg = getattr(cfg_copy.training, "osteophyte_optimization", {})
    if profile_cfg is None:
        return cfg_copy
    cfg_copy.training.max_epochs = int(profile_cfg.get("max_epochs", cfg_copy.training.max_epochs))
    cfg_copy.training.learning_rate = float(profile_cfg.get("learning_rate", cfg_copy.training.learning_rate))
    cfg_copy.training.weight_decay = float(profile_cfg.get("weight_decay", cfg_copy.training.weight_decay))
    if "scheduler_params" in cfg_copy.training:
        cfg_copy.training.scheduler_params.T_max = int(
            profile_cfg.get("max_epochs", cfg_copy.training.max_epochs)
        )
    return cfg_copy


def _resolve_site_base_checkpoint_dirs(cfg: DictConfig) -> dict[str, Path]:
    """Resolve the checkpoint directory to use for each osteophyte site."""
    ref_cfg = getattr(cfg.training, "osteophyte_refinement", {})
    default_override = getattr(ref_cfg, "base_checkpoint_dir", None)
    default_root = Path(str(cfg.checkpoint_dir if default_override in (None, "", "null") else default_override))
    per_site_cfg = getattr(ref_cfg, "base_checkpoint_dirs_by_site", {})

    resolved = {site: default_root for site in OsteophyteGrader.SITES}
    if per_site_cfg is None:
        return resolved

    if isinstance(per_site_cfg, DictConfig):
        items = per_site_cfg.items()
    else:
        items = dict(per_site_cfg).items()
    for site, override_dir in items:
        if override_dir is None:
            continue
        resolved[str(site)] = Path(str(override_dir))
    return resolved


def _freeze_non_target_parameters(module: OsteophyteModule, site: str, freeze_backbone: bool, freeze_non_target_heads: bool):
    if freeze_backbone:
        for parameter in module.model.backbone.parameters():
            parameter.requires_grad = False
    if freeze_non_target_heads:
        for head_site, head in module.model.heads.items():
            requires_grad = head_site == site
            for parameter in head.parameters():
                parameter.requires_grad = requires_grad


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg = _ensure_model_cfg(cfg)
    cfg = _apply_osteophyte_training_profile(cfg)
    seed_everything(cfg.seed)
    device = get_device()
    roi_dir = Path(cfg.feature_dir) / "rois"

    label_mode = getattr(cfg.training, "label_mode", "manual")
    allow_bootstrap_fallback = bool(getattr(cfg.training, "allow_bootstrap_fallback", False))
    confidence_cfg = getattr(cfg.training, "annotation_confidence", {})
    min_train_confidence = str(confidence_cfg.get("min_train", "low"))
    min_eval_confidence = str(confidence_cfg.get("min_eval", "low"))
    confidence_weights = {
        "low": float(confidence_cfg.get("weight_low", 0.5)),
        "medium": float(confidence_cfg.get("weight_medium", 0.75)),
        "high": float(confidence_cfg.get("weight_high", 1.0)),
    }

    labels_csv = resolve_annotation_csv(
        cfg.annotation_dir,
        "osteophyte_labels",
        mode=label_mode,
        allow_bootstrap_fallback=allow_bootstrap_fallback,
    )
    labels_df = np.asarray([])
    allowed_sources = None
    source_df = None
    if Path(labels_csv).exists():
        import pandas as pd

        source_df = pd.read_csv(labels_csv)
        labels_df, source_mode = select_label_subset(
            source_df,
            mode=label_mode,
            allow_bootstrap_fallback=allow_bootstrap_fallback,
        )
        if source_mode.startswith("manual_only"):
            allowed_sources = sorted(MANUAL_SOURCES)
        elif source_mode == "manual_plus_high_confidence":
            allowed_sources = sorted(EXPANDED_SOURCES)

    train_transform = get_train_transforms(cfg)
    val_transform = get_eval_transforms(cfg)
    ref_cfg = getattr(cfg.training, "osteophyte_refinement", {})
    selected_sites = list(ref_cfg.get("sites", [])) or list(OsteophyteGrader.SITES)
    learning_rate = float(ref_cfg.get("learning_rate", cfg.training.learning_rate))
    weight_decay = float(ref_cfg.get("weight_decay", cfg.training.weight_decay))
    patience = int(ref_cfg.get("patience", 12))
    freeze_non_target_heads = bool(ref_cfg.get("freeze_non_target_heads", True))
    freeze_backbone = bool(ref_cfg.get("freeze_backbone", False))
    class_balance_cfg = getattr(cfg.training, "osteophyte_class_balance", {})
    class_balance_power = float(class_balance_cfg.get("power", 1.0))
    class_balance_normalize = bool(class_balance_cfg.get("normalize", True))
    sampling_cfg = getattr(cfg.training, "osteophyte_sampling", {})
    sampling_use_confidence = bool(sampling_cfg.get("use_confidence_weights", False))
    sampling_multiplier_power = float(sampling_cfg.get("confidence_power", 1.0))
    sampling_max_ratio = sampling_cfg.get("max_weight_ratio_to_median", None)
    output_ckpt_dir = Path(cfg.checkpoint_dir) / "osteophyte"
    base_checkpoint_dirs = _resolve_site_base_checkpoint_dirs(cfg)
    base_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
    base_checkpoint_paths: dict[str, Path] = {}
    for site in selected_sites:
        source_ckpt_dir = base_checkpoint_dirs.get(site, Path(cfg.checkpoint_dir)) / "osteophyte"
        multitask_ckpt = find_best_lightning_checkpoint(
            source_ckpt_dir,
            pattern="osp-multitask-*.ckpt",
            monitor="val_kappa_mean",
        )
        if multitask_ckpt is None:
            raise FileNotFoundError(f"No multitask osteophyte checkpoint found in {source_ckpt_dir} for site={site}")
        base_checkpoint_paths[site] = multitask_ckpt
        base_state_dicts[site] = extract_model_state_dict(load_checkpoint(multitask_ckpt, map_location=device))

    print(f"Sites: {selected_sites}")
    print(f"Refinement LR={learning_rate} weight_decay={weight_decay} patience={patience}")
    print(f"Freeze non-target heads={freeze_non_target_heads} freeze backbone={freeze_backbone}")
    print(
        "Refinement sampler settings: "
        f"use_confidence_weights={sampling_use_confidence}, "
        f"confidence_power={sampling_multiplier_power}, "
        f"max_weight_ratio_to_median={sampling_max_ratio}"
    )

    for site in selected_sites:
        print(f"\n{'='*60}")
        print(f"Refining osteophyte site: {site}")
        print(f"{'='*60}")
        print(f"Base multitask checkpoint: {base_checkpoint_paths[site]}")

        train_ds = ROIDataset(
            str(roi_dir / "train"),
            str(labels_csv),
            site,
            train_transform,
            allowed_label_sources=allowed_sources,
            min_confidence=min_train_confidence,
            confidence_weights=confidence_weights,
        )
        val_ds = ROIDataset(
            str(roi_dir / "val"),
            str(labels_csv),
            site,
            val_transform,
            allowed_label_sources=allowed_sources,
            min_confidence=min_eval_confidence,
            confidence_weights=confidence_weights,
        )
        if len(train_ds) == 0 or len(val_ds) == 0:
            raise ValueError(f"No samples available for site={site}")

        counts = np.bincount(train_ds.labels, minlength=cfg.model.num_classes_per_head).astype(int)
        class_weights = compute_balanced_class_weights(
            train_ds.labels,
            num_classes=cfg.model.num_classes_per_head,
            power=class_balance_power,
            normalize=class_balance_normalize,
        )
        print(f"{site} train counts: {counts.tolist()}")
        print(f"{site} class weights: {[round(float(x), 4) for x in class_weights.tolist()]}")

        site_cfg = _clone_cfg(cfg)
        site_cfg.training.learning_rate = learning_rate
        site_cfg.training.weight_decay = weight_decay
        site_cfg.output_dir = str(Path(cfg.output_dir) / "site_refinement" / site)

        module = OsteophyteModule(
            site_cfg,
            site=site,
            class_weights_by_site=torch.tensor(class_weights, dtype=torch.float32),
        )
        module.model.load_state_dict(base_state_dicts[site], strict=True)
        _freeze_non_target_parameters(
            module,
            site=site,
            freeze_backbone=freeze_backbone,
            freeze_non_target_heads=freeze_non_target_heads,
        )

        sampler_kwargs = {}
        if sampling_use_confidence:
            sampler_kwargs["sample_weight_multipliers"] = train_ds.sample_weights
            sampler_kwargs["multiplier_power"] = sampling_multiplier_power
        if sampling_max_ratio is not None:
            sampler_kwargs["max_weight_ratio_to_median"] = float(sampling_max_ratio)
        sampler = create_weighted_sampler(train_ds.labels, **sampler_kwargs)
        train_loader = DataLoader(
            train_ds, batch_size=8, sampler=sampler,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.num_workers > 0,
            pin_memory=cfg.data.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=8, shuffle=False,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.num_workers > 0,
            pin_memory=cfg.data.num_workers > 0,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=get_accelerator(),
            precision=cfg.training.precision,
            accumulate_grad_batches=4,
            gradient_clip_val=cfg.training.gradient_clip_val,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_kappa", patience=patience, mode="max",
                ),
                pl.callbacks.ModelCheckpoint(
                    dirpath=output_ckpt_dir,
                    filename=f"osp-refined-{site}" + "-{epoch:03d}-{val_kappa:.4f}",
                    monitor="val_kappa",
                    mode="max",
                    save_top_k=3,
                ),
            ],
            default_root_dir=site_cfg.output_dir,
            logger=build_loggers(site_cfg, f"osteophyte_refined_{site}"),
        )

        trainer.fit(module, train_loader, val_loader)
        clear_memory()


if __name__ == "__main__":
    main()
