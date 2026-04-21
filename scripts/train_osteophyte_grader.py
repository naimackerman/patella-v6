"""Train SE-ResNet-50 osteophyte grader on extracted ROI patches."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from torch.utils.data import DataLoader

from src.modules.osteophyte_module import OsteophyteModule
from src.data.roi_dataset import ROIDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.sampler import create_multitask_weighted_sampler, create_weighted_sampler
from src.utils.annotation_paths import EXPANDED_SOURCES, MANUAL_SOURCES, resolve_annotation_csv, select_label_subset
from src.utils.class_weights import compute_multitask_balanced_class_weights
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_accelerator, clear_memory
from src.utils.lightning import build_loggers
from src.utils.seed import seed_everything


def _allow_safe_resume_checkpoint_types() -> None:
    if not hasattr(torch.serialization, "add_safe_globals"):
        return
    torch.serialization.add_safe_globals([
        DictConfig,
        ListConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        defaultdict,
        list,
        dict,
        int,
        Any,
    ])


def _load_compatible_model_weights(model: torch.nn.Module, checkpoint_path: Path, context: str) -> None:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_state = extract_model_state_dict(checkpoint)
    model_state = model.state_dict()

    compatible_state = {}
    skipped_missing = []
    skipped_shape = []
    for key, value in checkpoint_state.items():
        target_value = model_state.get(key)
        if target_value is None:
            skipped_missing.append(key)
            continue
        if target_value.shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(target_value.shape)))
            continue
        compatible_state[key] = value

    if not compatible_state:
        raise RuntimeError(
            f"No compatible parameters found while warm-starting {context} from {checkpoint_path}."
        )

    model.load_state_dict(compatible_state, strict=False)

    loaded_count = len(compatible_state)
    missing_count = len(skipped_missing)
    shape_mismatch_count = len(skipped_shape)
    print(
        f"Loaded {loaded_count} compatible parameter tensors for {context} from {checkpoint_path}"
    )
    if missing_count or shape_mismatch_count:
        print(
            f"Skipped {missing_count} missing keys and {shape_mismatch_count} shape-mismatched keys "
            f"while warm-starting {context}"
        )

    legacy_linear_head_keys = [
        key for key in checkpoint_state
        if key.startswith("heads.") and key.count(".") == 2 and key.endswith((".weight", ".bias"))
    ]
    skipped_head_keys = [
        key for key, _, _ in skipped_shape
        if key.startswith("heads.")
    ] + [key for key in skipped_missing if key.startswith("heads.")]
    if legacy_linear_head_keys and skipped_head_keys:
        print(
            "Warm-start checkpoint uses legacy linear osteophyte heads; "
            "the main-study MLP heads remain freshly initialized."
        )


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


def _build_dataloader(dataset, batch_size: int, sampler=None, shuffle: bool = False, num_workers: int = 0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=num_workers > 0,
    )


def _build_multitask_trainer(cfg: DictConfig, patience: int | None = None, logger_name: str = "osteophyte_multitask"):
    profile_cfg = getattr(cfg.training, "osteophyte_optimization", {})
    default_patience = int(profile_cfg.get("early_stopping_patience", 20))
    return pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=get_accelerator(),
        precision=cfg.training.precision,
        accumulate_grad_batches=4,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_kappa_mean", patience=patience or default_patience, mode="max",
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.checkpoint_dir) / "osteophyte",
                filename="osp-multitask-{epoch:03d}-{val_kappa_mean:.4f}",
                monitor="val_kappa_mean",
                mode="max",
                save_top_k=3,
            ),
        ],
        default_root_dir=cfg.output_dir,
        logger=build_loggers(cfg, logger_name),
    )


def _build_site_trainer(cfg: DictConfig, site: str, patience: int, logger_name: str):
    return pl.Trainer(
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
                dirpath=Path(cfg.checkpoint_dir) / "osteophyte",
                filename=f"osp-refined-{site}" + "-{epoch:03d}-{val_kappa:.4f}",
                monitor="val_kappa",
                mode="max",
                save_top_k=3,
            ),
        ],
        default_root_dir=cfg.output_dir,
        logger=build_loggers(cfg, logger_name),
    )


def _freeze_non_target_parameters(module: OsteophyteModule, site: str, freeze_backbone: bool, freeze_non_target_heads: bool):
    if freeze_backbone:
        for parameter in module.model.backbone.parameters():
            parameter.requires_grad = False
    if freeze_non_target_heads:
        for head_site, head in module.model.heads.items():
            requires_grad = head_site == site
            for parameter in head.parameters():
                parameter.requires_grad = requires_grad


def _compute_single_site_class_weights(labels: np.ndarray, num_classes: int, power: float, normalize: bool) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = (1.0 / counts) ** power
    if normalize:
        weights = weights * (num_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def _clone_cfg_with_training_overrides(cfg: DictConfig, learning_rate: float, weight_decay: float) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_copy.training.learning_rate = float(learning_rate)
    cfg_copy.training.weight_decay = float(weight_decay)
    return cfg_copy


def _clone_cfg_for_osteophyte_transforms(cfg: DictConfig) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if bool(getattr(cfg.training, "osteophyte_disable_horizontal_flip", False)):
        cfg_copy.preprocessing.augmentation.horizontal_flip_p = 0.0
        print("Osteophyte training augmentation: horizontal_flip_p=0.0")
    return cfg_copy


def _train_multitask(
    cfg: DictConfig,
    labels_csv: Path,
    roi_dir: Path,
    train_transform,
    val_transform,
    allowed_sources,
    min_train_confidence: str,
    min_eval_confidence: str,
    confidence_weights: dict[str, float],
    init_checkpoint: Path | None = None,
    resume_checkpoint: Path | None = None,
) -> Path | None:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.training.early_stopping.monitor = "val_kappa_mean"
    cfg.training.early_stopping.mode = "max"
    cfg.training.scheduler_monitor = "val_kappa_mean"

    train_ds = ROIDataset(
        str(roi_dir / "train"),
        str(labels_csv),
        None,
        train_transform,
        allowed_label_sources=allowed_sources,
        min_confidence=min_train_confidence,
        confidence_weights=confidence_weights,
    )
    val_ds = ROIDataset(
        str(roi_dir / "val"),
        str(labels_csv),
        None,
        val_transform,
        allowed_label_sources=allowed_sources,
        min_confidence=min_eval_confidence,
        confidence_weights=confidence_weights,
    )

    if len(train_ds) == 0:
        raise ValueError("No multitask osteophyte training samples available.")

    class_balance_cfg = getattr(cfg.training, "osteophyte_class_balance", {})
    class_weights_tensor = None
    if bool(class_balance_cfg.get("enabled", False)):
        class_weights = compute_multitask_balanced_class_weights(
            train_ds.labels,
            num_classes=cfg.model.num_classes_per_head,
            power=float(class_balance_cfg.get("power", 1.0)),
            normalize=bool(class_balance_cfg.get("normalize", True)),
        )
        site_names = list(train_ds.SITES)
        for site_idx, site_name in enumerate(site_names):
            counts = np.bincount(
                train_ds.labels[:, site_idx],
                minlength=cfg.model.num_classes_per_head,
            ).astype(int)
            current_weights = [round(float(x), 4) for x in class_weights[site_idx].tolist()]
            print(f"{site_name} class counts: {counts.tolist()}")
            print(f"{site_name} class weights: {current_weights}")
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    module = OsteophyteModule(cfg, site=None, class_weights_by_site=class_weights_tensor)
    if resume_checkpoint is not None:
        print(f"Resuming multitask osteophyte model from checkpoint: {resume_checkpoint}")
        if init_checkpoint is not None:
            print("Ignoring osteophyte warm-start checkpoint because resume checkpoint takes precedence.")
    elif init_checkpoint is not None:
        print(f"Warm-starting multitask osteophyte model from: {init_checkpoint}")
        _load_compatible_model_weights(
            module.model,
            init_checkpoint,
            context="multitask osteophyte model",
        )

    sampling_cfg = getattr(cfg.training, "osteophyte_sampling", {})
    sampling_strategy = str(sampling_cfg.get("strategy", "max_severity")).lower()
    sampling_use_confidence = bool(sampling_cfg.get("use_confidence_weights", False))
    sampling_multiplier_power = float(sampling_cfg.get("confidence_power", 1.0))
    sampling_max_ratio = sampling_cfg.get("max_weight_ratio_to_median", None)
    sampling_kwargs = {}
    if sampling_use_confidence:
        sampling_kwargs["sample_weight_multipliers"] = train_ds.sample_weights
        sampling_kwargs["multiplier_power"] = sampling_multiplier_power
    if sampling_max_ratio is not None:
        sampling_kwargs["max_weight_ratio_to_median"] = float(sampling_max_ratio)
    if sampling_strategy == "max_severity":
        sampler = create_weighted_sampler(
            train_ds.sampling_labels,
            **sampling_kwargs,
        )
    else:
        sampler = create_multitask_weighted_sampler(
            train_ds.labels,
            num_classes=cfg.model.num_classes_per_head,
            strategy=sampling_strategy,
            power=float(sampling_cfg.get("power", 1.0)),
            normalize=bool(sampling_cfg.get("normalize", True)),
            **sampling_kwargs,
        )
    print(
        "Osteophyte multitask sampling strategy: "
        f"{sampling_strategy} "
        f"(use_confidence_weights={sampling_use_confidence}, "
        f"confidence_power={sampling_multiplier_power}, "
        f"max_weight_ratio_to_median={sampling_max_ratio})"
    )

    train_loader = _build_dataloader(
        train_ds, batch_size=8, sampler=sampler, num_workers=cfg.data.num_workers
    )
    val_loader = _build_dataloader(
        val_ds, batch_size=8, shuffle=False, num_workers=cfg.data.num_workers
    )

    trainer = _build_multitask_trainer(cfg)
    if resume_checkpoint is not None:
        _allow_safe_resume_checkpoint_types()
        trainer.fit(module, train_loader, val_loader, ckpt_path=str(resume_checkpoint))
    else:
        trainer.fit(module, train_loader, val_loader)
    clear_memory()
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_path = callback.best_model_path or callback.last_model_path
            if best_path:
                return Path(best_path)
    return None


def _resolve_base_multitask_checkpoint(cfg: DictConfig, site: str) -> Path | None:
    refinement_cfg = getattr(cfg.training, "osteophyte_refinement", {})
    by_site_cfg = refinement_cfg.get("base_checkpoint_dirs_by_site", {})
    base_dir_override = None
    if by_site_cfg is not None and site in by_site_cfg and by_site_cfg[site] is not None:
        base_dir_override = Path(str(by_site_cfg[site]))
    elif refinement_cfg.get("base_checkpoint_dir"):
        base_dir_override = Path(str(refinement_cfg.get("base_checkpoint_dir")))

    base_dir = base_dir_override or (Path(cfg.checkpoint_dir) / "osteophyte")
    if base_dir.is_file():
        return base_dir
    return find_best_lightning_checkpoint(
        base_dir,
        pattern="osp-multitask-*.ckpt",
        monitor="val_kappa_mean",
    )


def _resolve_warm_start_checkpoint(cfg: DictConfig) -> Path | None:
    raw_value = getattr(cfg.training, "osteophyte_warm_start_checkpoint", None)
    if raw_value is None:
        return None
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text.lower() in {"none", "null"}:
        return None

    checkpoint_path = Path(raw_text)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Warm-start checkpoint not found for "
            f"training.osteophyte_warm_start_checkpoint: {checkpoint_path}"
        )
    return checkpoint_path


def _resolve_resume_multitask_checkpoint(cfg: DictConfig) -> Path | None:
    raw_value = getattr(cfg.training, "osteophyte_resume_multitask_checkpoint", None)
    if raw_value is None:
        return None
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text.lower() in {"none", "null"}:
        return None

    checkpoint_path = Path(raw_text)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Resume checkpoint not found for "
            f"training.osteophyte_resume_multitask_checkpoint: {checkpoint_path}"
        )
    return checkpoint_path


def _refine_site(
    cfg: DictConfig,
    site: str,
    labels_csv: Path,
    roi_dir: Path,
    train_transform,
    val_transform,
    allowed_sources,
    min_train_confidence: str,
    min_eval_confidence: str,
    confidence_weights: dict[str, float],
    base_ckpt_override: Path | None = None,
):
    refinement_cfg = getattr(cfg.training, "osteophyte_refinement", {})
    base_ckpt_path = base_ckpt_override or _resolve_base_multitask_checkpoint(cfg, site)
    if base_ckpt_path is None:
        raise FileNotFoundError(
            f"No multitask osteophyte checkpoint found for refinement site={site}. "
            "Train the base multitask model first or set training.osteophyte_refinement.base_checkpoint_dir."
        )

    print(f"\n{'='*60}")
    print(f"Refining osteophyte grader: {site}")
    print(f"Base checkpoint: {base_ckpt_path}")
    print(f"{'='*60}")

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
    if len(train_ds) == 0:
        raise ValueError(f"No osteophyte refinement samples available for site={site}.")

    class_balance_cfg = getattr(cfg.training, "osteophyte_class_balance", {})
    class_weights_tensor = None
    if bool(class_balance_cfg.get("enabled", False)):
        class_weights_tensor = _compute_single_site_class_weights(
            train_ds.labels,
            num_classes=cfg.model.num_classes_per_head,
            power=float(class_balance_cfg.get("power", 1.0)),
            normalize=bool(class_balance_cfg.get("normalize", True)),
        )
        counts = np.bincount(train_ds.labels, minlength=cfg.model.num_classes_per_head).astype(int)
        print(f"{site} refinement class counts: {counts.tolist()}")
        print(f"{site} refinement class weights: {[round(float(x), 4) for x in class_weights_tensor.tolist()]}")

    refine_cfg = _clone_cfg_with_training_overrides(
        cfg,
        learning_rate=float(refinement_cfg.get("learning_rate", cfg.training.learning_rate)),
        weight_decay=float(refinement_cfg.get("weight_decay", cfg.training.weight_decay)),
    )
    refine_cfg.training.early_stopping.monitor = "val_kappa"
    refine_cfg.training.early_stopping.mode = "max"
    refine_cfg.training.scheduler_monitor = "val_kappa"
    module = OsteophyteModule(refine_cfg, site=site, class_weights_by_site=class_weights_tensor)
    checkpoint = load_checkpoint(base_ckpt_path, map_location="cpu")
    module.model.load_state_dict(extract_model_state_dict(checkpoint), strict=True)
    _freeze_non_target_parameters(
        module,
        site=site,
        freeze_backbone=bool(refinement_cfg.get("freeze_backbone", False)),
        freeze_non_target_heads=bool(refinement_cfg.get("freeze_non_target_heads", True)),
    )

    sampling_cfg = getattr(cfg.training, "osteophyte_sampling", {})
    sampling_use_confidence = bool(sampling_cfg.get("use_confidence_weights", False))
    sampling_multiplier_power = float(sampling_cfg.get("confidence_power", 1.0))
    sampling_max_ratio = sampling_cfg.get("max_weight_ratio_to_median", None)
    sampling_kwargs = {}
    if sampling_use_confidence:
        sampling_kwargs["sample_weight_multipliers"] = train_ds.sample_weights
        sampling_kwargs["multiplier_power"] = sampling_multiplier_power
    if sampling_max_ratio is not None:
        sampling_kwargs["max_weight_ratio_to_median"] = float(sampling_max_ratio)
    sampler = create_weighted_sampler(train_ds.sampling_labels, **sampling_kwargs)

    train_loader = _build_dataloader(
        train_ds, batch_size=8, sampler=sampler, num_workers=cfg.data.num_workers
    )
    val_loader = _build_dataloader(
        val_ds, batch_size=8, shuffle=False, num_workers=cfg.data.num_workers
    )
    trainer = _build_site_trainer(
        refine_cfg,
        site=site,
        patience=int(refinement_cfg.get("patience", 12)),
        logger_name=f"osteophyte_refine_{site}",
    )
    trainer.fit(module, train_loader, val_loader)
    clear_memory()


def _has_base_multitask_checkpoint(cfg: DictConfig, refinement_sites: list[str]) -> bool:
    if _resolve_resume_multitask_checkpoint(cfg) is not None:
        return False
    if bool(getattr(cfg.training, "osteophyte_force_retrain_multitask", False)):
        return False
    if not refinement_sites:
        return _resolve_base_multitask_checkpoint(cfg, site=ROIDataset.SITES[0]) is not None
    for site in refinement_sites:
        if _resolve_base_multitask_checkpoint(cfg, site=site) is None:
            return False
    return True


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg = _ensure_model_cfg(cfg)
    cfg = _apply_osteophyte_training_profile(cfg)
    seed_everything(cfg.seed)

    roi_dir = Path(str(getattr(cfg, "osteophyte_roi_dir", Path(cfg.feature_dir) / "rois")))
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

    raw_labels_df = pd.read_csv(labels_csv)
    labels_df, source_mode = select_label_subset(
        raw_labels_df,
        mode=label_mode,
        allow_bootstrap_fallback=allow_bootstrap_fallback,
    )
    if len(labels_df) == 0:
        raise ValueError(f"No osteophyte labels available in {labels_csv} for mode={label_mode}")

    allowed_sources = None
    if source_mode.startswith("manual_only"):
        allowed_sources = sorted(MANUAL_SOURCES)
    elif source_mode == "manual_plus_high_confidence":
        allowed_sources = sorted(EXPANDED_SOURCES)
    elif source_mode.startswith("bootstrap_fallback"):
        allowed_sources = None

    print(f"Using osteophyte labels: {labels_csv}")
    print(f"Label subset mode: {source_mode} ({len(labels_df)} rows)")
    print(f"Osteophyte confidence policy: train>={min_train_confidence}, eval>={min_eval_confidence}, weights={confidence_weights}")
    if allowed_sources is not None:
        print(f"Allowed label sources: {allowed_sources}")
    if source_mode.startswith("bootstrap_fallback") or source_mode == "all_labels":
        print("Warning: no reviewed/manual osteophyte labels found; this run remains a bootstrap baseline.")

    warm_start_path = _resolve_warm_start_checkpoint(cfg)
    if warm_start_path is not None:
        print(f"Osteophyte warm-start checkpoint: {warm_start_path}")
    resume_multitask_path = _resolve_resume_multitask_checkpoint(cfg)
    if resume_multitask_path is not None:
        print(f"Osteophyte resume checkpoint: {resume_multitask_path}")

    transform_cfg = _clone_cfg_for_osteophyte_transforms(cfg)
    train_transform = get_train_transforms(transform_cfg)
    val_transform = get_eval_transforms(transform_cfg)

    refinement_cfg = getattr(cfg.training, "osteophyte_refinement", {})
    refinement_sites = [str(site) for site in refinement_cfg.get("sites", [])]
    if refinement_sites:
        base_ckpt_path = None
        for site in refinement_sites:
            if site not in ROIDataset.SITES:
                raise ValueError(f"Unknown osteophyte refinement site: {site}")
        if not _has_base_multitask_checkpoint(cfg, refinement_sites):
            print("\n" + "=" * 60)
            print("Training base multitask osteophyte model before refinement")
            print("=" * 60)
            base_ckpt_path = _train_multitask(
                cfg,
                labels_csv=Path(labels_csv),
                roi_dir=roi_dir,
                train_transform=train_transform,
                val_transform=val_transform,
                allowed_sources=allowed_sources,
                min_train_confidence=min_train_confidence,
                min_eval_confidence=min_eval_confidence,
                confidence_weights=confidence_weights,
                init_checkpoint=warm_start_path,
                resume_checkpoint=resume_multitask_path,
            )
        if base_ckpt_path is None:
            base_ckpt_path = _resolve_base_multitask_checkpoint(cfg, refinement_sites[0])
        for site in refinement_sites:
            _refine_site(
                cfg,
                site=site,
                labels_csv=Path(labels_csv),
                roi_dir=roi_dir,
                train_transform=train_transform,
                val_transform=val_transform,
                allowed_sources=allowed_sources,
                min_train_confidence=min_train_confidence,
                min_eval_confidence=min_eval_confidence,
                confidence_weights=confidence_weights,
                base_ckpt_override=base_ckpt_path,
            )
    else:
        print(f"\n{'='*60}")
        print("Training osteophyte grader: multitask")
        print(f"{'='*60}")
        _train_multitask(
            cfg,
            labels_csv=Path(labels_csv),
            roi_dir=roi_dir,
            train_transform=train_transform,
            val_transform=val_transform,
            allowed_sources=allowed_sources,
            min_train_confidence=min_train_confidence,
            min_eval_confidence=min_eval_confidence,
            confidence_weights=confidence_weights,
            init_checkpoint=warm_start_path,
            resume_checkpoint=resume_multitask_path,
        )


if __name__ == "__main__":
    main()
