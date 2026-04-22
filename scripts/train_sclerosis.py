"""Train hybrid sclerosis classifier on subchondral ROI patches + texture features."""

from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.modules.sclerosis_module import SclerosisModule
from src.data.sclerosis_dataset import SclerosisDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.sampler import create_weighted_sampler
from src.utils.annotation_paths import EXPANDED_SOURCES, MANUAL_SOURCES
from src.utils.device import get_accelerator, clear_memory
from src.utils.lightning import build_loggers
from src.utils.seed import seed_everything
from src.utils.annotation_confidence import confidence_at_least, confidence_weight
from src.utils.feature_scaling import fit_standardizer, save_standardizer, transform_with_standardizer
from src.utils.sclerosis_labels import (
    apply_sclerosis_label_scheme_to_cfg,
    map_sclerosis_grades,
    sclerosis_class_names,
)

SEPARATE_STRATEGIES = {"separate", "separate_by_side", "per_side", "side_specific"}
SHARED_STRATEGIES = {"shared", "shared_single_head", "shared_classifier", "single_head"}


class FreezeBackboneCallback(pl.Callback):
    """Temporarily freeze the CNN backbone during early epochs."""

    def __init__(self, freeze_epochs: int):
        super().__init__()
        self.freeze_epochs = max(0, int(freeze_epochs))
        self._frozen = False
        self._unfroze = False

    @staticmethod
    def _set_backbone_grad(pl_module: pl.LightningModule, requires_grad: bool) -> None:
        backbone = getattr(getattr(pl_module, "model", None), "cnn", None)
        if backbone is None:
            return
        for param in backbone.parameters():
            param.requires_grad = requires_grad

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.freeze_epochs <= 0:
            return
        self._set_backbone_grad(pl_module, False)
        self._frozen = True
        print(f"Sclerosis backbone frozen for first {self.freeze_epochs} epoch(s).")

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._frozen or self._unfroze:
            return
        if trainer.current_epoch >= self.freeze_epochs:
            self._set_backbone_grad(pl_module, True)
            self._unfroze = True
            print(f"Sclerosis backbone unfrozen at epoch {trainer.current_epoch}.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    label_scheme = str(getattr(cfg.training, "sclerosis_label_scheme", "severity"))
    cfg = apply_sclerosis_label_scheme_to_cfg(cfg, label_scheme)
    class_names = sclerosis_class_names(label_scheme)

    feature_dir = Path(cfg.feature_dir)
    scl_dir = Path(str(getattr(cfg, "sclerosis_output_dir", feature_dir / "sclerosis")))
    strategy = str(getattr(cfg.training, "sclerosis_strategy", "multitask_heads")).lower()
    separate_sides = strategy in SEPARATE_STRATEGIES
    shared_single_head = strategy in SHARED_STRATEGIES
    if separate_sides and bool(getattr(cfg.model, "use_side_specific_heads", False)):
        cfg = OmegaConf.merge(cfg, {"model": {"use_side_specific_heads": False}})
        print("Sclerosis separate-side training: using a single-head model per side (use_side_specific_heads=false).")
    elif shared_single_head and bool(getattr(cfg.model, "use_side_specific_heads", False)):
        cfg = OmegaConf.merge(cfg, {"model": {"use_side_specific_heads": False}})
        print("Sclerosis shared training: using a shared classifier head (use_side_specific_heads=false).")
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

    # Load precomputed ROI paths, texture features, and labels
    train_data = _apply_label_scheme_to_data(np.load(scl_dir / "train_sclerosis_data.npz", allow_pickle=True), label_scheme)
    val_data = _apply_label_scheme_to_data(np.load(scl_dir / "val_sclerosis_data.npz", allow_pickle=True), label_scheme)
    dev_pool_cfg = getattr(cfg.training, "sclerosis_dev_pool", {})
    use_dev_pool = bool(getattr(dev_pool_cfg, "enabled", False))

    if use_dev_pool:
        dev_data = _concat_split_data(train_data, val_data)
        dev_keep = _select_indices_by_label_mode(dev_data, label_mode, allow_bootstrap_fallback)
        dev_keep, dev_sample_weights = _apply_confidence_policy(
            dev_data, dev_keep, min_confidence=min_train_confidence, confidence_weights=confidence_weights
        )
        dev_keep, dev_sample_weights = _apply_roi_source_filter(
            dev_data,
            dev_keep,
            dev_sample_weights,
            allowed_sources=getattr(getattr(cfg.training, "sclerosis_roi_source_filter", {}), "train", []),
        )
        dev_sample_weights = _apply_label_source_weights(
            dev_data,
            dev_keep,
            dev_sample_weights,
            source_weights=getattr(cfg.training, "sclerosis_source_weights", {}),
        )
        train_keep, val_keep = _split_dev_pool_indices(dev_data, dev_keep, cfg.seed, dev_pool_cfg)
        train_weight_lookup = {int(idx): float(weight) for idx, weight in zip(dev_keep.tolist(), dev_sample_weights.tolist())}
        train_sample_weights = np.asarray([train_weight_lookup[int(idx)] for idx in train_keep.tolist()], dtype=np.float32)
        val_sample_weights = np.asarray([train_weight_lookup[int(idx)] for idx in val_keep.tolist()], dtype=np.float32)
        train_data = dev_data
        val_data = dev_data
    else:
        train_keep = _select_indices_by_label_mode(train_data, label_mode, allow_bootstrap_fallback)
        val_keep = _select_indices_by_label_mode(val_data, label_mode, allow_bootstrap_fallback)
        train_keep, train_sample_weights = _apply_confidence_policy(
            train_data, train_keep, min_confidence=min_train_confidence, confidence_weights=confidence_weights
        )
        val_keep, val_sample_weights = _apply_confidence_policy(
            val_data, val_keep, min_confidence=min_eval_confidence, confidence_weights=confidence_weights
        )
        train_keep, train_sample_weights = _apply_roi_source_filter(
            train_data,
            train_keep,
            train_sample_weights,
            allowed_sources=getattr(getattr(cfg.training, "sclerosis_roi_source_filter", {}), "train", []),
        )
        train_sample_weights = _apply_label_source_weights(
            train_data,
            train_keep,
            train_sample_weights,
            source_weights=getattr(cfg.training, "sclerosis_source_weights", {}),
        )
        val_keep, val_sample_weights = _apply_roi_source_filter(
            val_data,
            val_keep,
            val_sample_weights,
            allowed_sources=getattr(getattr(cfg.training, "sclerosis_roi_source_filter", {}), "eval", []),
        )

    if len(train_keep) == 0 or len(val_keep) == 0:
        raise ValueError(f"No sclerosis samples available for label_mode={label_mode}")

    print(f"Sclerosis label mode: {label_mode}")
    print(f"Sclerosis label scheme: {cfg.training.sclerosis_label_scheme} ({class_names})")
    if use_dev_pool:
        print(
            f"Sclerosis dev pool enabled: combined reviewed development pool with "
            f"internal fold holdout ({int(getattr(dev_pool_cfg, 'n_splits', 5))}-fold, holdout_fold={int(getattr(dev_pool_cfg, 'holdout_fold', 0))})."
        )
    print(f"Train samples: {len(train_keep)} / {len(train_data['grades'])}")
    print(f"Val samples:   {len(val_keep)} / {len(val_data['grades'])}")
    print(f"Sclerosis confidence policy: train>={min_train_confidence}, eval>={min_eval_confidence}, weights={confidence_weights}")
    print(f"Sclerosis label-source weights: {dict(getattr(cfg.training, 'sclerosis_source_weights', {}))}")
    if _data_has_key(train_data, "label_sources"):
        manual_present = pd.Series(train_data["label_sources"].astype(str)).isin(MANUAL_SOURCES).any()
        if not manual_present and label_mode in {"auto", "manual"} and allow_bootstrap_fallback:
            print("Warning: no reviewed/manual sclerosis labels found; this run remains a bootstrap baseline.")
    if _data_has_key(train_data, "roi_sources"):
        train_source_counts = Counter(np.asarray(train_data["roi_sources"])[train_keep].astype(str).tolist())
        val_source_counts = Counter(np.asarray(val_data["roi_sources"])[val_keep].astype(str).tolist())
        print(f"Sclerosis ROI sources (train subset): {dict(train_source_counts)}")
        print(f"Sclerosis ROI sources (val subset): {dict(val_source_counts)}")

    train_transform = get_train_transforms(cfg)
    val_transform = get_eval_transforms(cfg)
    target_roi_size = int(getattr(cfg.preprocessing.sclerosis_roi, "output_size", 96))

    primary_monitor = str(getattr(cfg.training, "sclerosis_primary_monitor", "val_f1_macro"))
    primary_mode = str(getattr(cfg.training, "sclerosis_primary_mode", "max"))
    secondary_monitor = str(getattr(cfg.training, "sclerosis_secondary_monitor", "val_auc_macro"))
    secondary_mode = str(getattr(cfg.training, "sclerosis_secondary_mode", "max"))
    sampling_cfg = getattr(cfg.training, "sclerosis_sampling", {})
    sampler_multiplier_power = float(getattr(sampling_cfg, "multiplier_power", 1.0))
    sampler_max_weight_ratio = getattr(sampling_cfg, "max_weight_ratio_to_median", None)
    backbone_freeze_epochs = int(getattr(cfg.training, "sclerosis_backbone_freeze_epochs", 0))
    accumulate_grad_batches = int(getattr(cfg.training, "accumulate_grad_batches", 1))
    log_every_n_steps = int(getattr(cfg.training, "log_every_n_steps", 10))
    scheduler_monitor = getattr(cfg.training, "scheduler_monitor", None)
    if scheduler_monitor in (None, "", "null"):
        cfg = OmegaConf.merge(cfg, {"training": {"scheduler_monitor": primary_monitor}})
    train_side_ids_all = _extract_side_ids(train_data)
    val_side_ids_all = _extract_side_ids(val_data)
    early_stopping_patience = int(getattr(getattr(cfg.training, "early_stopping", {}), "patience", 20))

    run_defs = [("shared", None)] if not separate_sides else [("medial", 0), ("lateral", 1)]
    for run_name, side_id in run_defs:
        _finish_wandb_run()
        if side_id is None:
            run_train_keep = train_keep
            run_val_keep = val_keep
            run_train_weights = train_sample_weights
            run_val_weights = val_sample_weights
        else:
            train_mask = train_side_ids_all[train_keep] == side_id
            val_mask = val_side_ids_all[val_keep] == side_id
            run_train_keep = train_keep[train_mask]
            run_val_keep = val_keep[val_mask]
            run_train_weights = train_sample_weights[train_mask]
            run_val_weights = val_sample_weights[val_mask]

        if len(run_train_keep) == 0 or len(run_val_keep) == 0:
            print(f"Skipping sclerosis {run_name}: no samples after filtering.")
            continue

        scaler_suffix = "" if side_id is None else f"_{run_name}"
        scaler_path = scl_dir / f"texture_standardizer{scaler_suffix}.npz"
        mean, scale = fit_standardizer(train_data["texture_features"][run_train_keep])
        save_standardizer(scaler_path, mean, scale)
        train_texture = transform_with_standardizer(train_data["texture_features"][run_train_keep], mean, scale)
        val_texture = transform_with_standardizer(val_data["texture_features"][run_val_keep], mean, scale)
        train_side_ids = train_side_ids_all[run_train_keep]
        val_side_ids = val_side_ids_all[run_val_keep]

        train_ds = SclerosisDataset(
            roi_paths=train_data["roi_paths"][run_train_keep].tolist(),
            texture_features=train_texture,
            side_ids=train_side_ids,
            grades=train_data["grades"][run_train_keep],
            sample_weights=run_train_weights,
            transform=train_transform,
            target_size=target_roi_size,
        )
        val_ds = SclerosisDataset(
            roi_paths=val_data["roi_paths"][run_val_keep].tolist(),
            texture_features=val_texture,
            side_ids=val_side_ids,
            grades=val_data["grades"][run_val_keep],
            sample_weights=run_val_weights,
            transform=val_transform,
            target_size=target_roi_size,
        )

        class_counts = np.bincount(train_ds.grades.numpy(), minlength=cfg.model.num_classes).astype(np.float64)
        class_weights = len(train_ds) / np.maximum(class_counts, 1.0)
        class_weights = class_weights / class_weights.mean()
        side_class_weights = None
        if side_id is None and strategy in {"multitask_heads", "shared_multitask", "shared_multitask_heads"}:
            side_class_weights = {}
            for side_value, side_name in ((0, "medial"), (1, "lateral")):
                mask = train_side_ids == side_value
                if not mask.any():
                    continue
                side_counts = np.bincount(train_ds.grades.numpy()[mask], minlength=cfg.model.num_classes).astype(np.float64)
                weights = mask.sum() / np.maximum(side_counts, 1.0)
                weights = weights / weights.mean()
                side_class_weights[side_value] = torch.tensor(weights, dtype=torch.float32)
                print(f"Sclerosis {side_name} class counts: {side_counts.astype(int).tolist()}")
                print(f"Sclerosis {side_name} class weights: {[round(float(x), 4) for x in weights.tolist()]}")
        print(f"Sclerosis run: {run_name}")
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples:   {len(val_ds)}")
        print(f"Sclerosis class counts: {class_counts.astype(int).tolist()}")
        print(f"Sclerosis class weights: {[round(float(x), 4) for x in class_weights.tolist()]}")

        sampler = create_weighted_sampler(
            train_ds.grades.numpy(),
            sample_weight_multipliers=run_train_weights,
            multiplier_power=sampler_multiplier_power,
            max_weight_ratio_to_median=sampler_max_weight_ratio,
        )

        train_loader = DataLoader(
            train_ds, batch_size=16, sampler=sampler,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.num_workers > 0,
            pin_memory=cfg.data.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=16, shuffle=False,
            num_workers=cfg.data.num_workers,
            persistent_workers=cfg.data.num_workers > 0,
            pin_memory=cfg.data.num_workers > 0,
        )

        module = SclerosisModule(
            cfg,
            class_weights=torch.tensor(class_weights, dtype=torch.float32),
            side_class_weights=side_class_weights,
        )
        ckpt_subdir = "sclerosis" if side_id is None else f"sclerosis_{run_name}"
        filename_prefix = "scl" if side_id is None else f"scl-{run_name}"
        logger_name = "sclerosis_hybrid" if side_id is None else f"sclerosis_hybrid_{run_name}"

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=get_accelerator(),
            precision=cfg.training.precision,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=cfg.training.gradient_clip_val,
            log_every_n_steps=log_every_n_steps,
            callbacks=[
                    FreezeBackboneCallback(backbone_freeze_epochs),
                    pl.callbacks.EarlyStopping(
                        monitor=primary_monitor, patience=early_stopping_patience, mode=primary_mode,
                    ),
                    pl.callbacks.ModelCheckpoint(
                        dirpath=Path(cfg.checkpoint_dir) / ckpt_subdir,
                        filename=f"{filename_prefix}-f1-{{epoch:03d}}-{{val_f1_macro:.4f}}",
                        monitor=primary_monitor,
                        mode=primary_mode,
                        save_top_k=3,
                    ),
                    pl.callbacks.ModelCheckpoint(
                        dirpath=Path(cfg.checkpoint_dir) / ckpt_subdir,
                        filename=f"{filename_prefix}-auc-{{epoch:03d}}-{{val_auc_macro:.4f}}",
                        monitor=secondary_monitor,
                        mode=secondary_mode,
                        save_top_k=1,
                    ),
            ],
            default_root_dir=cfg.output_dir,
            logger=build_loggers(cfg, logger_name),
        )

        trainer.fit(module, train_loader, val_loader)
        _finish_wandb_run()
        clear_memory()


def _select_indices_by_label_mode(
    data,
    label_mode: str,
    allow_bootstrap_fallback: bool = False,
) -> np.ndarray:
    if not _data_has_key(data, "label_sources"):
        if allow_bootstrap_fallback or (label_mode or "manual").lower() in {"bootstrap", "all", "pseudo"}:
            return np.arange(len(data["grades"]))
        raise ValueError(
            "Sclerosis dataset lacks label_sources metadata, so manual/expanded mode cannot be enforced."
        )

    sources = pd.Series(data["label_sources"].astype(str))
    label_mode = (label_mode or "manual").lower()
    if label_mode == "manual":
        keep = sources.isin(MANUAL_SOURCES)
        if keep.any():
            return np.flatnonzero(keep.to_numpy())
        if allow_bootstrap_fallback:
            return np.arange(len(sources))
        raise ValueError("Manual sclerosis mode requested, but no reviewed/manual label_source rows were found.")
    if label_mode == "expanded":
        keep = sources.isin(EXPANDED_SOURCES)
        if keep.any():
            return np.flatnonzero(keep.to_numpy())
        keep = sources.isin(MANUAL_SOURCES)
        if keep.any():
            return np.flatnonzero(keep.to_numpy())
        if allow_bootstrap_fallback:
            return np.arange(len(sources))
        raise ValueError("Expanded sclerosis mode requested, but no reviewed/high-confidence rows were found.")
    if label_mode == "auto":
        keep = sources.isin(MANUAL_SOURCES)
        if keep.any():
            return np.flatnonzero(keep.to_numpy())
        if allow_bootstrap_fallback:
            return np.arange(len(sources))
        raise ValueError("Auto sclerosis mode found no reviewed/manual rows and fallback is disabled.")
    return np.arange(len(sources))


def _apply_confidence_policy(
    data,
    keep: np.ndarray,
    min_confidence: str,
    confidence_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
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


def _apply_roi_source_filter(
    data,
    keep: np.ndarray,
    sample_weights: np.ndarray,
    allowed_sources,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = [str(item).strip() for item in list(allowed_sources or []) if str(item).strip()]
    if not allowed or not _data_has_key(data, "roi_sources"):
        return keep, sample_weights

    roi_sources = np.asarray(data["roi_sources"]).astype(str)
    mask = np.isin(roi_sources[keep], allowed)
    return keep[mask], sample_weights[mask]


def _apply_label_source_weights(
    data,
    keep: np.ndarray,
    sample_weights: np.ndarray,
    source_weights,
) -> np.ndarray:
    if not source_weights or not _data_has_key(data, "label_sources"):
        return sample_weights

    sources = np.asarray(data["label_sources"]).astype(str)
    weighted = np.asarray(sample_weights, dtype=np.float32).copy()
    for position, idx in enumerate(keep.tolist()):
        source = sources[idx]
        if source in source_weights:
            weighted[position] *= float(source_weights[source])
    return weighted


def _extract_side_ids(data) -> np.ndarray:
    if _data_has_key(data, "side_ids"):
        return np.asarray(data["side_ids"], dtype=np.int64)
    image_ids = np.asarray(data["image_ids"]).astype(str)
    return np.asarray([0 if iid.endswith("_medial") else 1 for iid in image_ids], dtype=np.int64)


def _data_has_key(data, key: str) -> bool:
    return key in data.keys() if isinstance(data, dict) else key in data.files


def _apply_label_scheme_to_data(data, label_scheme: str) -> dict[str, np.ndarray]:
    keys = data.keys() if isinstance(data, dict) else data.files
    mapped = {key: np.asarray(data[key]) for key in keys}
    mapped["grades"] = map_sclerosis_grades(mapped["grades"], label_scheme)
    return mapped


def _concat_split_data(*datas) -> dict[str, np.ndarray]:
    keys = set()
    for data in datas:
        keys.update(data.keys() if isinstance(data, dict) else data.files)
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        arrays = []
        for data in datas:
            if _data_has_key(data, key):
                arrays.append(np.asarray(data[key]))
        if not arrays:
            continue
        merged[key] = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    return merged


def _extract_group_ids(data) -> np.ndarray:
    image_ids = np.asarray(data["image_ids"]).astype(str)
    groups = []
    for image_id in image_ids:
        parts = image_id.rsplit("_", 1)
        groups.append(parts[0] if len(parts) == 2 and parts[1] in {"medial", "lateral"} else image_id)
    return np.asarray(groups)


def _split_dev_pool_indices(
    data,
    keep: np.ndarray,
    seed: int,
    dev_pool_cfg,
) -> tuple[np.ndarray, np.ndarray]:
    n_splits = max(3, int(getattr(dev_pool_cfg, "n_splits", 5)))
    holdout_fold = int(getattr(dev_pool_cfg, "holdout_fold", 0))
    grades = np.asarray(data["grades"], dtype=np.int64)[keep]
    groups = _extract_group_ids(data)[keep]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    splits = list(splitter.split(np.zeros(len(keep)), grades, groups))
    if not splits:
        raise ValueError("Could not build a dev-pool split for sclerosis training.")
    fold_idx = holdout_fold % len(splits)
    train_rel, val_rel = splits[fold_idx]
    return keep[train_rel], keep[val_rel]


def _finish_wandb_run() -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()



if __name__ == "__main__":
    main()
