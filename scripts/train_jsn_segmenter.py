"""Train U-Net++ for joint space segmentation."""

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader

from src.data.jsn_seg_dataset import JSNSegDataset
from src.data.transforms import get_segmentation_transforms
from src.modules.jsn_module import JSNModule
from src.utils.device import get_accelerator, clear_memory
from src.utils.lightning import build_loggers
from src.utils.seed import seed_everything


def _build_train_dataset(cfg: DictConfig, data_root: Path, train_mask_dir: Path, train_transform):
    train_ds = JSNSegDataset(str(data_root / "train"), str(train_mask_dir), train_transform)
    self_training_cfg = getattr(cfg.training, "jsn_self_training", {})
    if not bool(getattr(self_training_cfg, "enabled", False)):
        return train_ds

    pseudo_mask_dir = Path(str(getattr(self_training_cfg, "pseudo_mask_dir", "")))
    if not pseudo_mask_dir.exists():
        raise FileNotFoundError(f"JSN self-training pseudo-mask directory not found: {pseudo_mask_dir}")

    pseudo_ds = JSNSegDataset(str(data_root / "train"), str(pseudo_mask_dir), train_transform)
    if len(pseudo_ds) == 0:
        raise ValueError(f"No JSN pseudo-mask samples matched under {pseudo_mask_dir}")

    print(f"JSN self-training enabled: {len(train_ds)} manual train masks + {len(pseudo_ds)} pseudo masks")
    return ConcatDataset([train_ds, pseudo_ds])


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    # Datasets
    train_transform = get_segmentation_transforms(cfg, is_train=True)
    val_transform = get_segmentation_transforms(cfg, is_train=False)

    mask_dir = Path(cfg.annotation_dir) / "jsn_masks"
    data_root = Path(cfg.data.root)

    train_mask_dir = mask_dir / "train"
    val_mask_dir = mask_dir / "val"
    if not train_mask_dir.exists() or not val_mask_dir.exists():
        raise FileNotFoundError(
            f"Reviewed JSN masks are required under {mask_dir}/train and {mask_dir}/val. "
            "Run scripts/import_reviewed_annotations.py first."
        )

    train_ds = _build_train_dataset(cfg, data_root, train_mask_dir, train_transform)
    val_ds = JSNSegDataset(str(data_root / "val"), str(val_mask_dir), val_transform)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "JSN segmentation dataset is empty after matching images to masks. "
            "Check reviewed mask import layout and the dataset root."
        )

    checkpoint_subdir = str(getattr(cfg, "jsn_checkpoint_subdir", "jsn_segmenter"))
    logger_name = str(getattr(cfg, "jsn_logger_name", checkpoint_subdir))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        pin_memory=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        pin_memory=cfg.data.num_workers > 0,
    )

    # Model
    module = JSNModule(cfg)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=get_accelerator(),
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 2),
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=str(cfg.training.early_stopping.monitor),
                patience=cfg.training.early_stopping.patience,
                mode=str(cfg.training.early_stopping.mode),
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.checkpoint_dir) / checkpoint_subdir,
                filename="jsn-{epoch:03d}-{val_dice:.4f}",
                monitor="val_dice",
                mode="max",
                save_top_k=3,
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.checkpoint_dir) / checkpoint_subdir,
                filename="jsn-mjsw-{epoch:03d}-{val_mjsw_mae:.4f}",
                monitor="val_mjsw_mae",
                mode="min",
                save_top_k=3,
            ),
        ],
        default_root_dir=cfg.output_dir,
        logger=build_loggers(cfg, logger_name),
    )

    trainer.fit(module, train_loader, val_loader)
    clear_memory()


if __name__ == "__main__":
    main()
