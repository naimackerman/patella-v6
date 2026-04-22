"""Train ConvNeXt-Small + feature fusion hybrid classifier (Path B)."""

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import torch

from src.modules.kl_hybrid_module import KLHybridModule
from src.utils.device import get_accelerator, clear_memory
from src.utils.lightning import build_loggers
from src.utils.seed import seed_everything
from src.data.kl_dataset import KLHybridDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.data.sampler import create_weighted_sampler


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    # Image + feature datasets
    feature_dir = Path(cfg.feature_dir) / "aggregated"
    train_transform = get_train_transforms(cfg)
    val_transform = get_eval_transforms(cfg)

    train_ds = KLHybridDataset(
        cfg.data.root, "train", str(feature_dir / "train_features.npz"), train_transform,
    )
    val_ds = KLHybridDataset(
        cfg.data.root, "val", str(feature_dir / "val_features.npz"), val_transform,
    )
    if int(cfg.model.feature_dim) != int(train_ds.feat_dim):
        raise ValueError(
            f"Hybrid KL feature_dim mismatch: config expects {cfg.model.feature_dim}, "
            f"but aggregated features have {train_ds.feat_dim} columns. "
            "Regenerate features or update configs/model/convnext_hybrid.yaml."
        )

    sampler = create_weighted_sampler(train_ds.labels)

    train_loader = DataLoader(
        train_ds, batch_size=4, sampler=sampler,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Model
    module = KLHybridModule(cfg)
    resume_checkpoint = getattr(cfg, "checkpoint_path", None)
    if resume_checkpoint in (None, "", "null", "None"):
        resume_checkpoint = None
    else:
        resume_checkpoint = Path(str(resume_checkpoint))
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"KL hybrid resume checkpoint not found: {resume_checkpoint}")
        print(f"Resuming KL hybrid training from checkpoint: {resume_checkpoint}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.model.freeze_epochs + 50,
        accelerator=get_accelerator(),
        precision=cfg.training.precision,
        accumulate_grad_batches=8,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_qwk", patience=15, mode="max",
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=Path(cfg.checkpoint_dir) / "kl_hybrid",
                filename="hybrid-{epoch:03d}-{val_qwk:.4f}",
                monitor="val_qwk",
                mode="max",
                save_top_k=1,
            ),
        ],
        default_root_dir=cfg.output_dir,
        logger=build_loggers(cfg, "kl_hybrid"),
    )

    trainer.fit(
        module,
        train_loader,
        val_loader,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
    )
    clear_memory()


if __name__ == "__main__":
    main()
