"""Base PyTorch Lightning module with shared training infrastructure."""

from typing import Any, Dict

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.utils.device import clear_memory


class BaseModule(pl.LightningModule):
    """Base module providing shared optimizer, scheduler, and logging."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

    def configure_optimizers(self):
        train_cfg = self.cfg.training
        optimizer_name = train_cfg.get("optimizer", "adam")
        param_groups = self._parameter_groups()

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=train_cfg.learning_rate,
                weight_decay=train_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler_name = train_cfg.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.scheduler_params.T_max,
                eta_min=train_cfg.scheduler_params.eta_min,
            )
        elif scheduler_name == "reduce_on_plateau":
            scheduler_params = dict(train_cfg.scheduler_params)
            scheduler_params.pop("T_max", None)
            scheduler_params.pop("eta_min", None)
            scheduler_params.pop("verbose", None)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_params,
            )
            monitor = train_cfg.get("scheduler_monitor", None)
            if monitor in (None, "", "null"):
                monitor = train_cfg.get("early_stopping", {}).get("monitor", "val/loss")
            if monitor in (None, "", "null"):
                monitor = "val/loss"
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _parameter_groups(self):
        """Optional layer-wise LR for models with a backbone/head split."""
        train_cfg = self.cfg.training
        layer_cfg = train_cfg.get("layer_wise_lr", {})
        if not bool(layer_cfg.get("enabled", False)):
            return self.parameters()

        model = getattr(self, "model", None)
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return self.parameters()

        backbone_ids = {id(param) for param in backbone.parameters()}
        backbone_params = []
        head_params = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if id(param) in backbone_ids:
                backbone_params.append(param)
            else:
                head_params.append(param)

        if not backbone_params or not head_params:
            return self.parameters()

        backbone_ratio = float(layer_cfg.get("backbone_ratio", 0.1))
        return [
            {"params": backbone_params, "lr": float(train_cfg.learning_rate) * backbone_ratio},
            {"params": head_params, "lr": float(train_cfg.learning_rate)},
        ]

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics with optional prefix."""
        for name, value in metrics.items():
            key = f"{prefix}/{name}" if prefix else name
            self.log(key, value, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """Clear MPS/CUDA cache after validation to manage memory on M4."""
        clear_memory()
