"""PyTorch Lightning module for hybrid ConvNeXt + features KL classifier."""

import torch
import numpy as np
from omegaconf import DictConfig

from src.modules.base_module import BaseModule
from src.models.kl_hybrid import HybridKLClassifier
from src.losses.corn_loss import CORNLoss
from src.losses.soft_label import SoftLabelCELoss
from src.utils.metrics import quadratic_weighted_kappa


class KLHybridModule(BaseModule):
    """Lightning module for ConvNeXt + feature fusion KL classification.

    Supports two loss functions via ``cfg.model.loss_type``:
    - ``"soft_label"`` (default): Gaussian-smoothed soft-label CE.
    - ``"corn"``: Conditional Ordinal Regression (K-1 binary thresholds).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.loss_type = str(cfg.model.get("loss_type", "soft_label")).lower()
        num_classes = cfg.model.num_classes

        if self.loss_type == "corn":
            # CORN uses K-1 output logits
            corn_cfg = cfg.model.copy()
            corn_cfg.num_classes = num_classes - 1
            self.model = HybridKLClassifier(corn_cfg)
            self.criterion = CORNLoss(num_classes=num_classes)
            self._num_classes = num_classes
        else:
            self.model = HybridKLClassifier(cfg.model)
            self.criterion = SoftLabelCELoss(num_classes=num_classes, sigma=0.5)
            self._num_classes = num_classes

        self.freeze_epochs = cfg.model.get("freeze_epochs", 50)
        self.val_preds = []
        self.val_targets = []

    def forward(self, image, features):
        return self.model(image, features)

    def _predict_from_logits(self, logits: torch.Tensor) -> np.ndarray:
        """Convert logits to class predictions, handling both loss types."""
        if self.loss_type == "corn":
            # CORN: cumulative sigmoid thresholds -> predicted class
            probs = torch.sigmoid(logits)
            # P(Y > k) for each threshold; predicted class = number of thresholds exceeded
            return probs.round().sum(dim=1).long().clamp(0, self._num_classes - 1).cpu().numpy()
        return logits.argmax(dim=1).cpu().numpy()

    def on_fit_start(self):
        """Freeze backbone at the start of training."""
        self.model.freeze_backbone()

    def on_train_epoch_start(self):
        """Unfreeze backbone after freeze_epochs."""
        if self.current_epoch == self.freeze_epochs:
            self.model.unfreeze_backbone()
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group["lr"] = self.cfg.model.get("unfreeze_lr", 1e-5)

    def training_step(self, batch, batch_idx):
        images, features, labels = batch
        logits = self(images, features)
        loss = self.criterion(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, features, labels = batch
        logits = self(images, features)
        loss = self.criterion(logits, labels)

        preds = self._predict_from_logits(logits)
        targets = labels.cpu().numpy()
        self.val_preds.extend(preds.tolist())
        self.val_targets.extend(targets.tolist())

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.val_preds:
            preds = np.array(self.val_preds)
            targets = np.array(self.val_targets)
            qwk = quadratic_weighted_kappa(targets, preds)
            acc = (preds == targets).mean()
            self.log("val/qwk", qwk, prog_bar=True)
            self.log("val/accuracy", acc, prog_bar=True)
            self.log("val_qwk", qwk, prog_bar=False)
            self.log("val_accuracy", acc, prog_bar=False)

        self.val_preds.clear()
        self.val_targets.clear()
        super().on_validation_epoch_end()
