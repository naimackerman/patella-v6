"""PyTorch Lightning module for sclerosis classification training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score

from src.losses.ordinal_ce import OrdinalCrossEntropyLoss
from src.modules.base_module import BaseModule
from src.models.sclerosis_classifier import SclerosisClassifier
from src.utils.metrics import per_class_metrics
from src.utils.sclerosis_labels import sclerosis_class_names


class SclerosisModule(BaseModule):
    """Lightning module for hybrid sclerosis classification."""

    def __init__(
        self,
        cfg: DictConfig,
        class_weights: torch.Tensor | None = None,
        side_class_weights: dict[int, torch.Tensor] | None = None,
    ):
        super().__init__(cfg)
        self.model = SclerosisClassifier(cfg.model)
        self._register_optional_buffer("class_weights", class_weights)
        side_class_weights = side_class_weights or {}
        self._register_optional_buffer("medial_class_weights", side_class_weights.get(0))
        self._register_optional_buffer("lateral_class_weights", side_class_weights.get(1))
        self.use_ordinal_loss = bool(getattr(cfg.training, "sclerosis_use_ordinal_loss", True))
        self.ordinal_weight = float(getattr(cfg.training, "sclerosis_ordinal_weight", 0.25))
        if self.use_ordinal_loss:
            self.criterion = OrdinalCrossEntropyLoss(
                num_classes=cfg.model.num_classes,
                ordinal_weight=self.ordinal_weight,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.val_preds = []
        self.val_targets = []
        self.val_probs = []
        self.val_side_ids = []
        self.class_names = list(getattr(cfg.model, "class_names", sclerosis_class_names("severity")))
        self.num_classes = int(cfg.model.num_classes)

    def forward(self, roi_image, texture_features, side_ids=None):
        return self.model(roi_image, texture_features, side_ids)

    def _register_optional_buffer(self, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is not None:
            self.register_buffer(name, tensor.float())
        else:
            setattr(self, name, None)

    @staticmethod
    def _weighted_mean(losses: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        if sample_weights is None:
            return losses.mean()
        weights = sample_weights.to(losses.device).float()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def _get_side_class_weights(self, side_value: int) -> torch.Tensor | None:
        if side_value == 0:
            return self.medial_class_weights
        return self.lateral_class_weights

    def _compute_per_sample_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        side_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if side_ids is None or (self.medial_class_weights is None and self.lateral_class_weights is None):
            if self.use_ordinal_loss:
                return self.criterion(logits, labels, reduction="none", class_weight=self.class_weights)
            return F.cross_entropy(logits, labels, reduction="none", weight=self.class_weights)

        side_ids = side_ids.long()
        losses = torch.zeros(labels.shape[0], dtype=logits.dtype, device=logits.device)
        for side_value in (0, 1):
            mask = side_ids == side_value
            if not mask.any():
                continue
            side_weights = self._get_side_class_weights(side_value)
            if self.use_ordinal_loss:
                losses[mask] = self.criterion(
                    logits[mask],
                    labels[mask],
                    reduction="none",
                    class_weight=side_weights,
                )
            else:
                losses[mask] = F.cross_entropy(
                    logits[mask],
                    labels[mask],
                    reduction="none",
                    weight=side_weights,
                )
        return losses

    def training_step(self, batch, batch_idx):
        images, texture_feats, side_ids, labels, sample_weights = batch
        logits = self(images, texture_feats, side_ids)
        per_sample_loss = self._compute_per_sample_loss(logits, labels, side_ids)
        loss = self._weighted_mean(per_sample_loss, sample_weights)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, texture_feats, side_ids, labels, sample_weights = batch
        logits = self(images, texture_feats, side_ids)
        loss = self._weighted_mean(self._compute_per_sample_loss(logits, labels, side_ids), sample_weights)

        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        targets = labels.cpu().numpy()
        self.val_preds.extend(preds.tolist())
        self.val_targets.extend(targets.tolist())
        self.val_probs.extend(probs.tolist())
        self.val_side_ids.extend(side_ids.cpu().numpy().tolist())

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.val_preds:
            preds = np.array(self.val_preds)
            targets = np.array(self.val_targets)
            acc = (preds == targets).mean()
            self.log("val/accuracy", acc, prog_bar=True)
            self.log("val_accuracy", acc, prog_bar=False)

            metrics = per_class_metrics(targets, preds, class_names=self.class_names)
            for name, value in metrics.items():
                if name == "accuracy":
                    continue
                self.log(f"val/{name}", value)
                if name == "f1_macro":
                    self.log("val_f1_macro", value, prog_bar=False)

            probs = np.asarray(self.val_probs, dtype=np.float64)
            if len(np.unique(targets)) > 1:
                try:
                    auc = self._roc_auc(targets, probs)
                    self.log("val/auc_macro", float(auc), prog_bar=False)
                    self.log("val_auc_macro", float(auc), prog_bar=False)
                except ValueError:
                    pass

            side_ids = np.asarray(self.val_side_ids, dtype=np.int64)
            for side_value, side_name in ((0, "medial"), (1, "lateral")):
                mask = side_ids == side_value
                if not mask.any():
                    continue
                side_metrics = per_class_metrics(
                    targets[mask],
                    preds[mask],
                    class_names=self.class_names,
                )
                self.log(f"val/f1_macro_{side_name}", side_metrics["f1_macro"], prog_bar=False)
                self.log(f"val/accuracy_{side_name}", side_metrics["accuracy"], prog_bar=False)
                if len(np.unique(targets[mask])) > 1:
                    try:
                        side_auc = self._roc_auc(targets[mask], probs[mask])
                        self.log(f"val/auc_macro_{side_name}", float(side_auc), prog_bar=False)
                    except ValueError:
                        pass

        self.val_preds.clear()
        self.val_targets.clear()
        self.val_probs.clear()
        self.val_side_ids.clear()
        super().on_validation_epoch_end()

    def _roc_auc(self, targets: np.ndarray, probs: np.ndarray) -> float:
        if self.num_classes == 2:
            return float(roc_auc_score(targets, probs[:, 1]))
        return float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
