"""PyTorch Lightning module for osteophyte grading training."""

import torch
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score

from src.modules.base_module import BaseModule
from src.models.osteophyte_grader import OsteophyteGrader
from src.losses.focal_ordinal_ce import FocalOrdinalCrossEntropyLoss
from src.losses.ordinal_ce import OrdinalCrossEntropyLoss


class OsteophyteModule(BaseModule):
    """Lightning module for SE-ResNet-50 osteophyte grading."""

    def __init__(
        self,
        cfg: DictConfig,
        site: str | None = None,
        class_weights_by_site: torch.Tensor | None = None,
    ):
        super().__init__(cfg)
        self.model = OsteophyteGrader(cfg.model)
        loss_type = str(getattr(cfg.training, "osteophyte_loss", "ordinal_ce")).lower()
        ordinal_weight = float(
            getattr(
                cfg.training,
                "osteophyte_ordinal_weight",
                getattr(cfg.training, "ordinal_weight", 0.25),
            )
        )
        if loss_type in {"focal_ordinal", "ordinal_focal"}:
            self.criterion = FocalOrdinalCrossEntropyLoss(
                num_classes=cfg.model.num_classes_per_head,
                alpha=float(getattr(cfg.training, "focal_alpha", 0.25)),
                gamma=float(getattr(cfg.training, "focal_gamma", 2.0)),
                ordinal_weight=ordinal_weight,
            )
        elif loss_type in {"ordinal_ce", "ordinal"}:
            self.criterion = OrdinalCrossEntropyLoss(
                num_classes=cfg.model.num_classes_per_head,
                ordinal_weight=ordinal_weight,
            )
        else:
            raise ValueError(f"Unsupported osteophyte_loss: {loss_type}")
        self.site = site
        if class_weights_by_site is not None:
            self.register_buffer("class_weights_by_site", class_weights_by_site.float())
        else:
            self.class_weights_by_site = None
        if self.site is None:
            self.val_preds = {site_name: [] for site_name in OsteophyteGrader.SITES}
            self.val_targets = {site_name: [] for site_name in OsteophyteGrader.SITES}
            self.val_probs = {site_name: [] for site_name in OsteophyteGrader.SITES}
        else:
            self.val_preds = []
            self.val_targets = []
            self.val_probs = []

    def forward(self, x, site):
        return self.model.forward_single(x, site)

    @staticmethod
    def _weighted_mean(losses: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
        if sample_weights is None:
            return losses.mean()
        weights = sample_weights.to(losses.device).float()
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def _compute_multitask_loss(
        self,
        logits_by_site: dict[str, torch.Tensor],
        labels: torch.Tensor,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        site_losses = []
        for site_idx, site_name in enumerate(OsteophyteGrader.SITES):
            logits = logits_by_site[site_name]
            per_sample_loss = self.criterion(
                logits,
                labels[:, site_idx],
                reduction="none",
                class_weight=self._site_class_weights(site_name),
            )
            current_weights = None if sample_weights is None else sample_weights[:, site_idx]
            site_losses.append(self._weighted_mean(per_sample_loss, current_weights))
        return torch.stack(site_losses).mean()

    def _site_class_weights(self, site_name: str) -> torch.Tensor | None:
        if self.class_weights_by_site is None:
            return None
        if self.site is not None:
            return self.class_weights_by_site
        site_idx = OsteophyteGrader.SITES.index(site_name)
        return self.class_weights_by_site[site_idx]

    def training_step(self, batch, batch_idx):
        images, labels, _, sample_weights = batch
        if self.site is None:
            logits_by_site = self.model(
                images[:, 0],
                images[:, 1],
                images[:, 2],
                images[:, 3],
            )
            loss = self._compute_multitask_loss(logits_by_site, labels, sample_weights)
        else:
            logits = self.model.forward_single(images, self.site)
            per_sample_loss = self.criterion(
                logits,
                labels,
                reduction="none",
                class_weight=self._site_class_weights(self.site),
            )
            loss = self._weighted_mean(per_sample_loss, sample_weights)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, _, _ = batch
        if self.site is None:
            logits_by_site = self.model(
                images[:, 0],
                images[:, 1],
                images[:, 2],
                images[:, 3],
            )
            loss = self._compute_multitask_loss(logits_by_site, labels, sample_weights=None)

            for site_idx, site_name in enumerate(OsteophyteGrader.SITES):
                logits = logits_by_site[site_name]
                preds = logits.argmax(dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                targets = labels[:, site_idx].cpu().numpy()
                self.val_preds[site_name].extend(preds.tolist())
                self.val_targets[site_name].extend(targets.tolist())
                self.val_probs[site_name].extend(probs.tolist())
        else:
            logits = self.model.forward_single(images, self.site)
            loss = self.criterion(logits, labels, class_weight=self._site_class_weights(self.site))

            preds = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            targets = labels.cpu().numpy()
            self.val_preds.extend(preds.tolist())
            self.val_targets.extend(targets.tolist())
            self.val_probs.extend(probs.tolist())

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.site is None:
            kappas = []
            balanced_accs = []
            aucs = []
            for site_name in OsteophyteGrader.SITES:
                if not self.val_preds[site_name]:
                    continue
                preds = np.asarray(self.val_preds[site_name], dtype=np.int64)
                targets = np.asarray(self.val_targets[site_name], dtype=np.int64)
                probs = np.asarray(self.val_probs[site_name], dtype=np.float64)

                kappa = cohen_kappa_score(targets, preds, weights="quadratic")
                balanced_acc = balanced_accuracy_score(targets, preds)
                kappas.append(kappa)
                balanced_accs.append(balanced_acc)
                self.log(f"val/kappa_{site_name}", kappa, prog_bar=False)
                self.log(f"val/balanced_accuracy_{site_name}", balanced_acc, prog_bar=False)

                unique_targets = np.unique(targets)
                if len(unique_targets) > 1:
                    try:
                        auc = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
                        aucs.append(auc)
                        self.log(f"val/auc_macro_{site_name}", auc, prog_bar=False)
                    except ValueError:
                        pass

            if kappas:
                mean_kappa = float(np.mean(kappas))
                self.log("val/kappa_mean", mean_kappa, prog_bar=True)
                self.log("val_kappa_mean", mean_kappa, prog_bar=False)
                self.log("val_kappa", mean_kappa, prog_bar=False)
            if balanced_accs:
                self.log("val/balanced_accuracy_mean", float(np.mean(balanced_accs)), prog_bar=False)
            if aucs:
                self.log("val/auc_macro_mean", float(np.mean(aucs)), prog_bar=False)

            self.val_preds = {site_name: [] for site_name in OsteophyteGrader.SITES}
            self.val_targets = {site_name: [] for site_name in OsteophyteGrader.SITES}
            self.val_probs = {site_name: [] for site_name in OsteophyteGrader.SITES}
        else:
            if self.val_preds:
                kappa = cohen_kappa_score(self.val_targets, self.val_preds, weights="quadratic")
                self.log("val/kappa", kappa, prog_bar=True)
                self.log("val_kappa", kappa, prog_bar=False)
                self.log(f"val/kappa_{self.site}", kappa, prog_bar=False)
                balanced_acc = balanced_accuracy_score(self.val_targets, self.val_preds)
                self.log("val/balanced_accuracy", balanced_acc, prog_bar=False)
                self.log(f"val/balanced_accuracy_{self.site}", balanced_acc, prog_bar=False)

                probs = np.asarray(self.val_probs, dtype=np.float64)
                targets = np.asarray(self.val_targets, dtype=np.int64)
                unique_targets = np.unique(targets)
                if len(unique_targets) > 1:
                    try:
                        auc = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
                        self.log("val/auc_macro", float(auc), prog_bar=False)
                        self.log(f"val/auc_macro_{self.site}", float(auc), prog_bar=False)
                    except ValueError:
                        pass
            self.val_preds.clear()
            self.val_targets.clear()
            self.val_probs.clear()
        super().on_validation_epoch_end()
