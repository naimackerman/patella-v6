"""PyTorch Lightning module for joint space segmentation training."""

import torch
import numpy as np
from omegaconf import DictConfig

from src.modules.base_module import BaseModule
from src.models.jsn_segmenter import create_jsn_segmenter
from src.losses.dice_ce import DiceCELoss
from src.features.jsw_computation import compute_all_jsn_features, get_jsn_measurement_kwargs
from src.utils.metrics import dice_coefficient, hausdorff_95, icc


class JSNModule(BaseModule):
    """Lightning module for U-Net++ joint space segmentation."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = create_jsn_segmenter(cfg.model)
        self.criterion = DiceCELoss(num_classes=cfg.model.classes)
        self.measurement_kwargs = get_jsn_measurement_kwargs(cfg)
        self.val_dice_scores = []
        self.val_dice_medial_scores = []
        self.val_dice_lateral_scores = []
        self.val_hd95_scores = []
        self.val_hd95_medial_scores = []
        self.val_hd95_lateral_scores = []
        self.val_pred_mjsw_medial = []
        self.val_pred_mjsw_lateral = []
        self.val_gt_mjsw_medial = []
        self.val_gt_mjsw_lateral = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks, _ = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, _ = batch
        logits = self(images)
        loss = self.criterion(logits, masks)

        # Compute Dice score
        pred_masks = logits.argmax(dim=1).cpu().numpy()
        gt_masks = masks.cpu().numpy()

        for pred, gt in zip(pred_masks, gt_masks):
            scores = dice_coefficient(pred, gt, num_classes=self.cfg.model.classes)
            self.val_dice_scores.append(scores["dice_mean"])
            self.val_dice_medial_scores.append(float(scores.get("dice_class_1", 0.0)))
            self.val_dice_lateral_scores.append(float(scores.get("dice_class_2", 0.0)))

            hd_scores = []
            for cls in range(1, self.cfg.model.classes):
                pred_bin = pred == cls
                gt_bin = gt == cls
                if pred_bin.any() and gt_bin.any():
                    hd = hausdorff_95(pred_bin, gt_bin)
                    hd_scores.append(hd)
                    if cls == 1:
                        self.val_hd95_medial_scores.append(float(hd))
                    elif cls == 2:
                        self.val_hd95_lateral_scores.append(float(hd))
            if hd_scores:
                self.val_hd95_scores.append(float(np.mean(hd_scores)))

            pred_features = compute_all_jsn_features(pred, **self.measurement_kwargs)
            gt_features = compute_all_jsn_features(gt, **self.measurement_kwargs)
            self.val_pred_mjsw_medial.append(float(pred_features["mJSW_medial"]))
            self.val_pred_mjsw_lateral.append(float(pred_features["mJSW_lateral"]))
            self.val_gt_mjsw_medial.append(float(gt_features["mJSW_medial"]))
            self.val_gt_mjsw_lateral.append(float(gt_features["mJSW_lateral"]))

        self.log("val/loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.val_dice_scores:
            avg_dice = float(np.mean(self.val_dice_scores))
            self.log("val/dice", avg_dice, prog_bar=True)
            self.log("val_dice", avg_dice, prog_bar=False)
            if self.val_dice_medial_scores:
                self.log("val/dice_medial", float(np.mean(self.val_dice_medial_scores)), prog_bar=False)
            if self.val_dice_lateral_scores:
                self.log("val/dice_lateral", float(np.mean(self.val_dice_lateral_scores)), prog_bar=False)

        if self.val_hd95_scores:
            finite_hd = [score for score in self.val_hd95_scores if np.isfinite(score)]
            if finite_hd:
                self.log("val/hausdorff95", float(np.mean(finite_hd)), prog_bar=False)
            if self.val_hd95_medial_scores:
                finite_medial = [score for score in self.val_hd95_medial_scores if np.isfinite(score)]
                if finite_medial:
                    self.log("val/hausdorff95_medial", float(np.mean(finite_medial)), prog_bar=False)
            if self.val_hd95_lateral_scores:
                finite_lateral = [score for score in self.val_hd95_lateral_scores if np.isfinite(score)]
                if finite_lateral:
                    self.log("val/hausdorff95_lateral", float(np.mean(finite_lateral)), prog_bar=False)

        if self.val_pred_mjsw_medial and self.val_gt_mjsw_medial:
            pred_med = np.asarray(self.val_pred_mjsw_medial, dtype=np.float64)
            pred_lat = np.asarray(self.val_pred_mjsw_lateral, dtype=np.float64)
            gt_med = np.asarray(self.val_gt_mjsw_medial, dtype=np.float64)
            gt_lat = np.asarray(self.val_gt_mjsw_lateral, dtype=np.float64)
            mae = float(np.mean(np.concatenate([np.abs(pred_med - gt_med), np.abs(pred_lat - gt_lat)])))
            mean_icc = float(np.mean([icc(pred_med, gt_med), icc(pred_lat, gt_lat)]))
            self.log("val/mjsw_mae", mae, prog_bar=False)
            self.log("val/mjsw_icc", mean_icc, prog_bar=False)
            self.log("val_mjsw_mae", mae, prog_bar=False)
            self.log("val_mjsw_icc", mean_icc, prog_bar=False)

        self.val_dice_scores.clear()
        self.val_dice_medial_scores.clear()
        self.val_dice_lateral_scores.clear()
        self.val_hd95_scores.clear()
        self.val_hd95_medial_scores.clear()
        self.val_hd95_lateral_scores.clear()
        self.val_pred_mjsw_medial.clear()
        self.val_pred_mjsw_lateral.clear()
        self.val_gt_mjsw_medial.clear()
        self.val_gt_mjsw_lateral.clear()
        super().on_validation_epoch_end()
