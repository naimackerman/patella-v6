"""Evaluate the JSN segmenter on reviewed masks with Dice/Hausdorff/mJSW metrics."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.jsn_seg_dataset import JSNSegDataset
from src.data.transforms import get_segmentation_transforms
from src.features.jsw_computation import compute_jsn_measurements, get_jsn_measurement_kwargs
from src.models.jsn_segmenter import create_jsn_segmenter
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.utils.metrics import dice_coefficient, hausdorff_95, icc
from src.utils.seed import seed_everything


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "unetpp.yaml")


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    explicit = getattr(cfg, "checkpoint_path", None)
    if explicit:
        return Path(str(explicit))

    checkpoint_subdir = str(getattr(cfg, "jsn_checkpoint_subdir", "jsn_segmenter"))
    ckpt_dir = Path(cfg.checkpoint_dir) / checkpoint_subdir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No trained JSN checkpoint directory found in {ckpt_dir}")

    monitor = getattr(cfg, "checkpoint_monitor", None)
    if monitor in (None, "", "null", "None"):
        monitor = "val_dice"
    monitor = str(monitor)
    mode = str(getattr(cfg, "checkpoint_mode", "max"))
    ckpt_path = find_best_lightning_checkpoint(ckpt_dir, monitor=monitor, mode=mode)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No trained JSN checkpoint found in {ckpt_dir} for monitor='{monitor}' mode='{mode}'."
        )
    return ckpt_path


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)
    device = get_device()
    measurement_kwargs = get_jsn_measurement_kwargs(cfg)

    mask_dir = Path(cfg.annotation_dir) / "jsn_masks" / "test"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Reviewed JSN test masks not found: {mask_dir}")

    ckpt_path = _resolve_checkpoint_path(cfg)

    model = create_jsn_segmenter(model_cfg)
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.to(device)
    model.eval()

    dataset = JSNSegDataset(
        image_dir=str(Path(cfg.data.root) / "test"),
        mask_dir=str(mask_dir),
        transform=get_segmentation_transforms(cfg, is_train=False),
    )
    if len(dataset) == 0:
        raise ValueError("No JSN test samples matched the reviewed masks.")

    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)
    dice_scores = []
    dice_medial_scores = []
    dice_lateral_scores = []
    hd95_scores = []
    hd95_medial_scores = []
    hd95_lateral_scores = []
    pred_mjsw_med = []
    pred_mjsw_lat = []
    gt_mjsw_med = []
    gt_mjsw_lat = []

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            logits = model(images)
            pred_masks = logits.argmax(dim=1).cpu().numpy()
            gt_masks = masks.numpy()
            for pred, gt in zip(pred_masks, gt_masks):
                dice = dice_coefficient(pred, gt, num_classes=model_cfg.classes)
                dice_scores.append(dice["dice_mean"])
                dice_medial_scores.append(float(dice.get("dice_class_1", 0.0)))
                dice_lateral_scores.append(float(dice.get("dice_class_2", 0.0)))
                per_class_hd = []
                for cls in range(1, model_cfg.classes):
                    pred_bin = pred == cls
                    gt_bin = gt == cls
                    if pred_bin.any() and gt_bin.any():
                        hd = hausdorff_95(pred_bin, gt_bin)
                        per_class_hd.append(hd)
                        if cls == 1:
                            hd95_medial_scores.append(float(hd))
                        elif cls == 2:
                            hd95_lateral_scores.append(float(hd))
                if per_class_hd:
                    hd95_scores.append(float(np.mean(per_class_hd)))

                pred_features = compute_jsn_measurements(pred, **measurement_kwargs)
                gt_features = compute_jsn_measurements(gt, **measurement_kwargs)
                pred_mjsw_med.append(pred_features["mJSW_medial"])
                pred_mjsw_lat.append(pred_features["mJSW_lateral"])
                gt_mjsw_med.append(gt_features["mJSW_medial"])
                gt_mjsw_lat.append(gt_features["mJSW_lateral"])

    pred_mjsw_med = np.asarray(pred_mjsw_med, dtype=np.float64)
    pred_mjsw_lat = np.asarray(pred_mjsw_lat, dtype=np.float64)
    gt_mjsw_med = np.asarray(gt_mjsw_med, dtype=np.float64)
    gt_mjsw_lat = np.asarray(gt_mjsw_lat, dtype=np.float64)
    summary = {
        "checkpoint": str(ckpt_path),
        "num_samples": len(dataset),
        "dice_mean": float(np.mean(dice_scores)),
        "dice_medial_mean": float(np.mean(dice_medial_scores)),
        "dice_lateral_mean": float(np.mean(dice_lateral_scores)),
        "hausdorff95_mean": float(np.mean(hd95_scores)) if hd95_scores else float("nan"),
        "hausdorff95_medial_mean": float(np.mean(hd95_medial_scores)) if hd95_medial_scores else float("nan"),
        "hausdorff95_lateral_mean": float(np.mean(hd95_lateral_scores)) if hd95_lateral_scores else float("nan"),
        "mjsw_mae": float(np.mean(np.concatenate([np.abs(pred_mjsw_med - gt_mjsw_med), np.abs(pred_mjsw_lat - gt_mjsw_lat)]))),
        "mjsw_icc_medial": float(icc(pred_mjsw_med, gt_mjsw_med)),
        "mjsw_icc_lateral": float(icc(pred_mjsw_lat, gt_mjsw_lat)),
    }
    summary["mjsw_icc_mean"] = float(np.mean([summary["mjsw_icc_medial"], summary["mjsw_icc_lateral"]]))

    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_name = str(getattr(cfg, "jsn_evaluation_filename", "jsn_evaluation.json"))
    out_path = result_dir / output_name
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved JSN evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
