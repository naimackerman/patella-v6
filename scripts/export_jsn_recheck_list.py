"""Export a prioritized JSN hard-case recheck list for manual review."""

from __future__ import annotations

import csv
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.jsn_seg_dataset import JSNSegDataset
from src.data.transforms import get_segmentation_transforms
from src.models.jsn_segmenter import create_jsn_segmenter
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.utils.metrics import dice_coefficient, hausdorff_95
from src.utils.seed import seed_everything


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "unetpp.yaml")


def _resolve_checkpoint_path(cfg: DictConfig) -> Path:
    explicit = getattr(cfg, "checkpoint_path", None)
    if explicit:
        return Path(str(explicit))

    ckpt_dir = Path(cfg.checkpoint_dir) / "jsn_segmenter"
    monitor = getattr(cfg, "checkpoint_monitor", None)
    if monitor in (None, "", "null", "None"):
        monitor = "val_dice"
    monitor = str(monitor)
    mode = str(getattr(cfg, "checkpoint_mode", "max"))
    ckpt_path = find_best_lightning_checkpoint(ckpt_dir, monitor=monitor, mode=mode) if ckpt_dir.exists() else None
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No trained JSN checkpoint found in {ckpt_dir} for monitor='{monitor}' mode='{mode}'."
        )
    return ckpt_path



def _priority_label(grade: int, side: str, dice: float, hd95: float) -> str:
    if grade >= 4 or dice < 0.70 or hd95 > 5.0:
        return "critical"
    if grade >= 3 or dice < 0.80 or hd95 > 4.0:
        return "high"
    return "normal"


def _priority_key(row: dict) -> tuple:
    priority_order = {"critical": 0, "high": 1, "normal": 2}
    return (
        priority_order[row["review_priority"]],
        -int(row["grade"]),
        float(row["dice"]),
        -float(row["hd95"]),
        0 if row["side"] == "R" else 1,
        row["image_id"],
    )


def _review_focus(grade: int, side: str) -> str:
    notes = []
    if grade >= 4:
        notes.append("Verify bone-on-bone or near-contact at the narrowest compartment.")
        notes.append("Check whether femoral and tibial contours should touch in the center or endpoint region.")
    elif grade >= 3:
        notes.append("Verify the narrowest compartment and intercondylar notch contour placement.")
    else:
        notes.append("Verify smooth contour adherence at the narrowest compartment.")

    if side == "R":
        notes.append("Confirm medial/lateral assignment on the right-knee view.")

    notes.append("Check endpoint hooks and contour oversmoothing across the notch.")
    return " ".join(notes)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)
    device = get_device()

    split = str(getattr(cfg, "split", "test"))
    top_k = int(getattr(cfg, "top_k", 30))
    default_out = Path(cfg.annotation_dir) / "packages" / "jsn_contours" / "jsn_recheck_priority.csv"
    output_path = Path(str(getattr(cfg, "output_path", default_out)))

    ckpt_path = _resolve_checkpoint_path(cfg)

    model = create_jsn_segmenter(model_cfg)
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.to(device)
    model.eval()

    image_root = Path(cfg.data.root) / split
    mask_root = Path(cfg.annotation_dir) / "jsn_masks" / split
    dataset = JSNSegDataset(
        image_dir=str(image_root),
        mask_dir=str(mask_root),
        transform=get_segmentation_transforms(cfg, is_train=False),
    )
    if len(dataset) == 0:
        raise ValueError(f"No JSN samples found for split '{split}'.")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers)
    rows: list[dict] = []

    images_preprocessed = Path(cfg.annotation_dir) / "packages" / "jsn_contours" / "images_preprocessed"
    panels_root = Path(cfg.result_dir) / "jsn_prediction_panels" / split
    overlays_root = Path(cfg.annotation_dir) / "jsn_mask_overlays" / split

    with torch.no_grad():
        for image, mask, image_id in loader:
            image = image.to(device)
            logits = model(image)
            pred = logits.argmax(dim=1).squeeze().cpu().numpy()
            gt = mask.squeeze().numpy()

            dice = float(dice_coefficient(pred, gt, num_classes=model_cfg.classes)["dice_mean"])
            per_class_hd = []
            for cls in range(1, model_cfg.classes):
                pred_bin = pred == cls
                gt_bin = gt == cls
                if pred_bin.any() and gt_bin.any():
                    per_class_hd.append(hausdorff_95(pred_bin, gt_bin))
            hd95 = float(np.mean(per_class_hd)) if per_class_hd else float("nan")

            src_path = Path(image_id[0])
            image_name = src_path.stem
            grade = int(src_path.parent.name)
            side = "L" if image_name.endswith("L") else "R" if image_name.endswith("R") else "?"
            prefixed_name = f"{split}_{grade}_{image_name}.png"
            annotator_path = images_preprocessed / prefixed_name

            row = {
                "priority_rank": 0,
                "review_priority": _priority_label(grade, side, dice, hd95),
                "image_id": image_name,
                "split": split,
                "grade": grade,
                "side": side,
                "dice": round(dice, 6),
                "hd95": round(hd95, 6) if not np.isnan(hd95) else "",
                "source_image_path": str(src_path),
                "annotator_image_path": str(annotator_path) if annotator_path.exists() else "",
                "prediction_panel_path": str(panels_root / f"{image_name}.png"),
                "mask_overlay_path": str(overlays_root / f"{image_name}.png"),
                "review_focus": _review_focus(grade, side),
            }
            rows.append(row)

    rows.sort(key=_priority_key)
    for idx, row in enumerate(rows[:top_k], start=1):
        row["priority_rank"] = idx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = rows[:top_k]
    fieldnames = list(selected[0].keys()) if selected else [
        "priority_rank",
        "review_priority",
        "image_id",
        "split",
        "grade",
        "side",
        "dice",
        "hd95",
        "source_image_path",
        "annotator_image_path",
        "prediction_panel_path",
        "mask_overlay_path",
        "review_focus",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Saved prioritized JSN recheck list to {output_path}")
    print(f"Checkpoint used: {ckpt_path}")
    print(f"Rows exported: {len(selected)}")


if __name__ == "__main__":
    main()
