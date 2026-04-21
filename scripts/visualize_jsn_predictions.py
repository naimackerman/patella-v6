"""Visualize JSN predictions from the best checkpoint."""

from __future__ import annotations

from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.transforms import get_segmentation_transforms
from src.features.jsw_computation import compute_jsn_measurements, get_jsn_measurement_kwargs
from src.models.jsn_segmenter import create_jsn_segmenter
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.utils.seed import seed_everything


MASK_COLORS = {
    1: np.array([60, 200, 60], dtype=np.uint8),
    2: np.array([60, 80, 230], dtype=np.uint8),
}


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



def _overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    for cls, color in MASK_COLORS.items():
        overlay[mask == cls] = color
    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
    blended[mask == 0] = base[mask == 0]
    return blended


def _error_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = base.copy()
    correct = (pred == gt) & (gt > 0)
    missed = (gt > 0) & (pred == 0)
    wrong = (pred > 0) & (gt == 0)
    swapped = (pred > 0) & (gt > 0) & (pred != gt)

    out[correct] = np.array([60, 200, 60], dtype=np.uint8)
    out[missed] = np.array([0, 255, 255], dtype=np.uint8)
    out[wrong] = np.array([0, 0, 255], dtype=np.uint8)
    out[swapped] = np.array([255, 0, 255], dtype=np.uint8)
    return cv2.addWeighted(out, 0.5, base, 0.5, 0.0)


def _measurement_overlay(
    image: np.ndarray,
    measurement_pairs: list[tuple[tuple[float, float], tuple[float, float], float]],
) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for p1, p2, dist in measurement_pairs:
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 180, 0), 1, cv2.LINE_AA)
        mx, my = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        cv2.putText(
            canvas,
            f"{dist:.1f}",
            (mx + 2, my - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _panel(title: str, image: np.ndarray) -> np.ndarray:
    title_h = 28
    canvas = np.full((image.shape[0] + title_h, image.shape[1], 3), 245, dtype=np.uint8)
    canvas[title_h:] = image
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)
    device = get_device()
    measurement_kwargs = get_jsn_measurement_kwargs(cfg)
    split = str(getattr(cfg, "vis_split", "test"))
    max_images = int(getattr(cfg, "vis_max_images", 60))

    ckpt_path = _resolve_checkpoint_path(cfg)

    model = create_jsn_segmenter(model_cfg)
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.to(device)
    model.eval()

    image_root = Path(cfg.data.root) / split
    mask_root = Path(cfg.annotation_dir) / "jsn_masks" / split
    if not image_root.exists() or not mask_root.exists():
        raise FileNotFoundError(f"Missing image/mask root for split '{split}': {image_root}, {mask_root}")

    transform = get_segmentation_transforms(cfg, is_train=False)
    out_root = Path(cfg.result_dir) / "jsn_prediction_panels" / split
    out_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    for img_path in sorted(image_root.rglob("*.png")):
        image_id = img_path.stem
        mask_path = mask_root / f"{image_id}.png"
        if not mask_path.exists():
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or gt_mask is None:
            continue

        transformed = transform(image=image, mask=gt_mask)
        image_tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        gt_measurements = compute_jsn_measurements(gt_mask, **measurement_kwargs)
        pred_measurements = compute_jsn_measurements(pred_mask, **measurement_kwargs)

        raw_panel = _panel("Image", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        gt_panel = _panel("Ground Truth", _overlay_mask(image, gt_mask))
        gt_measure_panel = _panel("GT Measurements", _measurement_overlay(image, gt_measurements["measurement_pairs"]))
        pred_panel = _panel("Prediction", _overlay_mask(image, pred_mask))
        pred_measure_panel = _panel("Pred Measurements", _measurement_overlay(image, pred_measurements["measurement_pairs"]))
        err_panel = _panel("Error", _error_overlay(image, pred_mask, gt_mask))

        canvas = np.concatenate(
            [raw_panel, gt_panel, gt_measure_panel, pred_panel, pred_measure_panel, err_panel],
            axis=1,
        )
        cv2.imwrite(str(out_root / f"{image_id}.png"), canvas)

        saved += 1
        if saved >= max_images:
            break

    print(f"Saved {saved} prediction panels to {out_root}")
    print(f"Checkpoint used: {ckpt_path}")


if __name__ == "__main__":
    main()
