"""Generate high-confidence JSN pseudo-masks for self-training."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.transforms import get_eval_transforms
from src.models.jsn_segmenter import create_jsn_segmenter
from src.utils.checkpoints import extract_model_state_dict, find_best_lightning_checkpoint, load_checkpoint
from src.utils.device import get_device
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
    monitor = getattr(cfg, "checkpoint_monitor", None)
    if monitor in (None, "", "null", "None"):
        monitor = "val_dice"
    ckpt_path = find_best_lightning_checkpoint(
        ckpt_dir,
        monitor=str(monitor),
        mode=str(getattr(cfg, "checkpoint_mode", "max")),
    ) if ckpt_dir.exists() else None
    if ckpt_path is None:
        raise FileNotFoundError(f"No JSN checkpoint found in {ckpt_dir}")
    return ckpt_path


def _pseudo_confidence(probabilities: np.ndarray, pred_mask: np.ndarray) -> float:
    foreground = pred_mask > 0
    if not foreground.any():
        return 0.0
    max_probs = probabilities.max(axis=0)
    return float(max_probs[foreground].mean())


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)
    device = get_device()
    transform = get_eval_transforms(cfg)

    ckpt_path = _resolve_checkpoint_path(cfg)
    model = create_jsn_segmenter(model_cfg)
    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.to(device)
    model.eval()

    split = str(getattr(cfg, "jsn_pseudo_split", "train"))
    data_root = Path(cfg.data.root) / split
    reviewed_mask_dir = Path(cfg.annotation_dir) / "jsn_masks" / split
    output_root = Path(str(getattr(cfg, "jsn_pseudo_output_dir", Path(cfg.annotation_dir) / "jsn_pseudo_masks"))) / split
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root.parent / f"{split}_jsn_pseudo_manifest.csv"
    project_root = Path(cfg.project_root)

    threshold = float(getattr(cfg.training, "jsn_pseudo_confidence_threshold", 0.90))
    min_foreground = int(getattr(cfg.training, "jsn_pseudo_min_foreground_pixels", 8))

    rows: list[dict[str, str | int | float]] = []
    with torch.no_grad():
        for image_path in tqdm(sorted(data_root.rglob("*.png")), desc=f"JSN pseudo {split}"):
            image_id = image_path.stem
            if (reviewed_mask_dir / f"{image_id}.png").exists():
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            transformed = transform(image=image)
            tensor = torch.from_numpy(transformed["image"]).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_mask = probs.argmax(axis=0).astype(np.uint8)
            foreground_pixels = int((pred_mask > 0).sum())
            confidence = _pseudo_confidence(probs, pred_mask)
            accepted = confidence >= threshold and foreground_pixels >= min_foreground

            if accepted:
                relative = image_path.relative_to(data_root).with_suffix(".npy")
                out_path = output_root / relative
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(out_path), pred_mask)

            rows.append({
                "image_id": image_id,
                "split": split,
                "source_image_path": _repo_relative(image_path, project_root),
                "pseudo_mask_path": _repo_relative(output_root / image_path.relative_to(data_root).with_suffix(".npy"), project_root) if accepted else "",
                "confidence": round(confidence, 6),
                "foreground_pixels": foreground_pixels,
                "accepted": int(accepted),
            })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "image_id",
            "split",
            "source_image_path",
            "pseudo_mask_path",
            "confidence",
            "foreground_pixels",
            "accepted",
        ])
        writer.writeheader()
        writer.writerows(rows)

    accepted_count = sum(int(row["accepted"]) for row in rows)
    print(f"Checkpoint used: {ckpt_path}")
    print(f"Saved {accepted_count} accepted pseudo-masks to {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
