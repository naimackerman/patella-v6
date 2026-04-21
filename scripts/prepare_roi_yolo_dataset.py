"""Prepare a reviewed ROI dataset in YOLO format for Stage 1 training."""

from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import cv2
from omegaconf import DictConfig

from src.data.roi_annotations import load_roi_annotations
from src.models.roi_detector import ROI_CLASSES
from src.utils.seed import seed_everything


def _to_yolo_line(x1: float, y1: float, x2: float, y2: float, width: int, height: int, class_id: int) -> str:
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    annotations = load_roi_annotations(cfg.annotation_dir, cfg.data.root)
    out_root = Path(cfg.annotation_dir) / "roi_yolo"
    images_root = out_root / "images"
    labels_root = out_root / "labels"
    for split in ("train", "val", "test"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)

    grouped = annotations.groupby(["split", "image_id"], sort=True)
    for (split, image_id), rows in grouped:
        image_path = Path(rows.iloc[0]["path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load ROI training image: {image_path}")

        height, width = image.shape[:2]
        out_image = images_root / split / image_path.name
        out_label = labels_root / split / f"{image_id}.txt"
        shutil.copy2(image_path, out_image)

        lines = [
            _to_yolo_line(row.x1, row.y1, row.x2, row.y2, width, height, int(row.class_id))
            for row in rows.itertuples(index=False)
        ]
        out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_path = Path(cfg.annotation_dir) / "roi_yolo.yaml"
    yaml_text = "\n".join([
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(ROI_CLASSES)}",
        "names:",
        *[f"  {idx}: {name}" for idx, name in enumerate(ROI_CLASSES)],
        "",
    ])
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print(f"Prepared ROI YOLO dataset at {out_root}")
    print(f"Wrote dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
