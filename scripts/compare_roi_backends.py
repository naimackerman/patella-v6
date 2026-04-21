"""Compare detector, KNEEL, and heuristic ROI backends on reviewed ROI boxes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data.roi_annotations import load_roi_annotations
from src.features.kneel_landmarks import KNEELLandmarkDetector
from src.models.roi_detector import ROIDetector, ROI_CLASSES
from src.utils.seed import seed_everything


def _resolve_model_cfg(cfg: DictConfig) -> DictConfig:
    if "model" in cfg and cfg.model is not None:
        return cfg.model
    return OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model" / "yolov8.yaml")


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)
    annotations = load_roi_annotations(cfg.annotation_dir, cfg.data.root)
    image_ids = annotations["image_id"].drop_duplicates().tolist()[:50]
    gt_by_image = {
        image_id: {
            row.class_name: (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
            for row in annotations[annotations["image_id"] == image_id].itertuples(index=False)
        }
        for image_id in image_ids
    }

    ckpt_path = Path(cfg.checkpoint_dir) / "roi_detector" / "weights" / "best.pt"
    detector = None
    if ckpt_path.exists():
        detector = ROIDetector(
            model_path=str(ckpt_path),
            conf_threshold=float(model_cfg.conf_threshold),
            model_variant=str(model_cfg.variant),
        )
    landmark_detector = KNEELLandmarkDetector.from_preprocessing_cfg(cfg.preprocessing)

    summaries = {"detector": [], "kneel_repo_or_fallback": [], "heuristic": []}
    for image_id in image_ids:
        row = annotations[annotations["image_id"] == image_id].iloc[0]
        image = cv2.imread(str(row["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load ROI comparison image: {row['path']}")
        is_left = image_id.upper().endswith("L")

        gt_boxes = gt_by_image[image_id]
        if detector is not None:
            pred_boxes = {
                det["class_name"]: tuple(map(float, det["bbox_xyxy"]))
                for det in detector.predict(image)
            }
            _accumulate_backend_scores(summaries["detector"], gt_boxes, pred_boxes)

        kneel_boxes = ROIDetector.landmark_boxes(image, is_left=is_left, landmark_detector=landmark_detector)
        _accumulate_backend_scores(summaries["kneel_repo_or_fallback"], gt_boxes, kneel_boxes)

        heuristic_boxes = ROIDetector.geometric_boxes(image.shape[:2], is_left=is_left)
        _accumulate_backend_scores(summaries["heuristic"], gt_boxes, heuristic_boxes)

    report = {}
    for backend, values in summaries.items():
        report[backend] = {
            "mean_iou": float(np.mean(values)) if values else 0.0,
            "median_iou": float(np.median(values)) if values else 0.0,
            "num_box_pairs": len(values),
        }

    out_path = Path(cfg.result_dir) / "roi_backend_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved ROI backend comparison to {out_path}")


def _accumulate_backend_scores(target: list[float], gt_boxes: dict, pred_boxes: dict):
    for class_name in ROI_CLASSES:
        gt_box = gt_boxes.get(class_name)
        pred_box = pred_boxes.get(class_name)
        if gt_box is None or pred_box is None:
            continue
        target.append(_iou(gt_box, pred_box))


if __name__ == "__main__":
    main()
