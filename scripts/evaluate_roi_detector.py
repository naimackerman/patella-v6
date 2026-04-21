"""Evaluate reviewed ROI detections with mAP and detection-rate metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data.roi_annotations import load_roi_annotations
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


def _average_precision(tp_flags: np.ndarray, fp_flags: np.ndarray, num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)
    recall = tp_cum / max(num_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    for idx in range(len(precision) - 2, -1, -1):
        precision[idx] = max(precision[idx], precision[idx + 1])
    return float(np.trapz(precision, recall))


def _evaluate_threshold(predictions, ground_truths, iou_threshold: float):
    ap_by_class = {}
    detection_rate_by_class = {}
    for class_name in ROI_CLASSES:
        preds = [pred for pred in predictions if pred["class_name"] == class_name]
        preds.sort(key=lambda item: item["confidence"], reverse=True)
        gt_by_image = {
            image_id: [row for row in rows if row["class_name"] == class_name]
            for image_id, rows in ground_truths.items()
        }
        matched = {image_id: [False] * len(rows) for image_id, rows in gt_by_image.items()}
        tp_flags = []
        fp_flags = []
        for pred in preds:
            best_iou = 0.0
            best_index = -1
            gt_rows = gt_by_image.get(pred["image_id"], [])
            for idx, gt in enumerate(gt_rows):
                if matched[pred["image_id"]][idx]:
                    continue
                score = _iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
                if score > best_iou:
                    best_iou = score
                    best_index = idx
            if best_index >= 0 and best_iou >= iou_threshold:
                matched[pred["image_id"]][best_index] = True
                tp_flags.append(1.0)
                fp_flags.append(0.0)
            else:
                tp_flags.append(0.0)
                fp_flags.append(1.0)

        num_gt = sum(len(rows) for rows in gt_by_image.values())
        ap_by_class[class_name] = _average_precision(np.asarray(tp_flags), np.asarray(fp_flags), num_gt)
        matched_count = sum(sum(flags) for flags in matched.values())
        detection_rate_by_class[class_name] = float(matched_count / max(num_gt, 1))
    return ap_by_class, detection_rate_by_class


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    model_cfg = _resolve_model_cfg(cfg)
    annotations = load_roi_annotations(cfg.annotation_dir, cfg.data.root)

    ckpt_path = Path(cfg.checkpoint_dir) / "roi_detector" / "weights" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trained ROI detector checkpoint not found: {ckpt_path}")

    detector = ROIDetector(
        model_path=str(ckpt_path),
        conf_threshold=float(model_cfg.conf_threshold),
        model_variant=str(model_cfg.variant),
    )

    gt_by_image = defaultdict(list)
    image_meta = {}
    eval_rows = annotations[annotations["split"].isin(["val", "test"])].copy()
    for row in eval_rows.itertuples(index=False):
        gt_by_image[row.image_id].append({
            "class_name": row.class_name,
            "bbox_xyxy": (float(row.x1), float(row.y1), float(row.x2), float(row.y2)),
        })
        image_meta[row.image_id] = {"path": row.path, "split": row.split}

    predictions = []
    for image_id, meta in sorted(image_meta.items()):
        image = cv2.imread(str(meta["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load ROI evaluation image: {meta['path']}")
        for det in detector.predict(image):
            predictions.append({
                "image_id": image_id,
                "split": meta["split"],
                "class_name": det["class_name"],
                "confidence": float(det["confidence"]),
                "bbox_xyxy": tuple(map(float, det["bbox_xyxy"])),
            })

    thresholds = [round(value, 2) for value in np.arange(0.50, 1.00, 0.05)]
    aps = {}
    detection_rates = {}
    for threshold in thresholds:
        ap_by_class, detection_rate_by_class = _evaluate_threshold(predictions, gt_by_image, threshold)
        aps[str(threshold)] = ap_by_class
        detection_rates[str(threshold)] = detection_rate_by_class

    map50 = float(np.nanmean(list(aps["0.5"].values())))
    map5095 = float(np.nanmean([np.nanmean(list(aps[str(th)].values())) for th in thresholds]))
    detection_rate50 = float(np.nanmean(list(detection_rates["0.5"].values())))
    summary = {
        "checkpoint": str(ckpt_path),
        "num_eval_images": len(image_meta),
        "num_predictions": len(predictions),
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095,
        "detection_rate@0.5": detection_rate50,
        "per_class_ap@0.5": aps["0.5"],
        "per_class_detection_rate@0.5": detection_rates["0.5"],
        "iou_threshold_grid": thresholds,
    }

    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / "roi_detector_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved ROI evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
