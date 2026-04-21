"""ROI detection using YOLOv8-m for knee joint region localization."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from src.features.kneel_landmarks import (
    KNEELLandmarkDetector,
    compute_kneel_roi_boxes,
    extract_kneel_rois,
)
from src.utils.device import get_device


# ROI class names
ROI_CLASSES = ["joint_space", "medial_femur", "lateral_femur", "medial_tibia", "lateral_tibia"]


class ROIDetector:
    """Wrapper around Ultralytics YOLOv8-m for knee ROI detection."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.75,
        model_variant: str = "yolov8m.pt",
    ):
        from ultralytics import YOLO

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO(model_variant)
        self.conf_threshold = conf_threshold
        self.device = str(get_device())

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 224,
        batch: int = 8,
        project: str = "checkpoints",
        name: str = "roi_detector",
    ):
        """Train the YOLOv8-m model on annotated ROI data."""
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name,
            verbose=True,
        )

    def predict(self, image: np.ndarray) -> List[Dict]:
        """Run inference on a single image.

        Returns:
            List of dicts with keys: class_name, bbox_xyxy, confidence
        """
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = ROI_CLASSES[cls_id] if cls_id < len(ROI_CLASSES) else f"class_{cls_id}"
                    detections.append({
                        "class_name": cls_name,
                        "bbox_xyxy": box.xyxy[0].cpu().numpy().astype(int),
                        "confidence": float(box.conf.item()),
                    })
        return detections

    def extract_rois(
        self,
        image: np.ndarray,
        detections: List[Dict],
        osteophyte_roi_size: int = 140,
    ) -> Dict[str, np.ndarray]:
        """Crop ROI patches from the image based on detections.

        Returns:
            Dict mapping ROI name to cropped image patch.
        """
        h, w = image.shape[:2]
        rois = {}

        for det in detections:
            name = det["class_name"]
            x1, y1, x2, y2 = det["bbox_xyxy"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Resize osteophyte ROIs to standard size
            if name != "joint_space":
                crop = cv2.resize(crop, (osteophyte_roi_size, osteophyte_roi_size),
                                  interpolation=cv2.INTER_LINEAR)

            rois[name] = crop

        return rois

    @staticmethod
    def geometric_rois(image: np.ndarray, is_left: bool = False) -> Dict[str, np.ndarray]:
        """Fallback: compute ROIs geometrically from image center.

        Assumes the knee joint is roughly centered in the 224x224 image.

        Laterality convention (AP view):
          Right knee: medial = right half of image (cx:w)
          Left knee:  medial = left half of image (0:cx)
        """
        boxes = ROIDetector.geometric_boxes(image.shape[:2], is_left=is_left)
        return ROIDetector.crop_from_boxes(image, boxes)

    @staticmethod
    def landmark_rois(
        image: np.ndarray,
        is_left: bool = False,
        landmark_detector: Optional[KNEELLandmarkDetector] = None,
        apply_preprocessing: bool = True,
        require_reliable: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Landmark-based fallback using the configured KNEEL backend."""
        detector = landmark_detector or KNEELLandmarkDetector()
        landmarks = detector.predict(
            image,
            is_left=is_left,
            apply_preprocessing=apply_preprocessing,
            require_reliable=require_reliable,
        )
        return extract_kneel_rois(image, landmarks)

    @staticmethod
    def landmark_boxes(
        image: np.ndarray,
        is_left: bool = False,
        landmark_detector: Optional[KNEELLandmarkDetector] = None,
        apply_preprocessing: bool = True,
        require_reliable: bool = False,
    ) -> Dict[str, Tuple[int, int, int, int]]:
        detector = landmark_detector or KNEELLandmarkDetector()
        landmarks = detector.predict(
            image,
            is_left=is_left,
            apply_preprocessing=apply_preprocessing,
            require_reliable=require_reliable,
        )
        return compute_kneel_roi_boxes(image.shape[:2], landmarks)

    @staticmethod
    def geometric_boxes(
        image_shape: Tuple[int, int],
        is_left: bool = False,
    ) -> Dict[str, Tuple[int, int, int, int]]:
        h, w = image_shape[:2]
        cx, cy = w // 2, h // 2
        js_h = h // 4
        boxes = {
            "joint_space": (0, max(0, cy - js_h // 2), w, min(h, cy + js_h // 2)),
        }

        if is_left:
            boxes.update({
                "medial_femur": (0, 0, cx, cy),
                "lateral_femur": (cx, 0, w, cy),
                "medial_tibia": (0, cy, cx, h),
                "lateral_tibia": (cx, cy, w, h),
            })
        else:
            boxes.update({
                "medial_femur": (cx, 0, w, cy),
                "lateral_femur": (0, 0, cx, cy),
                "medial_tibia": (cx, cy, w, h),
                "lateral_tibia": (0, cy, cx, h),
            })
        return boxes

    @staticmethod
    def crop_from_boxes(
        image: np.ndarray,
        boxes: Dict[str, Tuple[int, int, int, int]],
        osteophyte_roi_size: int = 140,
    ) -> Dict[str, np.ndarray]:
        rois = {}
        h, w = image.shape[:2]
        for name, (x1, y1, x2, y2) in boxes.items():
            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, x1 + 1, w))
            y2 = int(np.clip(y2, y1 + 1, h))
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((16, 16), dtype=np.uint8)
            if name != "joint_space":
                crop = cv2.resize(crop, (osteophyte_roi_size, osteophyte_roi_size), interpolation=cv2.INTER_LINEAR)
            rois[name] = crop
        return rois
