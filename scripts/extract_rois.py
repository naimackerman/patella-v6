"""Extract ROI patches from all dataset images using trained detector."""

import csv
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.features.kneel_landmarks import KNEELLandmarkDetector
from src.models.roi_detector import ROIDetector, ROI_CLASSES
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    roi_strategy = getattr(cfg.preprocessing, "roi_strategy", "auto").lower()
    landmark_detector = KNEELLandmarkDetector.from_preprocessing_cfg(cfg.preprocessing)

    # Load trained detector
    model_path = Path(cfg.checkpoint_dir) / "roi_detector" / "weights" / "best.pt"
    if model_path.exists() and roi_strategy in {"auto", "detector"}:
        detector = ROIDetector(str(model_path))
    else:
        if roi_strategy == "landmark":
            print("Using landmark ROI strategy.")
        else:
            print(f"No trained model at {model_path}, using {roi_strategy} fallback")
        detector = None

    output_dir = Path(cfg.feature_dir) / "rois"
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    data_root = Path(cfg.data.root)

    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            for img_path in tqdm(
                sorted(grade_dir.glob("*.png")),
                desc=f"{split}/{grade_dir.name}",
            ):
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    failures.append({"image": str(img_path), "reason": "load_failed"})
                    continue

                image_id = img_path.stem
                is_left = image_id.upper().endswith("L")

                if detector is not None:
                    detections = detector.predict(image)
                    if len(detections) < 3:
                        rois = _fallback_rois(image, is_left, roi_strategy, landmark_detector)
                        failures.append({"image": str(img_path), "reason": "few_detections"})
                    else:
                        rois = detector.extract_rois(image, detections)
                else:
                    rois = _fallback_rois(image, is_left, roi_strategy, landmark_detector)

                for roi_name, roi_patch in rois.items():
                    out_path = output_dir / split / f"{image_id}_{roi_name}.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), roi_patch)

    # Log failures
    if failures:
        fail_path = Path(cfg.result_dir) / "roi_detection_failures.csv"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "reason"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"Logged {len(failures)} failures to {fail_path}")
def _fallback_rois(image, is_left: bool, roi_strategy: str, landmark_detector):
    if roi_strategy in {"auto", "landmark"}:
        try:
            return ROIDetector.landmark_rois(image, is_left=is_left, landmark_detector=landmark_detector)
        except Exception:
            pass
    return ROIDetector.geometric_rois(image, is_left=is_left)


if __name__ == "__main__":
    main()
