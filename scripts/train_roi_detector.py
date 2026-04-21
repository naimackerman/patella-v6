"""Train YOLOv8-m for knee ROI detection."""

import hydra
from omegaconf import DictConfig

from src.models.roi_detector import ROIDetector
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    model_cfg = cfg.model
    detector = ROIDetector(conf_threshold=model_cfg.conf_threshold)

    detector.train(
        data_yaml=f"{cfg.annotation_dir}/roi_yolo.yaml",
        epochs=model_cfg.epochs,
        imgsz=model_cfg.imgsz,
        batch=model_cfg.batch,
        project=cfg.checkpoint_dir,
        name="roi_detector",
    )


if __name__ == "__main__":
    main()
