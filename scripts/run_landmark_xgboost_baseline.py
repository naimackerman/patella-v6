"""Full-dataset landmark bootstrap evaluation without annotation-export overhead."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.features.bootstrap_heuristics import (
    ROI_SITES,
    QuantileCalibrator,
    build_sclerosis_features,
    fit_osteophyte_calibrator,
    fit_sclerosis_calibrator,
    heuristic_osteophyte_features,
)
from src.features.feature_aggregator import FeatureAggregator
from src.features.kneel_landmarks import (
    KNEELLandmarkDetector,
    estimate_jsn_features_from_landmarks,
    extract_kneel_rois,
    extract_kneel_subchondral_rois,
)
from src.models.kl_xgboost import KLXGBoostClassifier
from src.utils.metrics import per_class_metrics, quadratic_weighted_kappa
from src.utils.seed import seed_everything


def scan_dataset(data_root: Path):
    splits = {}
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        rows = []
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            grade = int(grade_dir.name)
            for img_path in sorted(grade_dir.glob("*.png")):
                rows.append({
                    "image_id": img_path.stem,
                    "grade": grade,
                    "path": img_path,
                    "is_left": img_path.stem.upper().endswith("L"),
                })
        splits[split] = rows
    return splits


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    data_root = Path(cfg.data.root)
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = Path(cfg.feature_dir)
    agg_dir = feature_dir / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    detector = KNEELLandmarkDetector.from_preprocessing_cfg(cfg.preprocessing)
    splits = scan_dataset(data_root)

    print("Fitting landmark bootstrap calibrators...")
    osp_rois = []
    scl_rois = []
    for rec in tqdm(splits.get("train", []), desc="Landmark calibrators"):
        image = cv2.imread(str(rec["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        landmarks = detector.predict(image, is_left=rec["is_left"])
        rois = extract_kneel_rois(image, landmarks)
        medial_roi, lateral_roi = extract_kneel_subchondral_rois(image, landmarks)
        for site in ROI_SITES:
            osp_rois.append((site, rois.get(site)))
        scl_rois.extend([medial_roi, lateral_roi])

    osp_calibrator = fit_osteophyte_calibrator(osp_rois)
    scl_calibrator = fit_sclerosis_calibrator(scl_rois)
    print(f"Osteophyte thresholds: {osp_calibrator.thresholds.tolist()}")
    print(f"Sclerosis thresholds: {scl_calibrator.thresholds.tolist()}")

    aggregator = FeatureAggregator()
    jsn_cache = {}

    for split, rows in splits.items():
        features = []
        labels = []
        image_ids = []

        for rec in tqdm(rows, desc=f"Landmark aggregate {split}"):
            image = cv2.imread(str(rec["path"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            landmarks = detector.predict(image, is_left=rec["is_left"])
            jsn = estimate_jsn_features_from_landmarks(landmarks)
            jsn_cache[rec["image_id"]] = {"features": jsn, "grade": rec["grade"], "split": split}

        kl0_med = [
            v["features"]["mJSW_medial"]
            for v in jsn_cache.values()
            if v["grade"] == 0 and v["split"] == "train"
        ]
        kl0_lat = [
            v["features"]["mJSW_lateral"]
            for v in jsn_cache.values()
            if v["grade"] == 0 and v["split"] == "train"
        ]
        ref_med = float(np.median(kl0_med)) if kl0_med else 15.0
        ref_lat = float(np.median(kl0_lat)) if kl0_lat else 15.0

        for rec in tqdm(rows, desc=f"Landmark features {split}"):
            image = cv2.imread(str(rec["path"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            landmarks = detector.predict(image, is_left=rec["is_left"])
            rois = extract_kneel_rois(image, landmarks)
            medial_roi, lateral_roi = extract_kneel_subchondral_rois(image, landmarks)

            jsn = dict(jsn_cache[rec["image_id"]]["features"])
            jsn["jsn_rate_medial"] = float(np.clip(100.0 * (1.0 - jsn["mJSW_medial"] / max(ref_med, 1e-6)), 0, 100))
            jsn["jsn_rate_lateral"] = float(np.clip(100.0 * (1.0 - jsn["mJSW_lateral"] / max(ref_lat, 1e-6)), 0, 100))

            osp = heuristic_osteophyte_features(rois, calibrator=osp_calibrator)
            scl = build_sclerosis_features(medial_roi, lateral_roi, calibrator=scl_calibrator)
            vec = aggregator.aggregate(jsn, osp, scl)

            features.append(vec)
            labels.append(rec["grade"])
            image_ids.append(rec["image_id"])

        np.savez(
            str(agg_dir / f"{split}_features.npz"),
            features=np.asarray(features, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int64),
            image_ids=np.asarray(image_ids),
        )

    train = np.load(agg_dir / "train_features.npz", allow_pickle=True)
    test = np.load(agg_dir / "test_features.npz", allow_pickle=True)
    aggregator.fit_normalizer(train["features"])
    np.savez(str(agg_dir / "normalizer_stats.npz"), mean=aggregator.train_mean, std=aggregator.train_std)

    X_train = aggregator.normalize(train["features"])
    y_train = train["labels"]
    X_test = aggregator.normalize(test["features"])
    y_test = test["labels"]

    model = KLXGBoostClassifier(cfg.model)
    cv_scores = model.fit_cv(X_train, y_train, n_folds=cfg.model.cv_folds)
    preds, probs = model.predict(X_test)
    qwk = quadratic_weighted_kappa(y_test, preds)
    metrics = per_class_metrics(y_test, preds, cfg.data.class_names)

    summary = {
        "feature_dir": str(agg_dir),
        "cv_qwk_mean": float(cv_scores["qwk_mean"]),
        "cv_qwk_std": float(cv_scores["qwk_std"]),
        "test_qwk": float(qwk),
        "test_accuracy": float(metrics["accuracy"]),
        "per_class_f1": {
            cls_name: float(metrics.get(f"f1_{cls_name}", 0.0))
            for cls_name in cfg.data.class_names
        },
        "osteophyte_thresholds": osp_calibrator.thresholds.astype(float).tolist(),
        "sclerosis_thresholds": scl_calibrator.thresholds.astype(float).tolist(),
    }

    out_path = result_dir / "landmark_xgboost_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
