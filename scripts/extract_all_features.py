"""Aggregate stage outputs into unified feature vectors for all images."""

from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.features.bootstrap_heuristics import (
    build_sclerosis_features,
    estimate_jsn_features,
    extract_geometric_subchondral_roi,
    extract_geometric_subchondral_roi_with_boxes,
    heuristic_osteophyte_features,
)
from src.features.feature_aggregator import FeatureAggregator
from src.features.kneel_landmarks import (
    KNEELLandmarkDetector,
    estimate_jsn_features_from_landmarks,
    extract_kneel_subchondral_rois,
)
from src.models.roi_detector import ROIDetector
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    feature_dir = Path(cfg.feature_dir)
    output_dir = feature_dir / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregator = FeatureAggregator()
    data_root = Path(cfg.data.root)
    roi_strategy = getattr(cfg.preprocessing, "roi_strategy", "geometric")
    landmark_detector = KNEELLandmarkDetector.from_preprocessing_cfg(cfg.preprocessing)
    jsn_maps = _load_split_feature_maps(feature_dir / "jsn", "jsn")
    osp_maps = _load_split_feature_maps(feature_dir / "osteophyte", "osteophyte")
    scl_maps = _load_split_feature_maps(feature_dir / "sclerosis", "sclerosis")
    lbp_maps = _load_split_feature_maps(feature_dir / "sclerosis", "sclerosis_lbp_histograms", suffix="")

    # Fit LBP PCA on training set histograms
    _fit_lbp_pca(aggregator, lbp_maps, output_dir)

    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        features_list = []
        labels_list = []
        image_ids = []

        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            grade = int(grade_dir.name)

            for img_path in tqdm(sorted(grade_dir.glob("*.png")), desc=f"{split}/KL{grade}"):
                image_id = img_path.stem
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                is_left = image_id.upper().endswith("L")

                jsn_features = _load_jsn(
                    jsn_maps.get(split, {}),
                    image_id,
                    image,
                    is_left,
                    roi_strategy,
                    landmark_detector,
                )
                osp_features = _load_osteophyte(
                    osp_maps.get(split, {}),
                    image_id,
                    feature_dir,
                    split,
                    image,
                    is_left,
                    roi_strategy,
                    landmark_detector,
                )
                scl_features = _load_sclerosis(
                    scl_maps.get(split, {}),
                    lbp_maps.get(split, {}),
                    image_id,
                    image,
                    is_left,
                    roi_strategy,
                    landmark_detector,
                )

                vec = aggregator.aggregate(jsn_features, osp_features, scl_features)
                features_list.append(vec)
                labels_list.append(grade)
                image_ids.append(image_id)

        features = np.array(features_list, dtype=np.float64)
        labels = np.array(labels_list, dtype=np.int64)

        np.savez(
            str(output_dir / f"{split}_features.npz"),
            features=features,
            labels=labels,
            image_ids=image_ids,
        )
        print(f"Saved {split}: {features.shape[0]} samples, {features.shape[1]} dims")

    train_data = np.load(output_dir / "train_features.npz")
    aggregator.fit_normalizer(train_data["features"])
    np.savez(
        str(output_dir / "normalizer_stats.npz"),
        mean=aggregator.train_mean,
        std=aggregator.train_std,
        median=aggregator.train_median,
    )


def _load_split_feature_maps(base_dir: Path, module_name: str, suffix: str = "_features") -> dict:
    split_maps = {}
    for split in ["train", "val", "test"]:
        file_path = base_dir / f"{split}_{module_name}{suffix}.npz"
        if not file_path.exists():
            continue
        data = np.load(file_path, allow_pickle=True)
        split_maps[split] = {key: data[key] for key in data.files}
    return split_maps


def _load_jsn(
    feature_map: dict,
    image_id: str,
    image: np.ndarray,
    is_left: bool,
    roi_strategy: str,
    landmark_detector: KNEELLandmarkDetector,
) -> dict:
    vector = feature_map.get(image_id)
    if vector is None:
        if (roi_strategy or "geometric").lower() in {"auto", "landmark"}:
            try:
                landmarks = landmark_detector.predict(image, is_left=is_left)
                return estimate_jsn_features_from_landmarks(landmarks)
            except Exception:
                pass
        return estimate_jsn_features(image, is_left=is_left)
    vector = np.asarray(vector, dtype=np.float64)
    return {
        "mJSW_medial": float(vector[0]),
        "mJSW_lateral": float(vector[1]),
        "jsw_profile": vector[2:18],
        "jsn_rate_medial": float(vector[18]),
        "jsn_rate_lateral": float(vector[19]),
        "jsw_ratio": float(vector[20]),
        "jsw_asymmetry": float(vector[21]),
    }


def _load_osteophyte(
    feature_map: dict,
    image_id: str,
    feature_dir: Path,
    split: str,
    image: np.ndarray,
    is_left: bool,
    roi_strategy: str,
    landmark_detector: KNEELLandmarkDetector,
) -> dict:
    vector = feature_map.get(image_id)
    if vector is not None:
        vector = np.asarray(vector, dtype=np.float64)
        return {
            "osp_grade_mf": float(vector[0]),
            "osp_grade_lf": float(vector[1]),
            "osp_grade_mt": float(vector[2]),
            "osp_grade_lt": float(vector[3]),
            "osp_sum": float(vector[4]),
            "osp_max": float(vector[5]),
            "osp_medial_sum": float(vector[6]),
            "osp_lateral_sum": float(vector[7]),
            "osp_femoral_sum": float(vector[8]),
            "osp_tibial_sum": float(vector[9]),
        }

    roi_images = _load_roi_images(
        feature_dir,
        split,
        image_id,
        image,
        is_left,
        roi_strategy,
        landmark_detector,
    )
    return heuristic_osteophyte_features(roi_images)


def _fit_lbp_pca(aggregator: FeatureAggregator, lbp_maps: dict, output_dir: Path):
    """Fit LBP PCA from training histograms and save parameters."""
    if not all(hasattr(aggregator, name) for name in ("fit_lbp_pca", "save_lbp_pca", "load_lbp_pca")):
        print(
            "FeatureAggregator does not expose LBP PCA methods; "
            "skipping raw LBP PCA and preserving the fixed 50-dim feature contract."
        )
        return

    train_lbp = lbp_maps.get("train", {})
    if not train_lbp:
        print("Warning: no LBP histograms found for training set, skipping PCA fit")
        # Try loading previously saved PCA
        pca_path = output_dir / "lbp_pca.npz"
        if pca_path.exists():
            aggregator.load_lbp_pca(str(pca_path))
            print(f"Loaded existing LBP PCA from {pca_path}")
        return

    med_hists = []
    lat_hists = []
    for image_id, hist_vec in train_lbp.items():
        hist_vec = np.asarray(hist_vec, dtype=np.float64)
        if len(hist_vec) == 108:
            med_hists.append(hist_vec[:54])
            lat_hists.append(hist_vec[54:])

    if len(med_hists) < 10:
        print(f"Warning: only {len(med_hists)} LBP histograms, skipping PCA fit")
        return

    aggregator.fit_lbp_pca(
        np.array(med_hists, dtype=np.float64),
        np.array(lat_hists, dtype=np.float64),
    )
    aggregator.save_lbp_pca(str(output_dir / "lbp_pca.npz"))
    print(f"Saved LBP PCA to {output_dir / 'lbp_pca.npz'}")


def _load_sclerosis(
    feature_map: dict,
    lbp_map: dict,
    image_id: str,
    image: np.ndarray,
    is_left: bool,
    roi_strategy: str,
    landmark_detector: KNEELLandmarkDetector,
) -> dict:
    vector = feature_map.get(image_id)
    if vector is not None:
        vector = np.asarray(vector, dtype=np.float64)
        result = {
            "scl_grade_medial": int(round(float(vector[0]))),
            "scl_grade_lateral": int(round(float(vector[1]))),
            "scl_intensity_medial": float(vector[2]),
            "scl_intensity_lateral": float(vector[3]),
            "scl_fractal_dim_med": float(vector[4]),
            "scl_fractal_dim_lat": float(vector[5]),
            "scl_glcm_contrast_med": float(vector[6]),
            "scl_glcm_dissimilarity_med": float(vector[7]),
            "scl_glcm_homogeneity_med": float(vector[8]),
            "scl_glcm_energy_med": float(vector[9]),
            "scl_glcm_correlation_med": float(vector[10]),
            "scl_glcm_contrast_lat": float(vector[11]),
            "scl_glcm_dissimilarity_lat": float(vector[12]),
            "scl_glcm_homogeneity_lat": float(vector[13]),
            "scl_glcm_energy_lat": float(vector[14]),
            "scl_glcm_correlation_lat": float(vector[15]),
            "scl_lbp_entropy_med": float(vector[16]),
            "scl_lbp_entropy_lat": float(vector[17]),
        }
        # Attach raw LBP histograms if available (for PCA projection in aggregator)
        lbp_vec = lbp_map.get(image_id)
        if lbp_vec is not None:
            lbp_vec = np.asarray(lbp_vec, dtype=np.float64)
            if len(lbp_vec) == 108:
                result["scl_lbp_hist_med"] = lbp_vec[:54]
                result["scl_lbp_hist_lat"] = lbp_vec[54:]
        return result

    if (roi_strategy or "geometric").lower() in {"auto", "landmark"}:
        try:
            landmarks = landmark_detector.predict(image, is_left=is_left)
            medial_roi, lateral_roi = extract_kneel_subchondral_rois(image, landmarks)
        except Exception:
            medial_roi, lateral_roi = extract_geometric_subchondral_roi(image, is_left=is_left)
    else:
        medial_roi, lateral_roi = extract_geometric_subchondral_roi(image, is_left=is_left)
    return build_sclerosis_features(medial_roi, lateral_roi)


def _load_roi_images(
    feature_dir: Path,
    split: str,
    image_id: str,
    image: np.ndarray,
    is_left: bool,
    roi_strategy: str,
    landmark_detector: KNEELLandmarkDetector,
) -> dict:
    roi_dir = feature_dir / "rois" / split
    roi_images = {}
    for site in ("medial_femur", "lateral_femur", "medial_tibia", "lateral_tibia"):
        roi_path = roi_dir / f"{image_id}_{site}.png"
        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) if roi_path.exists() else None
        roi_images[site] = roi
    if any(value is None for value in roi_images.values()) and (roi_strategy or "geometric").lower() in {"auto", "landmark"}:
        try:
            return ROIDetector.landmark_rois(image, is_left=is_left, landmark_detector=landmark_detector)
        except Exception:
            pass
    return roi_images


if __name__ == "__main__":
    main()
