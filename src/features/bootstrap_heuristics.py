"""Image-only bootstrap heuristics for annotation suggestions and fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import cv2
import numpy as np

from src.features.texture_features import (
    compute_fractal_dimension,
    extract_glcm_features,
    extract_intensity_stats,
    extract_lbp_features,
)


ROI_SITES = ("medial_femur", "lateral_femur", "medial_tibia", "lateral_tibia")


@dataclass
class QuantileCalibrator:
    """Quantile-based discretizer used for heuristic grade suggestions."""

    thresholds: np.ndarray

    def grade(self, score: float) -> int:
        return int(np.digitize(score, self.thresholds, right=False))

    def to_dict(self) -> Dict[str, list]:
        return {"thresholds": self.thresholds.astype(float).tolist()}

    def confidence(self, score: float) -> str:
        if not np.isfinite(score):
            return "low"
        if self.thresholds.size == 0:
            return "medium"
        distance = min(abs(float(score) - float(th)) for th in self.thresholds)
        if self.thresholds.size == 1:
            local_scale = max(abs(float(self.thresholds[0])), 1.0)
        else:
            gaps = np.diff(self.thresholds.astype(np.float64))
            positive_gaps = gaps[gaps > 1.0e-8]
            local_scale = float(np.median(positive_gaps)) if positive_gaps.size else 1.0
        normalized_margin = distance / max(local_scale, 1.0e-6)
        if normalized_margin >= 0.75:
            return "high"
        if normalized_margin >= 0.35:
            return "medium"
        return "low"

    @classmethod
    def from_scores(
        cls,
        scores: Iterable[float],
        quantiles: Tuple[float, ...],
        default_thresholds: Tuple[float, ...],
    ) -> "QuantileCalibrator":
        score_array = np.asarray(list(scores), dtype=np.float64)
        if score_array.size == 0:
            return cls(np.asarray(default_thresholds, dtype=np.float64))

        raw_thresholds = np.quantile(score_array, quantiles)
        thresholds = np.asarray(raw_thresholds, dtype=np.float64)
        thresholds = np.maximum.accumulate(thresholds)
        if np.unique(thresholds).size != thresholds.size:
            thresholds = np.asarray(default_thresholds, dtype=np.float64)
        return cls(thresholds)

    @classmethod
    def from_labeled_scores(
        cls,
        scores: Iterable[float],
        labels: Iterable[int],
        default_thresholds: Tuple[float, ...],
    ) -> "QuantileCalibrator":
        score_array = np.asarray(list(scores), dtype=np.float64)
        label_array = np.asarray(list(labels), dtype=np.int64)
        valid = np.isfinite(score_array)
        score_array = score_array[valid]
        label_array = label_array[valid]
        if score_array.size == 0 or label_array.size == 0:
            return cls(np.asarray(default_thresholds, dtype=np.float64))

        num_thresholds = len(default_thresholds)
        thresholds = []
        for boundary in range(num_thresholds):
            lower = score_array[label_array <= boundary]
            upper = score_array[label_array > boundary]
            if lower.size == 0 or upper.size == 0:
                thresholds.append(float(default_thresholds[boundary]))
                continue
            lower_anchor = float(np.quantile(lower, 0.75))
            upper_anchor = float(np.quantile(upper, 0.25))
            if upper_anchor <= lower_anchor:
                lower_anchor = float(np.median(lower))
                upper_anchor = float(np.median(upper))
            thresholds.append((lower_anchor + upper_anchor) / 2.0)

        threshold_array = np.asarray(thresholds, dtype=np.float64)
        threshold_array = np.maximum.accumulate(threshold_array)
        if np.unique(threshold_array).size != threshold_array.size:
            threshold_array = np.asarray(default_thresholds, dtype=np.float64)
        return cls(threshold_array)


def estimate_jsn_features(image: np.ndarray, is_left: bool = False) -> dict:
    """Estimate JSN features directly from image intensities."""
    h, w = image.shape[:2]
    cy = h // 2
    band_h = max(h // 6, 8)
    js_band = image[max(0, cy - band_h):min(h, cy + band_h), :]
    midline = w // 2

    if is_left:
        medial_band = js_band[:, :midline]
        lateral_band = js_band[:, midline:]
    else:
        medial_band = js_band[:, midline:]
        lateral_band = js_band[:, :midline]

    medial_profile = medial_band.mean(axis=1)
    lateral_profile = lateral_band.mean(axis=1)

    med_threshold = np.percentile(medial_profile, 45)
    lat_threshold = np.percentile(lateral_profile, 45)
    mjsw_med = float(np.sum(medial_profile < med_threshold))
    mjsw_lat = float(np.sum(lateral_profile < lat_threshold))

    profile_points = 16
    jsw_profile = np.zeros(profile_points, dtype=np.float64)
    for i in range(profile_points):
        x = int(w * (i + 0.5) / profile_points)
        col = js_band[:, max(0, x - 2):min(w, x + 3)].mean(axis=1)
        col_thresh = np.percentile(col, 45)
        jsw_profile[i] = float(np.sum(col < col_thresh))

    ratio = mjsw_med / max(mjsw_lat, 1e-6)
    asymmetry = abs(mjsw_med - mjsw_lat) / max(mjsw_med + mjsw_lat, 1e-6)
    return {
        "mJSW_medial": mjsw_med,
        "mJSW_lateral": mjsw_lat,
        "jsw_profile": jsw_profile,
        "jsn_rate_medial": 0.0,
        "jsn_rate_lateral": 0.0,
        "jsw_ratio": ratio,
        "jsw_asymmetry": asymmetry,
    }


def extract_geometric_subchondral_roi(
    image: np.ndarray,
    is_left: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    medial_roi, lateral_roi, _, _ = extract_geometric_subchondral_roi_with_boxes(image, is_left=is_left)
    return medial_roi, lateral_roi


def extract_geometric_subchondral_roi_with_boxes(
    image: np.ndarray,
    is_left: bool = False,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Tuple[int, int, int, int]],
    Optional[Tuple[int, int, int, int]],
]:
    """Approximate medial/lateral subchondral ROIs from image geometry alone."""
    h, w = image.shape[:2]
    cy = h // 2
    midline = w // 2

    y_top = min(max(cy + h // 8, 0), h - 1)
    y_bottom = min(y_top + max(h // 8, 10), h)
    if y_bottom <= y_top + 2:
        return None, None, None, None

    if is_left:
        medial_patch = image[y_top:y_bottom, :midline]
        lateral_patch = image[y_top:y_bottom, midline:]
        medial_box = (0, y_top, midline, y_bottom)
        lateral_box = (midline, y_top, w, y_bottom)
    else:
        medial_patch = image[y_top:y_bottom, midline:]
        lateral_patch = image[y_top:y_bottom, :midline]
        medial_box = (midline, y_top, w, y_bottom)
        lateral_box = (0, y_top, midline, y_bottom)

    medial_roi = cv2.resize(medial_patch, (64, 64), interpolation=cv2.INTER_LINEAR)
    lateral_roi = cv2.resize(lateral_patch, (64, 64), interpolation=cv2.INTER_LINEAR)
    return medial_roi, lateral_roi, medial_box, lateral_box


def osteophyte_roi_score(roi: Optional[np.ndarray], site: str) -> float:
    """Compute an image-only osteophyte proxy score for one ROI."""
    if roi is None or roi.size == 0:
        return 0.0

    roi_uint8 = _preprocess_roi(roi)
    height = roi_uint8.shape[0]
    margin_h = max(height // 4, 10)
    if "femur" in site:
        joint_margin = roi_uint8[-margin_h:, :]
    else:
        joint_margin = roi_uint8[:margin_h, :]

    edges = cv2.Canny(joint_margin, 30, 90)
    lap_var = float(cv2.Laplacian(joint_margin, cv2.CV_32F).var())
    top_hat = cv2.morphologyEx(
        joint_margin,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    edge_density = float(edges.mean() / 255.0)
    bright_response = float(top_hat.mean() / 255.0)
    texture_response = float(np.clip(lap_var / 2500.0, 0.0, 1.0))
    return 0.45 * bright_response + 0.35 * edge_density + 0.20 * texture_response


def heuristic_osteophyte_features(
    rois: Dict[str, np.ndarray],
    calibrator: Optional[QuantileCalibrator | Mapping[str, QuantileCalibrator]] = None,
) -> Dict[str, float]:
    """Convert ROI patches into the 10-dim osteophyte feature dict."""
    site_scores = {
        site: osteophyte_roi_score(rois.get(site), site)
        for site in ROI_SITES
    }
    if calibrator is None:
        raw_grades = {site: min(3, int(round(score * 3.0))) for site, score in site_scores.items()}
    elif isinstance(calibrator, Mapping):
        raw_grades = {}
        for site, score in site_scores.items():
            site_calibrator = calibrator.get(site)
            if site_calibrator is None:
                raw_grades[site] = min(3, int(round(score * 3.0)))
            else:
                raw_grades[site] = site_calibrator.grade(score)
    else:
        raw_grades = {site: calibrator.grade(score) for site, score in site_scores.items()}

    mf = float(raw_grades["medial_femur"])
    lf = float(raw_grades["lateral_femur"])
    mt = float(raw_grades["medial_tibia"])
    lt = float(raw_grades["lateral_tibia"])
    return {
        "osp_grade_mf": mf,
        "osp_grade_lf": lf,
        "osp_grade_mt": mt,
        "osp_grade_lt": lt,
        "osp_sum": mf + lf + mt + lt,
        "osp_max": max(mf, lf, mt, lt),
        "osp_medial_sum": mf + mt,
        "osp_lateral_sum": lf + lt,
        "osp_femoral_sum": mf + lf,
        "osp_tibial_sum": mt + lt,
    }


def sclerosis_roi_score(roi: Optional[np.ndarray]) -> float:
    """Compute an image-only sclerosis proxy score for one subchondral ROI."""
    if roi is None or roi.size == 0:
        return 0.0

    intensity = extract_intensity_stats(roi)
    glcm = extract_glcm_features(roi)
    fd = compute_fractal_dimension(roi)
    mean_intensity = intensity[0] / 255.0
    homogeneity = glcm[2]
    energy = glcm[3]
    correlation = max(glcm[4], 0.0)
    contrast_penalty = 0.05 * glcm[0] + 0.05 * glcm[1]
    fractal_bonus = 0.15 * np.clip(fd - 1.0, 0.0, 1.0)
    return float(mean_intensity + homogeneity + energy + correlation + fractal_bonus - contrast_penalty)


def build_sclerosis_features(
    medial_roi: Optional[np.ndarray],
    lateral_roi: Optional[np.ndarray],
    calibrator: Optional[QuantileCalibrator | Mapping[str, QuantileCalibrator]] = None,
) -> Dict[str, float]:
    """Create the 18-dim sclerosis feature dict from two ROI patches."""
    score_med = sclerosis_roi_score(medial_roi)
    score_lat = sclerosis_roi_score(lateral_roi)

    if calibrator is None:
        grade_med = min(2, int(round(score_med * 2.0)))
        grade_lat = min(2, int(round(score_lat * 2.0)))
    elif isinstance(calibrator, Mapping):
        med_calibrator = calibrator.get("medial")
        lat_calibrator = calibrator.get("lateral")
        grade_med = med_calibrator.grade(score_med) if med_calibrator is not None else min(2, int(round(score_med * 2.0)))
        grade_lat = lat_calibrator.grade(score_lat) if lat_calibrator is not None else min(2, int(round(score_lat * 2.0)))
    else:
        grade_med = calibrator.grade(score_med)
        grade_lat = calibrator.grade(score_lat)

    return {
        "scl_grade_medial": int(grade_med),
        "scl_grade_lateral": int(grade_lat),
        **_single_side_sclerosis_features("medial", medial_roi),
        **_single_side_sclerosis_features("lateral", lateral_roi),
    }


def fit_osteophyte_calibrator(rois: Iterable[Tuple[str, np.ndarray]]) -> QuantileCalibrator:
    """Fit an unsupervised grade calibrator for osteophyte proxy scores."""
    scores = [osteophyte_roi_score(roi, site) for site, roi in rois if roi is not None]
    return QuantileCalibrator.from_scores(
        scores,
        quantiles=(0.55, 0.78, 0.93),
        default_thresholds=(0.10, 0.18, 0.26),
    )


def fit_osteophyte_calibrators_from_reviewed_scores(
    labeled_scores_by_site: Mapping[str, Iterable[Tuple[float, int]]],
) -> Dict[str, QuantileCalibrator]:
    calibrators: Dict[str, QuantileCalibrator] = {}
    for site in ROI_SITES:
        labeled_scores = list(labeled_scores_by_site.get(site, []))
        scores = [float(score) for score, _ in labeled_scores]
        labels = [int(label) for _, label in labeled_scores]
        calibrators[site] = QuantileCalibrator.from_labeled_scores(
            scores,
            labels,
            default_thresholds=(0.10, 0.18, 0.26),
        )
    return calibrators


def fit_sclerosis_calibrator(rois: Iterable[np.ndarray]) -> QuantileCalibrator:
    """Fit an unsupervised grade calibrator for sclerosis proxy scores."""
    scores = [sclerosis_roi_score(roi) for roi in rois if roi is not None]
    return QuantileCalibrator.from_scores(
        scores,
        quantiles=(0.65, 0.90),
        default_thresholds=(1.2, 1.6),
    )


def fit_sclerosis_calibrators_from_reviewed_scores(
    labeled_scores_by_side: Mapping[str, Iterable[Tuple[float, int]]],
) -> Dict[str, QuantileCalibrator]:
    calibrators: Dict[str, QuantileCalibrator] = {}
    for side in ("medial", "lateral"):
        labeled_scores = list(labeled_scores_by_side.get(side, []))
        scores = [float(score) for score, _ in labeled_scores]
        labels = [int(label) for _, label in labeled_scores]
        calibrators[side] = QuantileCalibrator.from_labeled_scores(
            scores,
            labels,
            default_thresholds=(1.2, 1.6),
        )
    return calibrators


def _single_side_sclerosis_features(side: str, roi: Optional[np.ndarray]) -> Dict[str, float]:
    prefix = "med" if side == "medial" else "lat"
    if roi is None or roi.size == 0:
        return {
            f"scl_intensity_{side}": 0.0,
            f"scl_fractal_dim_{prefix}": 1.0,
            f"scl_glcm_contrast_{prefix}": 0.0,
            f"scl_glcm_dissimilarity_{prefix}": 0.0,
            f"scl_glcm_homogeneity_{prefix}": 0.0,
            f"scl_glcm_energy_{prefix}": 0.0,
            f"scl_glcm_correlation_{prefix}": 0.0,
            f"scl_lbp_entropy_{prefix}": 0.0,
        }

    intensity_stats = extract_intensity_stats(roi)
    glcm = extract_glcm_features(roi)
    fd = compute_fractal_dimension(roi)
    lbp = extract_lbp_features(roi)
    lbp_entropy = float(-np.sum(lbp[lbp > 0] * np.log2(lbp[lbp > 0] + 1e-10)))
    return {
        f"scl_intensity_{side}": float(intensity_stats[0]),
        f"scl_fractal_dim_{prefix}": float(fd),
        f"scl_glcm_contrast_{prefix}": float(glcm[0]),
        f"scl_glcm_dissimilarity_{prefix}": float(glcm[1]),
        f"scl_glcm_homogeneity_{prefix}": float(glcm[2]),
        f"scl_glcm_energy_{prefix}": float(glcm[3]),
        f"scl_glcm_correlation_{prefix}": float(glcm[4]),
        f"scl_lbp_entropy_{prefix}": lbp_entropy,
    }


def _preprocess_roi(roi: np.ndarray) -> np.ndarray:
    roi_uint8 = roi.astype(np.uint8) if roi.dtype != np.uint8 else roi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(roi_uint8)
