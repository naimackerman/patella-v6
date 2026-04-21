"""KNEEL landmark backends for ROI extraction and JSN estimation.

This module supports:
- ``heuristic``: a fully local landmark bootstrap fallback
- ``kneel_repo``: an adapter that calls the real ``imedslab/KNEEL`` API

The repo-backed path expects the KNEEL microservice to be running locally or on
the configured endpoint. Because KNEEL expects bilateral DICOM input, this
module wraps a single PNG crop into a synthetic bilateral DICOM before sending
the request.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CompartmentLandmarks:
    x_range: Tuple[int, int]
    femoral: np.ndarray
    tibial: np.ndarray
    synthetic: bool = False
    valid_columns: int = 0


@dataclass
class KneeLandmarks:
    joint_y: float
    lateral: CompartmentLandmarks
    medial: CompartmentLandmarks
    backend: str = "heuristic"
    low_confidence: bool = False
    range_fallback_used: bool = False


class KNEELLandmarkDetector:
    """Landmark detector with heuristic and repo-backed backends."""

    def __init__(
        self,
        backend: str = "heuristic",
        weights_path: Optional[str] = None,
        api_url: Optional[str] = None,
        request_timeout_s: float = 45.0,
        default_spacing_mm: float = 0.2,
        bilateral_padding_px: int = 0,
        allow_backend_fallback: bool = True,
    ):
        self.backend = backend
        self.weights_path = weights_path
        self.allow_backend_fallback = allow_backend_fallback
        self._repo_client = None
        if backend == "kneel_repo":
            self._repo_client = _KNEELRepoClient(
                api_url=api_url or "http://127.0.0.1:5000/kneel/predict/bilateral",
                request_timeout_s=request_timeout_s,
                default_spacing_mm=default_spacing_mm,
                bilateral_padding_px=bilateral_padding_px,
            )
        elif backend != "heuristic":
            raise ValueError(f"Unsupported KNEEL backend: {backend}")

    @classmethod
    def from_preprocessing_cfg(cls, preprocessing_cfg):
        backend = getattr(preprocessing_cfg, "landmark_backend", "heuristic")
        kneel_cfg = getattr(preprocessing_cfg, "kneel_repo", {})
        return cls(
            backend=backend,
            api_url=getattr(kneel_cfg, "api_url", None),
            request_timeout_s=float(getattr(kneel_cfg, "request_timeout_s", 45.0)),
            default_spacing_mm=float(getattr(kneel_cfg, "default_spacing_mm", 0.2)),
            bilateral_padding_px=int(getattr(kneel_cfg, "bilateral_padding_px", 0)),
            allow_backend_fallback=bool(getattr(preprocessing_cfg, "allow_landmark_backend_fallback", True)),
        )

    def predict(
        self,
        image: np.ndarray,
        is_left: bool = False,
        apply_preprocessing: bool = True,
        require_reliable: bool = False,
    ) -> KneeLandmarks:
        if self.backend == "heuristic":
            landmarks = detect_knee_landmarks(image, is_left=is_left, apply_preprocessing=apply_preprocessing)
            if require_reliable and landmarks.low_confidence:
                raise ValueError("heuristic_landmarks_low_confidence")
            return landmarks

        try:
            points = self._repo_client.predict_png(image, is_left=is_left)
            landmarks = kneel_points_to_landmarks(points, image.shape[:2], is_left=is_left, backend=self.backend)
            if require_reliable and landmarks.low_confidence:
                raise ValueError("repo_landmarks_low_confidence")
            return landmarks
        except Exception:
            if not self.allow_backend_fallback:
                raise
            landmarks = detect_knee_landmarks(image, is_left=is_left, apply_preprocessing=apply_preprocessing)
            if require_reliable and landmarks.low_confidence:
                raise ValueError("heuristic_landmarks_low_confidence")
            return landmarks


class _KNEELRepoClient:
    """Adapter for the dockerized imedslab/KNEEL microservice."""

    def __init__(
        self,
        api_url: str,
        request_timeout_s: float,
        default_spacing_mm: float,
        bilateral_padding_px: int = 0,
    ):
        self.api_url = api_url
        self.request_timeout_s = request_timeout_s
        self.default_spacing_mm = default_spacing_mm
        self.bilateral_padding_px = bilateral_padding_px

    def predict_png(self, image: np.ndarray, is_left: bool = False) -> np.ndarray:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("requests is required for preprocessing.landmark_backend=kneel_repo") from exc

        dicom_bytes = _png_to_bilateral_dicom_bytes(
            image=image,
            is_left=is_left,
            spacing_mm=self.default_spacing_mm,
            bilateral_padding_px=self.bilateral_padding_px,
        )
        payload = {"dicom": base64.b64encode(dicom_bytes).decode("ascii")}
        response = requests.post(self.api_url, json=payload, timeout=self.request_timeout_s)
        response.raise_for_status()

        result = response.json()
        side_key = "L" if is_left else "R"
        points = np.asarray(result.get(side_key), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"KNEEL API returned invalid landmark shape for side {side_key}: {points!r}")

        width = image.shape[1] + self.bilateral_padding_px
        if is_left:
            points[:, 0] -= width
        return points


def detect_knee_landmarks(
    image: np.ndarray,
    is_left: bool = False,
    apply_preprocessing: bool = True,
) -> KneeLandmarks:
    """Estimate joint landmarks from image appearance."""
    proc = _preprocess_image(image, apply_clahe=apply_preprocessing)
    joint_y = _estimate_joint_row(proc)

    lateral_range, medial_range, range_fallback_used = _estimate_compartment_ranges(proc, joint_y, is_left=is_left)

    lateral = _estimate_compartment(proc, lateral_range, joint_y)
    medial = _estimate_compartment(proc, medial_range, joint_y)
    low_confidence = (
        range_fallback_used
        or lateral.synthetic
        or medial.synthetic
        or lateral.valid_columns < 12
        or medial.valid_columns < 12
    )
    return KneeLandmarks(
        joint_y=joint_y,
        lateral=lateral,
        medial=medial,
        low_confidence=low_confidence,
        range_fallback_used=range_fallback_used,
    )


def kneel_points_to_landmarks(
    points: np.ndarray,
    image_shape: Tuple[int, int],
    is_left: bool = False,
    backend: str = "kneel_repo",
) -> KneeLandmarks:
    """Convert raw KNEEL point predictions into the internal landmark structure."""
    valid = np.asarray(points, dtype=np.float64)
    if valid.ndim != 2 or valid.shape[1] != 2:
        raise ValueError("KNEEL point array must have shape (N, 2).")
    valid = valid[np.isfinite(valid).all(axis=1)]
    if len(valid) < 8:
        raise ValueError("KNEEL prediction returned too few valid landmarks.")

    h, w = image_shape[:2]
    valid[:, 0] = np.clip(valid[:, 0], 0, w - 1)
    valid[:, 1] = np.clip(valid[:, 1], 0, h - 1)
    valid = valid[np.argsort(valid[:, 0])]

    mid_x = float(np.median(valid[:, 0]))
    left_pts = valid[valid[:, 0] <= mid_x]
    right_pts = valid[valid[:, 0] > mid_x]
    if len(left_pts) < 4 or len(right_pts) < 4:
        half = len(valid) // 2
        left_pts = valid[:half]
        right_pts = valid[half:]

    if is_left:
        medial_pts, lateral_pts = left_pts, right_pts
    else:
        lateral_pts, medial_pts = left_pts, right_pts

    lateral = _compartment_from_kneel_points(lateral_pts, image_shape)
    medial = _compartment_from_kneel_points(medial_pts, image_shape)
    return KneeLandmarks(
        joint_y=float(np.median(valid[:, 1])),
        lateral=lateral,
        medial=medial,
        backend=backend,
    )


def landmarks_from_jsn_mask(mask: np.ndarray) -> KneeLandmarks:
    """Build compartment landmarks from a JSN segmentation mask.

    The JSN mask already encodes anatomical medial/lateral compartments:
    class 1 = medial joint space, class 2 = lateral joint space.
    """
    from src.features.jsw_computation import extract_contours

    medial_femoral, medial_tibial = extract_contours(mask, class_id=1)
    lateral_femoral, lateral_tibial = extract_contours(mask, class_id=2)
    if medial_femoral is None or medial_tibial is None:
        raise ValueError("jsn_mask_missing_medial_contours")
    if lateral_femoral is None or lateral_tibial is None:
        raise ValueError("jsn_mask_missing_lateral_contours")

    medial = CompartmentLandmarks(
        x_range=(
            int(max(0, np.floor(min(medial_femoral[:, 0].min(), medial_tibial[:, 0].min())))),
            int(np.ceil(max(medial_femoral[:, 0].max(), medial_tibial[:, 0].max()))),
        ),
        femoral=medial_femoral.astype(np.float64),
        tibial=medial_tibial.astype(np.float64),
        synthetic=False,
        valid_columns=min(len(medial_femoral), len(medial_tibial)),
    )
    lateral = CompartmentLandmarks(
        x_range=(
            int(max(0, np.floor(min(lateral_femoral[:, 0].min(), lateral_tibial[:, 0].min())))),
            int(np.ceil(max(lateral_femoral[:, 0].max(), lateral_tibial[:, 0].max()))),
        ),
        femoral=lateral_femoral.astype(np.float64),
        tibial=lateral_tibial.astype(np.float64),
        synthetic=False,
        valid_columns=min(len(lateral_femoral), len(lateral_tibial)),
    )
    joint_y = float(np.median(np.concatenate([
        medial.femoral[:, 1],
        medial.tibial[:, 1],
        lateral.femoral[:, 1],
        lateral.tibial[:, 1],
    ])))
    low_confidence = (
        medial.valid_columns < 10
        or lateral.valid_columns < 10
    )
    return KneeLandmarks(
        joint_y=joint_y,
        lateral=lateral,
        medial=medial,
        backend="jsn_mask",
        low_confidence=low_confidence,
        range_fallback_used=False,
    )


def estimate_jsn_features_from_landmarks(
    landmarks: KneeLandmarks,
    reference_mjsw_medial: float = 15.0,
    reference_mjsw_lateral: float = 15.0,
    n_points: int = 16,
) -> Dict[str, object]:
    """Compute the standard 22-dim JSN feature dict from landmark surfaces."""
    half_points = max(1, n_points // 2)
    lateral_profile = _sample_profile(landmarks.lateral.femoral, landmarks.lateral.tibial, half_points)
    medial_profile = _sample_profile(landmarks.medial.femoral, landmarks.medial.tibial, half_points)
    jsw_profile = np.concatenate([lateral_profile, medial_profile]).astype(np.float64)

    mjsw_lateral = float(lateral_profile.min()) if lateral_profile.size else 0.0
    mjsw_medial = float(medial_profile.min()) if medial_profile.size else 0.0

    jsn_rate_medial = _compute_jsn_rate(mjsw_medial, reference_mjsw_medial)
    jsn_rate_lateral = _compute_jsn_rate(mjsw_lateral, reference_mjsw_lateral)
    jsw_ratio = mjsw_medial / max(mjsw_lateral, 1e-6)
    jsw_asymmetry = abs(mjsw_medial - mjsw_lateral)

    return {
        "mJSW_medial": mjsw_medial,
        "mJSW_lateral": mjsw_lateral,
        "jsw_profile": jsw_profile,
        "jsn_rate_medial": jsn_rate_medial,
        "jsn_rate_lateral": jsn_rate_lateral,
        "jsw_ratio": jsw_ratio,
        "jsw_asymmetry": jsw_asymmetry,
    }


def extract_kneel_rois(
    image: np.ndarray,
    landmarks: KneeLandmarks,
    osteophyte_roi_size: int = 140,
) -> Dict[str, np.ndarray]:
    """Extract joint-space and osteophyte ROIs from landmark anchors."""
    boxes = compute_kneel_roi_boxes(image.shape[:2], landmarks)
    rois = {}
    for name, (x1, y1, x2, y2) in boxes.items():
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((16, 16), dtype=np.uint8)
        if name != "joint_space":
            crop = cv2.resize(crop, (osteophyte_roi_size, osteophyte_roi_size), interpolation=cv2.INTER_LINEAR)
        rois[name] = crop
    return rois


def compute_kneel_roi_boxes(
    image_shape: Tuple[int, int],
    landmarks: KneeLandmarks,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Compute ROI bounding boxes from landmark geometry.

    Osteophytes form at the marginal bone edge near the joint line, so the
    site ROIs are intentionally edge-anchored and vertically asymmetric rather
    than small centered squares. That keeps the lateral/medial cortical margin
    inside the crop while preserving more intra-bone context.
    """
    h, w = image_shape[:2]
    base_roi_width = max(int(0.42 * min(h, w)), 80)
    base_roi_height = max(int(0.38 * min(h, w)), 80)

    site_params = {
        "lateral_femur": {
            "band_fraction": 0.24,
            "width_scale": 1.12,
            "height_scale": 1.10,
            "outward_frac": 0.28,
            "upper_frac": 0.78,
            "lower_frac": 0.22,
        },
        "lateral_tibia": {
            "band_fraction": 0.18,
            "width_scale": 1.00,
            "height_scale": 1.00,
            "outward_frac": 0.22,
            "upper_frac": 0.28,
            "lower_frac": 0.72,
        },
        "medial_femur": {
            "band_fraction": 0.18,
            "width_scale": 1.00,
            "height_scale": 1.00,
            "outward_frac": 0.22,
            "upper_frac": 0.72,
            "lower_frac": 0.28,
        },
        "medial_tibia": {
            "band_fraction": 0.24,
            "width_scale": 1.12,
            "height_scale": 1.10,
            "outward_frac": 0.28,
            "upper_frac": 0.22,
            "lower_frac": 0.78,
        },
    }

    lat_anchor_x, lat_band = _edge_band_anchor(
        landmarks.lateral.x_range,
        w,
        band_fraction=site_params["lateral_femur"]["band_fraction"],
    )
    med_anchor_x, med_band = _edge_band_anchor(
        landmarks.medial.x_range,
        w,
        band_fraction=site_params["medial_femur"]["band_fraction"],
    )
    lat_fem_y = _surface_y_in_band(landmarks.lateral.femoral, lat_band)
    lat_tib_y = _surface_y_in_band(landmarks.lateral.tibial, lat_band)
    med_fem_y = _surface_y_in_band(landmarks.medial.femoral, med_band)
    med_tib_y = _surface_y_in_band(landmarks.medial.tibial, med_band)

    boxes = {
        "lateral_femur": _edge_anchored_box(
            outer_x=lat_anchor_x,
            anchor_y=lat_fem_y,
            box_width=max(80, int(round(base_roi_width * site_params["lateral_femur"]["width_scale"]))),
            box_height=max(80, int(round(base_roi_height * site_params["lateral_femur"]["height_scale"]))),
            image_shape=image_shape,
            outward_frac=site_params["lateral_femur"]["outward_frac"],
            upper_frac=site_params["lateral_femur"]["upper_frac"],
            lower_frac=site_params["lateral_femur"]["lower_frac"],
        ),
        "lateral_tibia": _edge_anchored_box(
            outer_x=lat_anchor_x,
            anchor_y=lat_tib_y,
            box_width=max(80, int(round(base_roi_width * site_params["lateral_tibia"]["width_scale"]))),
            box_height=max(80, int(round(base_roi_height * site_params["lateral_tibia"]["height_scale"]))),
            image_shape=image_shape,
            outward_frac=site_params["lateral_tibia"]["outward_frac"],
            upper_frac=site_params["lateral_tibia"]["upper_frac"],
            lower_frac=site_params["lateral_tibia"]["lower_frac"],
        ),
        "medial_femur": _edge_anchored_box(
            outer_x=med_anchor_x,
            anchor_y=med_fem_y,
            box_width=max(80, int(round(base_roi_width * site_params["medial_femur"]["width_scale"]))),
            box_height=max(80, int(round(base_roi_height * site_params["medial_femur"]["height_scale"]))),
            image_shape=image_shape,
            outward_frac=site_params["medial_femur"]["outward_frac"],
            upper_frac=site_params["medial_femur"]["upper_frac"],
            lower_frac=site_params["medial_femur"]["lower_frac"],
        ),
        "medial_tibia": _edge_anchored_box(
            outer_x=med_anchor_x,
            anchor_y=med_tib_y,
            box_width=max(80, int(round(base_roi_width * site_params["medial_tibia"]["width_scale"]))),
            box_height=max(80, int(round(base_roi_height * site_params["medial_tibia"]["height_scale"]))),
            image_shape=image_shape,
            outward_frac=site_params["medial_tibia"]["outward_frac"],
            upper_frac=site_params["medial_tibia"]["upper_frac"],
            lower_frac=site_params["medial_tibia"]["lower_frac"],
        ),
    }

    x0 = max(0, min(landmarks.lateral.x_range[0], landmarks.medial.x_range[0]) - int(0.04 * w))
    x1 = min(w, max(landmarks.lateral.x_range[1], landmarks.medial.x_range[1]) + int(0.04 * w))
    y_top = max(
        0,
        int(round(min(landmarks.lateral.femoral[:, 1].min(), landmarks.medial.femoral[:, 1].min()) - 0.08 * h)),
    )
    y_bottom = min(
        h,
        int(round(max(landmarks.lateral.tibial[:, 1].max(), landmarks.medial.tibial[:, 1].max()) + 0.08 * h)),
    )
    boxes["joint_space"] = (x0, y_top, max(x0 + 1, x1), max(y_top + 1, y_bottom))
    return boxes


def extract_kneel_subchondral_rois(
    image: np.ndarray,
    landmarks: KneeLandmarks,
    output_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    medial, lateral, _, _ = extract_kneel_subchondral_rois_with_boxes(
        image=image,
        landmarks=landmarks,
        output_size=output_size,
    )
    return medial, lateral


def extract_kneel_subchondral_rois_with_boxes(
    image: np.ndarray,
    landmarks: KneeLandmarks,
    output_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Extract medial/lateral subchondral ROIs directly from landmark surfaces."""
    medial, medial_box = _crop_subchondral(image, landmarks.medial, output_size=output_size)
    lateral, lateral_box = _crop_subchondral(image, landmarks.lateral, output_size=output_size)
    return medial, lateral, medial_box, lateral_box
    return medial, lateral


def _preprocess_image(image: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
    image_uint8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
    proc = image_uint8
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        proc = clahe.apply(proc)
    return cv2.GaussianBlur(proc, (5, 5), 0)


def _estimate_compartment_ranges(
    image: np.ndarray,
    joint_y: float,
    is_left: bool,
) -> Tuple[Tuple[int, int], Tuple[int, int], bool]:
    """Estimate compartment x-ranges from the full joint-space band.

    Returns `(lateral_range, medial_range, range_fallback_used)`.
    """
    h, w = image.shape[:2]
    band_h = max(int(0.08 * h), 10)
    y0 = max(0, int(round(joint_y)) - band_h)
    y1 = min(h, int(round(joint_y)) + band_h)
    band = image[y0:y1, :]

    left_default = _estimate_half_range(image, 0, w // 2, joint_y)
    right_default = _estimate_half_range(image, w // 2, w, joint_y)
    if band.size == 0:
        if is_left:
            return right_default, left_default, True
        return left_default, right_default, True

    col_mean = cv2.GaussianBlur(band.mean(axis=0).astype(np.float32).reshape(1, -1), (1, 9), 0).ravel()
    threshold = np.percentile(col_mean, 55)
    dark = col_mean <= threshold
    min_width = max(int(0.08 * w), 12)
    segments = [
        seg for seg in _contiguous_segments(dark)
        if (seg[1] - seg[0] + 1) >= min_width
    ]

    mid_x = 0.5 * w
    left_candidates = [seg for seg in segments if 0.5 * (seg[0] + seg[1]) < mid_x]
    right_candidates = [seg for seg in segments if 0.5 * (seg[0] + seg[1]) >= mid_x]

    range_fallback_used = False
    if left_candidates:
        left_range = max(
            left_candidates,
            key=lambda seg: (seg[1] - seg[0], -abs(0.5 * (seg[0] + seg[1]) - 0.25 * w)),
        )
    else:
        left_range = left_default
        range_fallback_used = True

    if right_candidates:
        right_range = max(
            right_candidates,
            key=lambda seg: (seg[1] - seg[0], -abs(0.5 * (seg[0] + seg[1]) - 0.75 * w)),
        )
    else:
        right_range = right_default
        range_fallback_used = True

    left_range = _expand_range(left_range, w)
    right_range = _expand_range(right_range, w)

    if is_left:
        return right_range, left_range, range_fallback_used
    return left_range, right_range, range_fallback_used


def _estimate_joint_row(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    x0 = int(0.10 * w)
    x1 = int(0.90 * w)
    row_mean = image[:, x0:x1].mean(axis=1)
    y0 = int(0.28 * h)
    y1 = int(0.72 * h)
    return float(np.argmin(row_mean[y0:y1]) + y0)


def _estimate_half_range(image: np.ndarray, x0: int, x1: int, joint_y: float) -> Tuple[int, int]:
    h = image.shape[0]
    band_h = max(int(0.08 * h), 10)
    y0 = max(0, int(round(joint_y)) - band_h)
    y1 = min(h, int(round(joint_y)) + band_h)
    band = image[y0:y1, x0:x1]
    if band.size == 0:
        return x0, x1 - 1

    col_mean = cv2.GaussianBlur(band.mean(axis=0).astype(np.float32).reshape(1, -1), (1, 9), 0).ravel()
    threshold = np.percentile(col_mean, 55)
    dark = col_mean <= threshold
    segments = _contiguous_segments(dark)
    if not segments:
        seg_start = int(0.20 * (x1 - x0))
        seg_end = int(0.80 * (x1 - x0))
    else:
        center = 0.5 * (x1 - x0)
        seg_start, seg_end = max(
            segments,
            key=lambda seg: (seg[1] - seg[0], -abs(0.5 * (seg[0] + seg[1]) - center)),
        )
    abs_start = x0 + seg_start
    abs_end = min(x1 - 1, x0 + seg_end)
    if abs_end <= abs_start + 4:
        abs_start = x0 + int(0.18 * (x1 - x0))
        abs_end = x0 + int(0.82 * (x1 - x0))
    return int(abs_start), int(abs_end)


def _estimate_compartment(image: np.ndarray, x_range: Tuple[int, int], joint_y: float) -> CompartmentLandmarks:
    h = image.shape[0]
    search_h = max(int(0.22 * h), 24)
    y0 = max(0, int(round(joint_y)) - search_h)
    y1 = min(h, int(round(joint_y)) + search_h)

    xs = []
    femoral_y = []
    tibial_y = []
    for x in range(x_range[0], x_range[1] + 1):
        column = image[y0:y1, x].astype(np.float64)
        if column.size < 8:
            continue
        smooth = cv2.GaussianBlur(column.reshape(-1, 1).astype(np.float32), (1, 9), 0).ravel()
        valley = int(np.argmin(smooth))
        if valley < 2 or valley > len(smooth) - 3:
            continue
        grad = np.gradient(smooth)
        fem_idx = int(np.argmin(grad[:valley])) if valley > 0 else 0
        tib_idx = int(valley + np.argmax(grad[valley:])) if valley < len(grad) else len(grad) - 1
        if tib_idx <= fem_idx + 1:
            continue
        gap = tib_idx - fem_idx
        if gap < 2 or gap > max(int(0.30 * h), 32):
            continue
        xs.append(x)
        femoral_y.append(y0 + fem_idx)
        tibial_y.append(y0 + tib_idx)

    synthetic = False
    if len(xs) < 8:
        xs = list(range(x_range[0], x_range[1] + 1))
        fem_guess = joint_y - max(4, int(0.04 * h))
        tib_guess = joint_y + max(4, int(0.04 * h))
        femoral_y = [fem_guess] * len(xs)
        tibial_y = [tib_guess] * len(xs)
        synthetic = True

    femoral = np.column_stack([xs, _smooth_series(np.asarray(femoral_y, dtype=np.float64))])
    tibial = np.column_stack([xs, _smooth_series(np.asarray(tibial_y, dtype=np.float64))])
    return CompartmentLandmarks(
        x_range=x_range,
        femoral=femoral,
        tibial=tibial,
        synthetic=synthetic,
        valid_columns=len(xs),
    )


def _compartment_from_kneel_points(
    points: np.ndarray,
    image_shape: Tuple[int, int],
) -> CompartmentLandmarks:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        raise ValueError("Need at least 4 KNEEL points for a compartment.")

    y_mid = float(np.median(points[:, 1]))
    femoral = points[points[:, 1] <= y_mid]
    tibial = points[points[:, 1] > y_mid]

    if len(femoral) < 2 or len(tibial) < 2:
        order = np.argsort(points[:, 1])
        half = max(2, len(points) // 2)
        femoral = points[order[:half]]
        tibial = points[order[-half:]]

    femoral = _densify_surface(femoral)
    tibial = _densify_surface(tibial)
    x_min = int(max(0, np.floor(min(femoral[:, 0].min(), tibial[:, 0].min()))))
    x_max = int(min(image_shape[1] - 1, np.ceil(max(femoral[:, 0].max(), tibial[:, 0].max()))))
    return CompartmentLandmarks(
        x_range=(x_min, x_max),
        femoral=femoral,
        tibial=tibial,
        synthetic=False,
        valid_columns=len(points),
    )


def _sample_profile(femoral: np.ndarray, tibial: np.ndarray, n_points: int) -> np.ndarray:
    x_start = max(float(femoral[:, 0].min()), float(tibial[:, 0].min()))
    x_end = min(float(femoral[:, 0].max()), float(tibial[:, 0].max()))
    if x_end <= x_start:
        return np.zeros(n_points, dtype=np.float64)

    sample_x = np.linspace(x_start, x_end, n_points)
    profile = np.zeros(n_points, dtype=np.float64)
    for i, x in enumerate(sample_x):
        fem_y = np.interp(x, femoral[:, 0], femoral[:, 1])
        tib_y = np.interp(x, tibial[:, 0], tibial[:, 1])
        profile[i] = max(0.0, tib_y - fem_y)
    return profile


def _compute_jsn_rate(mjsw: float, reference: float) -> float:
    if reference <= 1e-6:
        return 100.0
    return float(np.clip(100.0 * (1.0 - mjsw / reference), 0.0, 100.0))


def _expand_range(x_range: Tuple[int, int], image_width: int, margin_frac: float = 0.06) -> Tuple[int, int]:
    margin = max(3, int(round(margin_frac * image_width)))
    x0 = max(0, int(x_range[0]) - margin)
    x1 = min(image_width - 1, int(x_range[1]) + margin)
    return x0, max(x0 + 1, x1)


def _edge_band_anchor(
    x_range: Tuple[int, int],
    image_width: int,
    band_fraction: float = 0.18,
) -> Tuple[int, Tuple[int, int]]:
    x0, x1 = map(int, x_range)
    width = max(2, x1 - x0 + 1)
    band_width = max(4, int(round(width * band_fraction)))
    center = 0.5 * (x0 + x1)
    if center < image_width / 2.0:
        band = (x0, min(x1, x0 + band_width))
    else:
        band = (max(x0, x1 - band_width), x1)
    anchor_x = int(round(0.5 * (band[0] + band[1])))
    return anchor_x, band


def _surface_y_in_band(surface: np.ndarray, band: Tuple[int, int]) -> float:
    x0, x1 = band
    mask = (surface[:, 0] >= x0) & (surface[:, 0] <= x1)
    if mask.any():
        return float(np.median(surface[mask, 1]))
    center_x = 0.5 * (x0 + x1)
    return float(np.interp(center_x, surface[:, 0], surface[:, 1]))


def _crop_square(image: np.ndarray, cx: int, cy: int, side: int, out_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    half = side // 2
    x0 = int(np.clip(cx - half, 0, max(0, w - side)))
    y0 = int(np.clip(cy - half, 0, max(0, h - side)))
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        patch = np.zeros((side, side), dtype=np.uint8)
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _square_box(
    cx: int,
    cy: int,
    side: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    h, w = image_shape[:2]
    half = side // 2
    x0 = int(np.clip(cx - half, 0, max(0, w - side)))
    y0 = int(np.clip(cy - half, 0, max(0, h - side)))
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    return x0, y0, x1, y1


def _edge_anchored_box(
    outer_x: int,
    anchor_y: float,
    box_width: int,
    box_height: int,
    image_shape: Tuple[int, int],
    outward_frac: float,
    upper_frac: float,
    lower_frac: float,
) -> Tuple[int, int, int, int]:
    """Create an osteophyte ROI box anchored on the compartment edge.

    The outer cortical margin should remain near one side of the crop, with
    most of the width extending inward toward the joint. Vertically, femoral
    boxes extend more above the joint line and tibial boxes extend more below.
    """
    h, w = image_shape[:2]
    outer_x = int(round(np.clip(outer_x, 0, w - 1)))

    is_left_edge = outer_x < (w / 2.0)
    outward_frac = float(np.clip(outward_frac, 0.05, 0.45))
    inward_frac = 1.0 - outward_frac
    if is_left_edge:
        x0 = int(round(outer_x - outward_frac * box_width))
        x1 = int(round(outer_x + inward_frac * box_width))
    else:
        x0 = int(round(outer_x - inward_frac * box_width))
        x1 = int(round(outer_x + outward_frac * box_width))

    upper_frac = float(np.clip(upper_frac, 0.10, 0.90))
    lower_frac = float(np.clip(lower_frac, 0.10, 0.90))
    frac_sum = upper_frac + lower_frac
    if frac_sum <= 0:
        upper_frac, lower_frac = 0.5, 0.5
    else:
        upper_frac /= frac_sum
        lower_frac /= frac_sum
    y0 = int(round(anchor_y - upper_frac * box_height))
    y1 = int(round(anchor_y + lower_frac * box_height))

    x0 = int(np.clip(x0, 0, max(0, w - box_width)))
    y0 = int(np.clip(y0, 0, max(0, h - box_height)))
    x1 = min(w, x0 + box_width)
    y1 = min(h, y0 + box_height)
    return x0, y0, x1, y1


def _crop_subchondral(
    image: np.ndarray,
    compartment: CompartmentLandmarks,
    output_size: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h = image.shape[0]
    x0, x1 = compartment.x_range
    x0 = max(0, x0)
    x1 = min(image.shape[1], x1)
    tibial_y = compartment.tibial[:, 1]
    y_top = int(np.clip(np.percentile(tibial_y, 40) + max(2, int(0.015 * h)), 0, h - 2))
    depth = max(10, int(0.10 * h))
    y_bottom = min(h, y_top + depth)
    patch = image[y_top:y_bottom, x0:x1]
    if patch.size == 0:
        patch = np.zeros((depth, max(1, x1 - x0)), dtype=np.uint8)
    return cv2.resize(patch, (output_size, output_size), interpolation=cv2.INTER_LINEAR), (x0, y_top, x1, y_bottom)


def _smooth_series(values: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    if values.size < 3:
        return values
    kernel_size = min(kernel_size, values.size if values.size % 2 == 1 else values.size - 1)
    kernel_size = max(kernel_size, 3)
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    padded = np.pad(values, (kernel_size // 2, kernel_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _contiguous_segments(mask: np.ndarray) -> list[Tuple[int, int]]:
    segments = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _densify_surface(points: np.ndarray, n_samples: int = 32) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    order = np.argsort(points[:, 0])
    points = points[order]
    unique_x, unique_idx = np.unique(points[:, 0], return_index=True)
    points = points[unique_idx]
    if len(points) < 2:
        x = points[:, 0]
        y = points[:, 1]
        return np.column_stack([x, y])

    sample_x = np.linspace(points[:, 0].min(), points[:, 0].max(), n_samples)
    sample_y = np.interp(sample_x, points[:, 0], points[:, 1])
    return np.column_stack([sample_x, sample_y])


def _png_to_bilateral_dicom_bytes(
    image: np.ndarray,
    is_left: bool,
    spacing_mm: float,
    bilateral_padding_px: int = 0,
) -> bytes:
    try:
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    except ImportError as exc:
        raise ImportError("pydicom is required for preprocessing.landmark_backend=kneel_repo") from exc

    image_uint16 = _normalize_to_uint16(image)
    h, w = image_uint16.shape[:2]
    half_w = w + bilateral_padding_px
    bilateral = np.zeros((h, half_w * 2), dtype=np.uint16)
    if is_left:
        bilateral[:, half_w:half_w + w] = image_uint16
    else:
        bilateral[:, :w] = image_uint16

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "OT"
    dataset.PatientName = "KNEEL^PNG"
    dataset.PatientID = "KNEELPNG"
    dataset.Rows = bilateral.shape[0]
    dataset.Columns = bilateral.shape[1]
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelSpacing = [float(spacing_mm), float(spacing_mm)]
    dataset.ImagerPixelSpacing = [float(spacing_mm), float(spacing_mm)]
    dataset.PixelData = bilateral.tobytes()

    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, dataset, write_like_original=False)
    return buffer.getvalue()


def _normalize_to_uint16(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("KNEEL DICOM adapter expects a single-channel image.")
    arr = arr.astype(np.float32)
    arr -= arr.min()
    max_val = float(arr.max())
    if max_val > 0:
        arr /= max_val
    return np.round(arr * 65535.0).astype(np.uint16)
