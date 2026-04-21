"""Joint Space Width computation from segmentation masks.

Extracts compartment-specific contours, computes mJSW, JSW profile,
JSN rate, and related features (22 dimensions total).
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist


def extract_contours(
    mask: np.ndarray,
    class_id: int | None = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract femoral and tibial boundary contours from a mask.

    Args:
        mask: (H, W) segmentation mask with classes 0=bg, 1=medial JS, 2=lateral JS.
        class_id: Optional compartment class to isolate. When omitted, both
            compartments are combined into a binary mask.

    Returns:
        Tuple of (femoral_contour, tibial_contour), each (N, 2) arrays of (x, y) points,
        or (None, None) if extraction fails.
    """
    if class_id is None:
        js_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
    else:
        js_mask = (mask == class_id).astype(np.uint8)

    if js_mask.sum() < 5:
        return None, None

    h, w = mask.shape
    femoral_points = []
    tibial_points = []

    for x in range(w):
        col = js_mask[:, x]
        nonzero = np.where(col > 0)[0]
        if len(nonzero) == 0:
            continue
        femoral_points.append([x, nonzero[0]])
        tibial_points.append([x, nonzero[-1]])

    if len(femoral_points) < 2 or len(tibial_points) < 2:
        return None, None

    femoral = np.array(femoral_points, dtype=np.float64)
    tibial = np.array(tibial_points, dtype=np.float64)
    return femoral, tibial


def _smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a 1D signal with an edge-preserving moving average."""
    if window <= 1 or len(values) < 3:
        return values.astype(np.float64, copy=False)

    size = min(int(window), len(values))
    if size % 2 == 0:
        size = max(1, size - 1)
    if size <= 1:
        return values.astype(np.float64, copy=False)

    kernel = np.ones(size, dtype=np.float64) / float(size)
    pad = size // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _resample_contour(contour: np.ndarray, sample_x: np.ndarray, smoothing_window: int) -> np.ndarray:
    """Resample a contour at requested x positions with optional smoothing."""
    order = np.argsort(contour[:, 0])
    x = contour[order, 0].astype(np.float64)
    y = contour[order, 1].astype(np.float64)

    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        sums = np.zeros(len(unique_x), dtype=np.float64)
        counts = np.zeros(len(unique_x), dtype=np.float64)
        np.add.at(sums, inverse, y)
        np.add.at(counts, inverse, 1.0)
        x = unique_x
        y = sums / np.maximum(counts, 1.0)

    y = _smooth_signal(y, smoothing_window)
    return np.interp(sample_x, x, y)


def _trim_count(length: int, trim_fraction: float, min_points: int) -> int:
    """Convert a fractional edge trim into a safe point count."""
    if length <= min_points or trim_fraction <= 0.0:
        return 0
    requested = int(np.floor(length * float(trim_fraction)))
    max_trim = max(0, (int(length) - int(min_points)) // 2)
    return int(min(max(requested, 0), max_trim))


def compute_jsw_profile(
    femoral: np.ndarray,
    tibial: np.ndarray,
    n_points: int,
    smoothing_window: int = 5,
    local_x_window_fraction: float = 0.10,
    endpoint_trim_fraction: float = 0.10,
) -> np.ndarray:
    """Compute JSW using pairwise Euclidean distances at sampled contour points."""
    profile, _ = compute_measurement_pairs_from_contours(
        femoral,
        tibial,
        n_points=n_points,
        smoothing_window=smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )
    return profile


def compute_measurement_pairs_from_contours(
    femoral: np.ndarray,
    tibial: np.ndarray,
    n_points: int,
    smoothing_window: int = 5,
    search_density_factor: int = 4,
    local_x_window_fraction: float = 0.10,
    endpoint_trim_fraction: float = 0.10,
) -> Tuple[np.ndarray, list[tuple[tuple[float, float], tuple[float, float], float]]]:
    """Compute sampled JSW profile and explicit femur-tibia measurement pairs.

    This follows the research-plan intent more closely than vertical-only width:
    femoral sample points are chosen along the mediolateral axis, then each point
    is matched to the nearest tibial boundary point in Euclidean space.

    To reduce false minima from the intercondylar notch and compartment endpoints,
    sampling is restricted to the interior portion of the compartment and tibial
    matching is constrained to a local mediolateral window.
    """
    x_min = max(float(femoral[:, 0].min()), float(tibial[:, 0].min()))
    x_max = min(float(femoral[:, 0].max()), float(tibial[:, 0].max()))
    if x_max <= x_min:
        return np.zeros(n_points, dtype=np.float64), []

    search_points = max(int(n_points) * max(int(search_density_factor), 1), int(n_points))
    dense_x = np.linspace(x_min, x_max, search_points)
    fem_y = _resample_contour(femoral, dense_x, smoothing_window=smoothing_window)
    tib_y = _resample_contour(tibial, dense_x, smoothing_window=smoothing_window)

    femoral_dense = np.column_stack([dense_x, fem_y]).astype(np.float64)
    tibial_dense = np.column_stack([dense_x, tib_y]).astype(np.float64)
    trim = _trim_count(search_points, trim_fraction=endpoint_trim_fraction, min_points=int(n_points))
    sample_start = trim
    sample_end = search_points - trim - 1
    if sample_end < sample_start:
        sample_start = 0
        sample_end = search_points - 1
    sample_idx = np.linspace(sample_start, sample_end, int(n_points)).round().astype(int)
    local_x_window = max(float(x_max - x_min) * float(local_x_window_fraction), 1.0)
    interior_x_min = float(dense_x[sample_start])
    interior_x_max = float(dense_x[sample_end])
    tibial_interior = tibial_dense[
        (tibial_dense[:, 0] >= interior_x_min - 1e-6)
        & (tibial_dense[:, 0] <= interior_x_max + 1e-6)
    ]
    if len(tibial_interior) == 0:
        tibial_interior = tibial_dense

    profile = []
    pairs = []
    for idx in sample_idx:
        p1 = femoral_dense[idx]
        valid_tibial = tibial_interior[tibial_interior[:, 1] >= p1[1] - 1e-6]
        local_tibial = valid_tibial[np.abs(valid_tibial[:, 0] - p1[0]) <= local_x_window]
        if len(local_tibial) > 0:
            tibial_candidates = local_tibial
        elif len(valid_tibial) > 0:
            tibial_candidates = valid_tibial
        else:
            tibial_candidates = tibial_interior
        distances = cdist(p1[None, :], tibial_candidates, metric="euclidean")[0]
        nearest_idx = int(np.argmin(distances))
        p2 = tibial_candidates[nearest_idx]
        dist = float(max(0.0, distances[nearest_idx]))
        profile.append(dist)
        pairs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), dist))

    return np.asarray(profile, dtype=np.float64), pairs


def compute_compartment_profile(
    mask: np.ndarray,
    class_id: int,
    n_points: int,
    smoothing_window: int = 5,
    local_x_window_fraction: float = 0.10,
    endpoint_trim_fraction: float = 0.10,
) -> np.ndarray:
    """Compute a JSW profile for a single compartment class."""
    femoral, tibial = extract_contours(mask, class_id=class_id)
    if femoral is None or tibial is None:
        return np.zeros(n_points, dtype=np.float64)
    return compute_jsw_profile(
        femoral,
        tibial,
        n_points=n_points,
        smoothing_window=smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )


def compute_compartment_measurements(
    mask: np.ndarray,
    class_id: int,
    n_points: int,
    smoothing_window: int = 5,
    local_x_window_fraction: float = 0.10,
    endpoint_trim_fraction: float = 0.10,
) -> Tuple[np.ndarray, list[tuple[tuple[float, float], tuple[float, float], float]]]:
    """Compute profile and explicit measurement pairs for one compartment."""
    femoral, tibial = extract_contours(mask, class_id=class_id)
    if femoral is None or tibial is None:
        return np.zeros(n_points, dtype=np.float64), []
    return compute_measurement_pairs_from_contours(
        femoral,
        tibial,
        n_points=n_points,
        smoothing_window=smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )


def compute_mjsw(jsw_profile: np.ndarray, profile_smoothing_window: int = 3) -> Tuple[float, float]:
    """Compute minimum JSW for medial and lateral compartments.

    Anatomical convention in the flattened profile:
    - first half = lateral compartment
    - second half = medial compartment
    """
    n = len(jsw_profile)
    mid = n // 2

    lateral_profile = _smooth_signal(jsw_profile[:mid], profile_smoothing_window)
    medial_profile = _smooth_signal(jsw_profile[mid:], profile_smoothing_window)

    mjsw_lateral = float(lateral_profile.min()) if len(lateral_profile) > 0 else 0.0
    mjsw_medial = float(medial_profile.min()) if len(medial_profile) > 0 else 0.0
    return mjsw_medial, mjsw_lateral


def compute_jsn_rate(mjsw: float, reference_mjsw: float) -> float:
    """Compute JSN rate as percentage relative to normal reference."""
    if reference_mjsw < 1e-6:
        return 100.0
    rate = 100.0 * (1.0 - mjsw / reference_mjsw)
    return float(np.clip(rate, 0.0, 100.0))


def compute_all_jsn_features(
    mask: np.ndarray,
    reference_mjsw_medial: float = 15.0,
    reference_mjsw_lateral: float = 15.0,
    n_points: int = 16,
    internal_profile_points_per_compartment: int = 64,
    contour_smoothing_window: int = 5,
    local_x_window_fraction: float = 0.10,
    endpoint_trim_fraction: float = 0.10,
    mjsw_profile_smoothing_window: int = 3,
) -> Dict[str, object]:
    """Compute all 22 JSN feature dimensions from a segmentation mask.

    The profile is stored in anatomical order:
    - first 8 values = lateral compartment
    - last 8 values = medial compartment
    """
    half = max(1, n_points // 2)
    dense_half = max(half, int(internal_profile_points_per_compartment))

    lateral_profile = compute_compartment_profile(
        mask,
        class_id=2,
        n_points=half,
        smoothing_window=contour_smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )
    medial_profile = compute_compartment_profile(
        mask,
        class_id=1,
        n_points=half,
        smoothing_window=contour_smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )
    jsw_profile = np.concatenate([lateral_profile, medial_profile]).astype(np.float64)

    dense_lateral_profile = compute_compartment_profile(
        mask,
        class_id=2,
        n_points=dense_half,
        smoothing_window=contour_smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )
    dense_medial_profile = compute_compartment_profile(
        mask,
        class_id=1,
        n_points=dense_half,
        smoothing_window=contour_smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )
    dense_profile = np.concatenate([dense_lateral_profile, dense_medial_profile]).astype(np.float64)

    mjsw_medial, mjsw_lateral = compute_mjsw(
        dense_profile,
        profile_smoothing_window=mjsw_profile_smoothing_window,
    )
    jsn_rate_medial = compute_jsn_rate(mjsw_medial, reference_mjsw_medial)
    jsn_rate_lateral = compute_jsn_rate(mjsw_lateral, reference_mjsw_lateral)

    jsw_ratio = mjsw_medial / (mjsw_lateral + 1e-6)
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


def compute_jsn_measurements(
    mask: np.ndarray,
    reference_mjsw_medial: float = 15.0,
    reference_mjsw_lateral: float = 15.0,
    n_points: int = 16,
    internal_profile_points_per_compartment: int = 64,
    contour_smoothing_window: int = 5,
    local_x_window_fraction: float = 0.10,
    endpoint_trim_fraction: float = 0.10,
    mjsw_profile_smoothing_window: int = 3,
) -> Dict[str, object]:
    """Compute JSN features together with explicit measurement pairs.

    The exported `jsw_profile` remains 16-point by default for plan alignment,
    while `mJSW` is computed from denser internal sampling.
    """
    features = compute_all_jsn_features(
        mask,
        reference_mjsw_medial=reference_mjsw_medial,
        reference_mjsw_lateral=reference_mjsw_lateral,
        n_points=n_points,
        internal_profile_points_per_compartment=internal_profile_points_per_compartment,
        contour_smoothing_window=contour_smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
        mjsw_profile_smoothing_window=mjsw_profile_smoothing_window,
    )
    half = max(1, n_points // 2)
    lateral_profile, lateral_pairs = compute_compartment_measurements(
        mask,
        class_id=2,
        n_points=half,
        smoothing_window=contour_smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )
    medial_profile, medial_pairs = compute_compartment_measurements(
        mask,
        class_id=1,
        n_points=half,
        smoothing_window=contour_smoothing_window,
        local_x_window_fraction=local_x_window_fraction,
        endpoint_trim_fraction=endpoint_trim_fraction,
    )
    measurements = dict(features)
    measurements["jsw_profile"] = np.concatenate([lateral_profile, medial_profile]).astype(np.float64)
    measurements["measurement_pairs"] = lateral_pairs + medial_pairs
    measurements["measurement_pairs_lateral"] = lateral_pairs
    measurements["measurement_pairs_medial"] = medial_pairs
    return measurements


def get_jsn_measurement_kwargs(cfg) -> Dict[str, int]:
    """Resolve JSN measurement parameters from config with safe defaults."""
    preprocessing = getattr(cfg, "preprocessing", None)
    jsn_cfg = getattr(preprocessing, "jsn_measurement", None)
    return {
        "internal_profile_points_per_compartment": int(
            getattr(jsn_cfg, "internal_profile_points_per_compartment", 64)
        ),
        "contour_smoothing_window": int(getattr(jsn_cfg, "contour_smoothing_window", 5)),
        "local_x_window_fraction": float(getattr(jsn_cfg, "local_x_window_fraction", 0.10)),
        "endpoint_trim_fraction": float(getattr(jsn_cfg, "endpoint_trim_fraction", 0.10)),
        "mjsw_profile_smoothing_window": int(getattr(jsn_cfg, "mjsw_profile_smoothing_window", 3)),
    }


def jsn_features_to_vector(features: Dict) -> np.ndarray:
    """Flatten JSN feature dict into a 22-dim numpy vector."""
    vec = [
        features["mJSW_medial"],
        features["mJSW_lateral"],
    ]
    vec.extend(features["jsw_profile"].tolist())
    vec.extend([
        features["jsn_rate_medial"],
        features["jsn_rate_lateral"],
        features["jsw_ratio"],
        features["jsw_asymmetry"],
    ])
    return np.array(vec, dtype=np.float64)
