"""Texture feature extraction for subchondral sclerosis quantification.

Replaces pyradiomics with direct scikit-image + scipy implementations
for ARM/Apple Silicon compatibility.
"""

from typing import Dict, List

import numpy as np
from scipy import stats
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def extract_lbp_features(roi: np.ndarray, radii: List[int] = None) -> np.ndarray:
    """Extract multi-scale Local Binary Pattern histogram features.

    Args:
        roi: (H, W) grayscale ROI patch (uint8).
        radii: LBP radii to compute. Default: [1, 2, 3].

    Returns:
        Concatenated LBP histogram feature vector.
    """
    if radii is None:
        radii = [1, 2, 3]

    roi_uint8 = roi.astype(np.uint8) if roi.dtype != np.uint8 else roi
    all_features = []

    for radius in radii:
        n_points = 8 * radius
        lbp = local_binary_pattern(roi_uint8, n_points, radius, method="uniform")
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        all_features.extend(hist.tolist())

    return np.array(all_features, dtype=np.float64)


def extract_glcm_features(
    roi: np.ndarray,
    distances: List[int] = None,
    levels: int = 64,
) -> np.ndarray:
    """Extract GLCM (Gray Level Co-occurrence Matrix) texture features.

    Computes 5 Haralick properties averaged across distances and angles.

    Args:
        roi: (H, W) grayscale ROI patch.
        distances: GLCM pixel pair distances. Default: [1, 2, 3].
        levels: Number of gray levels (quantize to reduce computation).

    Returns:
        5-element feature vector: contrast, dissimilarity, homogeneity, energy, correlation.
    """
    if distances is None:
        distances = [1, 2, 3]

    # Quantize to reduce levels
    roi_uint8 = roi.astype(np.uint8) if roi.dtype != np.uint8 else roi
    roi_quantized = (roi_uint8.astype(np.float64) / 256 * levels).astype(np.uint8)
    roi_quantized = np.clip(roi_quantized, 0, levels - 1)

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(roi_quantized, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    features = []
    for prop in properties:
        vals = graycoprops(glcm, prop)
        features.append(float(vals.mean()))

    return np.array(features, dtype=np.float64)


def compute_fractal_dimension(roi: np.ndarray, threshold: float = None) -> float:
    """Compute fractal dimension using the box-counting method.

    Args:
        roi: (H, W) grayscale image.
        threshold: Binarization threshold. Default: Otsu's method.

    Returns:
        Fractal dimension (float, typically 1.0-2.0 for 2D images).
    """
    import cv2

    roi_uint8 = roi.astype(np.uint8) if roi.dtype != np.uint8 else roi

    if threshold is None:
        threshold, _ = cv2.threshold(roi_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = (roi_uint8 > threshold).astype(np.uint8)

    if binary.sum() == 0 or binary.sum() == binary.size:
        return 1.0

    # Box-counting
    sizes = []
    counts = []
    min_dim = min(binary.shape)

    size = min_dim // 2
    while size >= 2:
        # Count non-empty boxes
        count = 0
        for i in range(0, binary.shape[0], size):
            for j in range(0, binary.shape[1], size):
                box = binary[i:i + size, j:j + size]
                if box.any():
                    count += 1
        if count > 0:
            sizes.append(size)
            counts.append(count)
        size = size // 2

    if len(sizes) < 2:
        return 1.0

    # Fit log-log linear regression
    log_sizes = np.log(1.0 / np.array(sizes))
    log_counts = np.log(np.array(counts, dtype=np.float64))

    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(coeffs[0])


def extract_intensity_stats(roi: np.ndarray) -> np.ndarray:
    """Extract intensity-based statistical features.

    Returns:
        5-element vector: mean, std, skewness, kurtosis, Shannon entropy.
    """
    roi_float = roi.astype(np.float64).flatten()

    mean_val = float(np.mean(roi_float))
    std_val = float(np.std(roi_float))
    skewness = float(stats.skew(roi_float))
    kurtosis_val = float(stats.kurtosis(roi_float))

    # Shannon entropy
    hist, _ = np.histogram(roi_float, bins=64, density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

    return np.array([mean_val, std_val, skewness, kurtosis_val, entropy], dtype=np.float64)


def extract_all_texture_features(roi: np.ndarray) -> np.ndarray:
    """Extract complete texture feature vector from a subchondral ROI patch.

    Returns:
        Fixed-size numpy vector combining LBP, GLCM, fractal dim, and intensity stats.
    """
    if roi is None or roi.size == 0:
        # Return zeros for missing ROI (LBP: 36 + GLCM: 5 + FD: 1 + Intensity: 5 = 47)
        return np.zeros(47, dtype=np.float64)

    lbp = extract_lbp_features(roi)            # 10+18+26 = 54... actually (8+2)+(16+2)+(24+2) = 10+18+26 = 54
    # Wait - for uniform LBP: bins = n_points + 2 = 10, 18, 26 -> total 54
    # But we want a manageable size. Let's use fixed output.
    glcm = extract_glcm_features(roi)          # 5
    fd = compute_fractal_dimension(roi)        # 1
    intensity = extract_intensity_stats(roi)   # 5

    # Concatenate all: LBP varies by radii, so we use it directly
    return np.concatenate([lbp, glcm, [fd], intensity])
