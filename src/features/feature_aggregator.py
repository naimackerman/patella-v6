"""Feature aggregation: combine JSN + Osteophyte + Sclerosis into 50-dim vector."""

from typing import Dict, Optional, Tuple

import numpy as np


class FeatureAggregator:
    """Combines features from all three quantification modules into a unified vector.

    Feature composition (50 dimensions):
    - JSN: 22 dims (mJSW_med, mJSW_lat, jsw_profile[16], jsn_rate_med, jsn_rate_lat, ratio, asymmetry)
    - Osteophyte: 10 dims (4 grades, sum, max, medial_sum, lateral_sum, femoral_sum, tibial_sum)
    - Sclerosis: 18 dims (2 grades, 2 intensities, 2 fractal dims, 5 GLCM med, 5 GLCM lat, 2 LBP entropy)
    """

    JSN_DIM = 22
    OSP_DIM = 10
    SCL_DIM = 18
    TOTAL_DIM = JSN_DIM + OSP_DIM + SCL_DIM  # 50

    def __init__(self):
        self.train_mean = None
        self.train_std = None

    def jsn_to_vector(self, jsn: Dict) -> np.ndarray:
        """Convert JSN feature dict to 22-dim vector."""
        vec = [
            jsn.get("mJSW_medial", 0.0),
            jsn.get("mJSW_lateral", 0.0),
        ]
        profile = jsn.get("jsw_profile", np.zeros(16))
        vec.extend(profile.tolist() if hasattr(profile, "tolist") else list(profile))
        vec.extend([
            jsn.get("jsn_rate_medial", 0.0),
            jsn.get("jsn_rate_lateral", 0.0),
            jsn.get("jsw_ratio", 0.0),
            jsn.get("jsw_asymmetry", 0.0),
        ])
        return np.array(vec, dtype=np.float64)

    def osteophyte_to_vector(self, osp: Dict) -> np.ndarray:
        """Convert osteophyte feature dict to 10-dim vector."""
        return np.array([
            osp.get("osp_grade_mf", 0),
            osp.get("osp_grade_lf", 0),
            osp.get("osp_grade_mt", 0),
            osp.get("osp_grade_lt", 0),
            osp.get("osp_sum", 0),
            osp.get("osp_max", 0),
            osp.get("osp_medial_sum", 0),
            osp.get("osp_lateral_sum", 0),
            osp.get("osp_femoral_sum", 0),
            osp.get("osp_tibial_sum", 0),
        ], dtype=np.float64)

    def sclerosis_to_vector(self, scl: Dict) -> np.ndarray:
        """Convert sclerosis feature dict to 18-dim vector."""
        return np.array([
            scl.get("scl_grade_medial", 0),
            scl.get("scl_grade_lateral", 0),
            scl.get("scl_intensity_medial", 0.0),
            scl.get("scl_intensity_lateral", 0.0),
            scl.get("scl_fractal_dim_med", 1.0),
            scl.get("scl_fractal_dim_lat", 1.0),
            scl.get("scl_glcm_contrast_med", 0.0),
            scl.get("scl_glcm_dissimilarity_med", 0.0),
            scl.get("scl_glcm_homogeneity_med", 0.0),
            scl.get("scl_glcm_energy_med", 0.0),
            scl.get("scl_glcm_correlation_med", 0.0),
            scl.get("scl_glcm_contrast_lat", 0.0),
            scl.get("scl_glcm_dissimilarity_lat", 0.0),
            scl.get("scl_glcm_homogeneity_lat", 0.0),
            scl.get("scl_glcm_energy_lat", 0.0),
            scl.get("scl_glcm_correlation_lat", 0.0),
            scl.get("scl_lbp_entropy_med", 0.0),
            scl.get("scl_lbp_entropy_lat", 0.0),
        ], dtype=np.float64)

    def aggregate(self, jsn: Dict, osp: Dict, scl: Dict) -> np.ndarray:
        """Combine all features into a single 50-dim vector."""
        jsn_vec = self.jsn_to_vector(jsn)
        osp_vec = self.osteophyte_to_vector(osp)
        scl_vec = self.sclerosis_to_vector(scl)
        return np.concatenate([jsn_vec, osp_vec, scl_vec])

    def fit_normalizer(self, feature_matrix: np.ndarray):
        """Compute z-score normalization statistics from training data.

        Args:
            feature_matrix: (N, 50) training feature matrix.
        """
        self.train_mean = feature_matrix.mean(axis=0)
        self.train_std = feature_matrix.std(axis=0)
        self.train_std[self.train_std < 1e-8] = 1.0

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using training statistics."""
        if self.train_mean is None:
            return features
        return (features - self.train_mean) / self.train_std

    def get_feature_names(self) -> list:
        """Return ordered list of feature names."""
        names = [
            "mJSW_medial", "mJSW_lateral",
        ]
        names.extend([f"jsw_profile_{i}" for i in range(16)])
        names.extend([
            "jsn_rate_medial", "jsn_rate_lateral", "jsw_ratio", "jsw_asymmetry",
            "osp_grade_mf", "osp_grade_lf", "osp_grade_mt", "osp_grade_lt",
            "osp_sum", "osp_max", "osp_medial_sum", "osp_lateral_sum",
            "osp_femoral_sum", "osp_tibial_sum",
            "scl_grade_medial", "scl_grade_lateral",
            "scl_intensity_medial", "scl_intensity_lateral",
            "scl_fractal_dim_med", "scl_fractal_dim_lat",
            "scl_glcm_contrast_med", "scl_glcm_dissimilarity_med",
            "scl_glcm_homogeneity_med", "scl_glcm_energy_med",
            "scl_glcm_correlation_med",
            "scl_glcm_contrast_lat", "scl_glcm_dissimilarity_lat",
            "scl_glcm_homogeneity_lat", "scl_glcm_energy_lat",
            "scl_glcm_correlation_lat",
            "scl_lbp_entropy_med", "scl_lbp_entropy_lat",
        ])
        return names
