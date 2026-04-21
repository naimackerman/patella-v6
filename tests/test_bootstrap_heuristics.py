import numpy as np

from src.features.bootstrap_heuristics import (
    build_sclerosis_features,
    estimate_jsn_features,
    heuristic_osteophyte_features,
)


def test_estimate_jsn_features_returns_expected_schema():
    image = np.tile(np.linspace(0, 255, 224, dtype=np.uint8), (224, 1))
    features = estimate_jsn_features(image, is_left=False)

    assert set(features.keys()) == {
        "mJSW_medial",
        "mJSW_lateral",
        "jsw_profile",
        "jsn_rate_medial",
        "jsn_rate_lateral",
        "jsw_ratio",
        "jsw_asymmetry",
    }
    assert features["jsw_profile"].shape == (16,)


def test_heuristic_osteophyte_features_stay_in_valid_range():
    roi = np.full((140, 140), 140, dtype=np.uint8)
    rois = {
        "medial_femur": roi,
        "lateral_femur": roi,
        "medial_tibia": roi,
        "lateral_tibia": roi,
    }
    features = heuristic_osteophyte_features(rois)

    for key in ("osp_grade_mf", "osp_grade_lf", "osp_grade_mt", "osp_grade_lt"):
        assert 0.0 <= features[key] <= 3.0


def test_build_sclerosis_features_returns_18_dim_schema():
    roi = np.full((64, 64), 180, dtype=np.uint8)
    features = build_sclerosis_features(roi, roi)
    assert len(features) == 18
    assert 0 <= features["scl_grade_medial"] <= 2
    assert 0 <= features["scl_grade_lateral"] <= 2
