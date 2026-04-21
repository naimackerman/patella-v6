import unittest
from unittest.mock import patch

import numpy as np

from src.features.kneel_landmarks import (
    CompartmentLandmarks,
    KNEELLandmarkDetector,
    KneeLandmarks,
    compute_kneel_roi_boxes,
    _surface_y_in_band,
    detect_knee_landmarks,
    landmarks_from_jsn_mask,
)
from src.models.roi_detector import ROIDetector


class TestKNEELLandmarks(unittest.TestCase):
    def test_detect_knee_landmarks_marks_blank_image_low_confidence(self):
        image = np.zeros((224, 224), dtype=np.uint8)
        landmarks = detect_knee_landmarks(image, is_left=False, apply_preprocessing=False)

        self.assertTrue(landmarks.low_confidence)
        self.assertTrue(landmarks.lateral.synthetic)
        self.assertTrue(landmarks.medial.synthetic)

    def test_landmark_detector_require_reliable_raises_for_synthetic_landmarks(self):
        image = np.zeros((224, 224), dtype=np.uint8)
        detector = KNEELLandmarkDetector(backend="heuristic")

        with self.assertRaisesRegex(ValueError, "low_confidence"):
            detector.predict(image, is_left=False, apply_preprocessing=False, require_reliable=True)

    def test_landmark_boxes_require_reliable_propagates_failure(self):
        image = np.zeros((224, 224), dtype=np.uint8)
        detector = KNEELLandmarkDetector(backend="heuristic")

        with self.assertRaisesRegex(ValueError, "low_confidence"):
            ROIDetector.landmark_boxes(
                image,
                is_left=False,
                landmark_detector=detector,
                apply_preprocessing=False,
                require_reliable=True,
            )

    def test_predict_forwards_apply_preprocessing_flag(self):
        detector = KNEELLandmarkDetector(backend="heuristic")
        fake_landmarks = KneeLandmarks(
            joint_y=100.0,
            lateral=CompartmentLandmarks((0, 10), np.zeros((2, 2)), np.zeros((2, 2))),
            medial=CompartmentLandmarks((11, 20), np.zeros((2, 2)), np.zeros((2, 2))),
        )

        with patch("src.features.kneel_landmarks.detect_knee_landmarks", return_value=fake_landmarks) as mocked:
            result = detector.predict(np.zeros((16, 16), dtype=np.uint8), apply_preprocessing=False)

        self.assertIs(result, fake_landmarks)
        _, kwargs = mocked.call_args
        self.assertFalse(kwargs["apply_preprocessing"])

    def test_surface_y_in_band_uses_band_median_not_single_edge_outlier(self):
        surface = np.array([
            [0, 0],
            [2, 50],
            [4, 52],
            [6, 51],
            [8, 49],
            [10, 50],
        ], dtype=np.float64)

        y_value = _surface_y_in_band(surface, (0, 8))
        self.assertGreaterEqual(y_value, 45.0)
        self.assertLessEqual(y_value, 55.0)

    def test_landmarks_from_jsn_mask_uses_anatomical_mask_classes(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:16, 3:16] = 1
        mask[11:17, 17:30] = 2

        landmarks = landmarks_from_jsn_mask(mask)

        self.assertFalse(landmarks.low_confidence)
        self.assertLess(landmarks.medial.x_range[0], landmarks.lateral.x_range[0])
        self.assertEqual(landmarks.backend, "jsn_mask")

    def test_site_specific_boxes_expand_lateral_femur_and_medial_tibia(self):
        femoral = np.array([[10, 30], [20, 30], [30, 31]], dtype=np.float64)
        tibial = np.array([[10, 40], [20, 41], [30, 41]], dtype=np.float64)
        lateral = CompartmentLandmarks((10, 30), femoral, tibial)

        femoral_m = np.array([[70, 30], [80, 30], [90, 31]], dtype=np.float64)
        tibial_m = np.array([[70, 40], [80, 41], [90, 41]], dtype=np.float64)
        medial = CompartmentLandmarks((70, 90), femoral_m, tibial_m)

        landmarks = KneeLandmarks(
            joint_y=35.0,
            lateral=lateral,
            medial=medial,
            backend="jsn_mask",
        )
        boxes = compute_kneel_roi_boxes((100, 100), landmarks)

        lat_fem = boxes["lateral_femur"]
        lat_tib = boxes["lateral_tibia"]
        med_fem = boxes["medial_femur"]
        med_tib = boxes["medial_tibia"]

        self.assertGreater(lat_fem[2] - lat_fem[0], lat_tib[2] - lat_tib[0])
        self.assertGreater(med_tib[2] - med_tib[0], med_fem[2] - med_fem[0])
        self.assertLessEqual(lat_fem[0], lat_tib[0])
        self.assertGreaterEqual(med_tib[2], med_fem[2])


if __name__ == "__main__":
    unittest.main()
