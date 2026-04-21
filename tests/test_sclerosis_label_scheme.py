import unittest

import numpy as np
from omegaconf import OmegaConf

from src.utils.sclerosis_labels import (
    apply_sclerosis_label_scheme_to_cfg,
    map_sclerosis_grades,
    sclerosis_class_names,
)


class TestSclerosisLabelScheme(unittest.TestCase):
    def test_severity_keeps_three_classes(self):
        grades = np.asarray([0, 1, 2, 0])

        mapped = map_sclerosis_grades(grades, "severity")

        np.testing.assert_array_equal(mapped, grades)
        self.assertEqual(sclerosis_class_names("severity"), ["none", "mild", "significant"])

    def test_binary_present_collapses_mild_and_significant(self):
        grades = np.asarray([0, 1, 2, 0, 2])

        mapped = map_sclerosis_grades(grades, "binary_present")

        np.testing.assert_array_equal(mapped, np.asarray([0, 1, 1, 0, 1]))
        self.assertEqual(sclerosis_class_names("binary"), ["none", "present"])

    def test_cfg_num_classes_matches_label_scheme(self):
        cfg = OmegaConf.create({
            "training": {"sclerosis_label_scheme": "severity"},
            "model": {"num_classes": 3, "class_names": ["none", "mild", "significant"]},
        })

        cfg = apply_sclerosis_label_scheme_to_cfg(cfg, "binary_present")

        self.assertEqual(cfg.training.sclerosis_label_scheme, "binary_present")
        self.assertEqual(cfg.model.num_classes, 2)
        self.assertEqual(list(cfg.model.class_names), ["none", "present"])


if __name__ == "__main__":
    unittest.main()
