import unittest

import numpy as np

from src.features.subchondral_roi import (
    extract_subchondral_roi_with_boxes,
    extract_subchondral_roi_with_boxes_and_source,
)


class TestSubchondralROI(unittest.TestCase):
    def test_extract_subchondral_roi_uses_mask_labels_before_midline_split(self):
        image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        mask = np.zeros((64, 64), dtype=np.uint8)
        # Put the medial component on the left half to simulate an off-center prediction
        # where a naive midline split would fail for a right-knee medial compartment.
        mask[20:25, 6:24] = 1
        mask[20:25, 40:58] = 2

        medial_roi, lateral_roi, medial_box, lateral_box, source = extract_subchondral_roi_with_boxes_and_source(
            mask,
            image,
            is_left=False,
            output_size=32,
        )

        self.assertEqual(source, "jsn_guided")
        self.assertIsNotNone(medial_roi)
        self.assertIsNotNone(lateral_roi)
        self.assertIsNotNone(medial_box)
        self.assertIsNotNone(lateral_box)
        self.assertLess(medial_box[2], 32)
        self.assertGreater(lateral_box[0], 32)

    def test_extract_subchondral_roi_supports_per_side_offsets(self):
        image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:25, 4:30] = 2   # lateral compartment
        mask[20:25, 34:60] = 1  # medial compartment

        _, _, medial_box_default, lateral_box_default = extract_subchondral_roi_with_boxes(
            mask,
            image,
            is_left=False,
            output_size=32,
        )
        _, _, medial_box_tuned, lateral_box_tuned = extract_subchondral_roi_with_boxes(
            mask,
            image,
            is_left=False,
            medial_inner_offset_pct=0.0,
            medial_outer_offset_pct=0.20,
            lateral_inner_offset_pct=0.0,
            lateral_outer_offset_pct=0.20,
            output_size=32,
        )

        self.assertIsNotNone(medial_box_default)
        self.assertIsNotNone(lateral_box_default)
        self.assertIsNotNone(medial_box_tuned)
        self.assertIsNotNone(lateral_box_tuned)

        # Right knee: medial outer edge is on the right, lateral outer edge on the left.
        self.assertLess(medial_box_tuned[2], medial_box_default[2])
        self.assertGreater(lateral_box_tuned[0], lateral_box_default[0])


if __name__ == "__main__":
    unittest.main()
