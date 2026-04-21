import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.extract_roi_with_fullimage_clahe import (
    DEFAULT_OSTEOPHYTE_ROI_SIZE,
    ROI_SITES,
    extract_geometric_rois,
    image_has_all_roi_outputs,
)


class TestExtractRoiWithFullimageClahe(unittest.TestCase):
    def test_geometric_fallback_uses_research_roi_size_by_default(self):
        image = np.full((224, 224), 128, dtype=np.uint8)

        rois = extract_geometric_rois(image, is_left=True)

        self.assertEqual(set(rois), set(ROI_SITES))
        for roi in rois.values():
            self.assertEqual(roi.shape, (DEFAULT_OSTEOPHYTE_ROI_SIZE, DEFAULT_OSTEOPHYTE_ROI_SIZE))

    def test_skip_existing_rejects_old_224_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            split_dir = output_dir / "train"
            split_dir.mkdir(parents=True)
            for site in ROI_SITES:
                cv2.imwrite(str(split_dir / f"sample_{site}.png"), np.zeros((224, 224), dtype=np.uint8))

            self.assertFalse(
                image_has_all_roi_outputs(
                    output_dir,
                    split="train",
                    image_id="sample",
                    roi_size=DEFAULT_OSTEOPHYTE_ROI_SIZE,
                )
            )


if __name__ == "__main__":
    unittest.main()
