import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.kl_dataset import KLHybridDataset


class KLHybridDatasetNormalizationTests(unittest.TestCase):
    def test_dataset_applies_saved_feature_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "data" / "train" / "0"
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / "sample.png"
            cv2.imwrite(str(image_path), np.full((16, 16), 128, dtype=np.uint8))

            feature_dir = root / "features" / "aggregated"
            feature_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                feature_dir / "train_features.npz",
                image_ids=np.array(["sample"]),
                features=np.array([[10.0, 30.0]], dtype=np.float32),
                labels=np.array([0], dtype=np.int64),
            )
            np.savez(
                feature_dir / "normalizer_stats.npz",
                mean=np.array([5.0, 10.0], dtype=np.float32),
                std=np.array([5.0, 10.0], dtype=np.float32),
            )

            dataset = KLHybridDataset(
                str(root / "data"),
                "train",
                str(feature_dir / "train_features.npz"),
            )

            _, features, label = dataset[0]
            self.assertEqual(int(label), 0)
            np.testing.assert_allclose(features.numpy(), np.array([1.0, 2.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
