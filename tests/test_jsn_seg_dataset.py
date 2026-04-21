from pathlib import Path

import cv2
import numpy as np

from src.data.jsn_seg_dataset import JSNSegDataset


def test_jsn_seg_dataset_finds_nested_images_and_masks(tmp_path: Path):
    image_dir = tmp_path / "images" / "0"
    mask_dir = tmp_path / "masks" / "0"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.full((16, 16), 128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1

    cv2.imwrite(str(image_dir / "sample.png"), image)
    cv2.imwrite(str(mask_dir / "sample.png"), mask)

    dataset = JSNSegDataset(str(tmp_path / "images"), str(tmp_path / "masks"))
    assert len(dataset) == 1

    image_tensor, mask_tensor, sample_path = dataset[0]
    assert image_tensor.shape == (1, 16, 16)
    assert mask_tensor.shape == (16, 16)
    assert sample_path.endswith("sample.png")
