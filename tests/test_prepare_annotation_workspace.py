from pathlib import Path
from types import SimpleNamespace

from scripts.prepare_annotation_workspace import _package_image_filename


def test_package_image_filename_is_specific_and_stable():
    row = SimpleNamespace(split="train", kl_grade=3, image_id="9001695L")
    src = Path("/tmp/9001695L.png")

    assert _package_image_filename(row, src) == "train_3_9001695L.png"
