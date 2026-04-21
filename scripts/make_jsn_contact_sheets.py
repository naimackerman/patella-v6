"""Create paginated contact sheets for JSN overlay or mask QA."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig


def _build_contact_sheet(
    image_paths: list[Path],
    title: str,
    columns: int = 5,
    tile_width: int = 180,
    tile_height: int = 180,
    label_height: int = 24,
    header_height: int = 36,
    gutter: int = 8,
) -> np.ndarray:
    rows = math.ceil(len(image_paths) / columns)
    canvas_h = header_height + gutter + rows * (tile_height + label_height + gutter) + gutter
    canvas_w = columns * (tile_width + gutter) + gutter
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    cv2.putText(
        canvas,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    for idx, image_path in enumerate(image_paths):
        row = idx // columns
        col = idx % columns
        x0 = gutter + col * (tile_width + gutter)
        y0 = header_height + gutter + row * (tile_height + label_height + gutter)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            tile = np.full((tile_height, tile_width, 3), 220, dtype=np.uint8)
            cv2.putText(tile, "load_failed", (12, tile_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1, cv2.LINE_AA)
        else:
            tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

        canvas[y0:y0 + tile_height, x0:x0 + tile_width] = tile
        cv2.rectangle(canvas, (x0, y0), (x0 + tile_width, y0 + tile_height), (180, 180, 180), 1)
        cv2.putText(
            canvas,
            image_path.stem,
            (x0 + 4, y0 + tile_height + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    return canvas


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    annotation_dir = Path(cfg.annotation_dir)
    source_name = str(getattr(cfg, "source_name", "jsn_mask_overlays"))
    source_root_override = getattr(cfg, "source_root", None)
    output_root_override = getattr(cfg, "output_root", None)
    columns = int(getattr(cfg, "columns", 5))
    page_size = int(getattr(cfg, "page_size", 25))
    tile_width = int(getattr(cfg, "tile_width", 180))
    tile_height = int(getattr(cfg, "tile_height", 180))

    source_root = Path(str(source_root_override)) if source_root_override else annotation_dir / source_name
    output_root = Path(str(output_root_override)) if output_root_override else annotation_dir / f"{source_name}_contact_sheets"
    output_root.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    for split in ("train", "val", "test"):
        split_dir = source_root / split
        if not split_dir.exists():
            continue

        image_paths = sorted(split_dir.glob("*.png"))
        if not image_paths:
            continue

        split_out = output_root / split
        split_out.mkdir(parents=True, exist_ok=True)

        num_pages = math.ceil(len(image_paths) / page_size)
        for page_idx in range(num_pages):
            start = page_idx * page_size
            end = min(len(image_paths), start + page_size)
            batch = image_paths[start:end]
            title = f"{source_name} | {split} | page {page_idx + 1}/{num_pages} | images {start + 1}-{end}"
            sheet = _build_contact_sheet(
                batch,
                title=title,
                columns=columns,
                tile_width=tile_width,
                tile_height=tile_height,
            )
            out_path = split_out / f"{split}_page_{page_idx + 1:02d}.png"
            cv2.imwrite(str(out_path), sheet)
            total_pages += 1

    print(f"Saved contact sheets to {output_root}")
    print(f"Generated {total_pages} page(s)")


if __name__ == "__main__":
    main()
