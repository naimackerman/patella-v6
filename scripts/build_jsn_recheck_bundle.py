"""Build a dedicated asset bundle for the JSN recheck priority list."""

from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig


def _safe_stem(row: dict) -> str:
    return (
        f"{int(row['priority_rank']):02d}_"
        f"{row['review_priority']}_"
        f"{row['split']}_{row['grade']}_{row['image_id']}"
    )


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: str, dst: Path) -> str:
    if not src:
        return ""
    src_path = Path(src)
    if not src_path.exists():
        return ""
    shutil.copy2(src_path, dst)
    return str(dst)


def _build_contact_sheet(
    image_paths: list[Path],
    title: str,
    columns: int = 4,
    tile_width: int = 220,
    tile_height: int = 220,
    label_height: int = 22,
    header_height: int = 36,
    gutter: int = 8,
) -> np.ndarray:
    rows = max(1, math.ceil(len(image_paths) / columns))
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
            (x0 + 4, y0 + tile_height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    return canvas


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    package_dir = Path(cfg.annotation_dir) / "packages" / "jsn_contours"
    source_csv = Path(str(getattr(cfg, "recheck_csv", package_dir / "jsn_recheck_priority.csv")))
    bundle_root = Path(str(getattr(cfg, "bundle_root", package_dir / "recheck_bundle")))
    page_size = int(getattr(cfg, "page_size", 12))
    columns = int(getattr(cfg, "columns", 4))
    tile_width = int(getattr(cfg, "tile_width", 220))
    tile_height = int(getattr(cfg, "tile_height", 220))

    if not source_csv.exists():
        raise FileNotFoundError(f"Recheck CSV not found: {source_csv}")

    rows = list(csv.DictReader(source_csv.open()))
    if not rows:
        raise ValueError(f"No rows found in {source_csv}")

    _reset_dir(bundle_root)
    annotator_dir = bundle_root / "annotator_images"
    panel_dir = bundle_root / "prediction_panels"
    overlay_dir = bundle_root / "mask_overlays"
    source_dir = bundle_root / "source_images"
    sheets_dir = bundle_root / "contact_sheets"
    for directory in (annotator_dir, panel_dir, overlay_dir, source_dir, sheets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bundle_rows = []
    for row in rows:
        stem = _safe_stem(row)
        annotator_path = _copy_if_exists(row.get("annotator_image_path", ""), annotator_dir / f"{stem}.png")
        panel_path = _copy_if_exists(row.get("prediction_panel_path", ""), panel_dir / f"{stem}.png")
        overlay_path = _copy_if_exists(row.get("mask_overlay_path", ""), overlay_dir / f"{stem}.png")
        source_path = _copy_if_exists(row.get("source_image_path", ""), source_dir / f"{stem}.png")

        updated = dict(row)
        updated["bundle_annotator_image_path"] = annotator_path
        updated["bundle_prediction_panel_path"] = panel_path
        updated["bundle_mask_overlay_path"] = overlay_path
        updated["bundle_source_image_path"] = source_path
        bundle_rows.append(updated)

    bundle_csv = bundle_root / "jsn_recheck_bundle.csv"
    with bundle_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(bundle_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bundle_rows)

    panel_images = sorted(panel_dir.glob("*.png"))
    num_pages = math.ceil(len(panel_images) / page_size)
    for page_idx in range(num_pages):
        start = page_idx * page_size
        end = min(len(panel_images), start + page_size)
        batch = panel_images[start:end]
        sheet = _build_contact_sheet(
            batch,
            title=f"JSN recheck prediction panels | page {page_idx + 1}/{num_pages} | cases {start + 1}-{end}",
            columns=columns,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        cv2.imwrite(str(sheets_dir / f"prediction_panels_page_{page_idx + 1:02d}.png"), sheet)

    readme = bundle_root / "README.md"
    readme.write_text(
        "\n".join([
            "# JSN Recheck Bundle",
            "",
            f"Source CSV: `{source_csv}`",
            "",
            "Contents:",
            "- `jsn_recheck_bundle.csv`: bundle-local CSV with copied asset paths",
            "- `annotator_images/`: preprocessed images to reopen for contour correction",
            "- `prediction_panels/`: model prediction panels for the same cases",
            "- `mask_overlays/`: imported ground-truth mask overlays",
            "- `source_images/`: original test images",
            "- `contact_sheets/`: paginated preview of the copied prediction panels",
            "",
            "File names are prefixed with priority rank and metadata so they sort in review order.",
        ]),
        encoding="utf-8",
    )

    print(f"Saved JSN recheck bundle to {bundle_root}")
    print(f"CSV: {bundle_csv}")
    print(f"Rows bundled: {len(bundle_rows)}")


if __name__ == "__main__":
    main()
