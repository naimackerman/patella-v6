"""Check completeness of extracted osteophyte ROI patches against the dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROI_SITES = ["medial_femur", "lateral_femur", "medial_tibia", "lateral_tibia"]


def scan_dataset(input_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split_dir in sorted(input_dir.iterdir()):
        if not split_dir.is_dir() or split_dir.name not in {"train", "val", "test"}:
            continue
        split = split_dir.name
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            for image_path in sorted(grade_dir.glob("*.png")):
                rows.append({
                    "image_id": image_path.stem,
                    "split": split,
                })
    return rows


def missing_sites(output_dir: Path, split: str, image_id: str) -> list[str]:
    split_dir = output_dir / split
    return [
        site for site in ROI_SITES
        if not (split_dir / f"{image_id}_{site}.png").exists()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=str,
        default="KneeXrayData/ClsKLData/kneeKL224",
        help="Dataset directory with train/val/test/<grade> image folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="features/rois_osteophyte_clahe_full",
        help="ROI output directory containing split subfolders.",
    )
    parser.add_argument(
        "--write-missing-csv",
        type=str,
        default=None,
        help="Optional CSV path to write incomplete image rows.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rows = scan_dataset(input_dir)

    complete = 0
    partial = 0
    missing = 0
    split_summary = {
        "train": {"complete": 0, "partial": 0, "missing": 0},
        "val": {"complete": 0, "partial": 0, "missing": 0},
        "test": {"complete": 0, "partial": 0, "missing": 0},
    }
    incomplete_rows: list[dict[str, str]] = []

    for row in rows:
        image_id = row["image_id"]
        split = row["split"]
        missing_for_image = missing_sites(output_dir, split, image_id)
        if not missing_for_image:
            complete += 1
            split_summary[split]["complete"] += 1
            continue
        if len(missing_for_image) == len(ROI_SITES):
            missing += 1
            split_summary[split]["missing"] += 1
        else:
            partial += 1
            split_summary[split]["partial"] += 1
        incomplete_rows.append({
            "image_id": image_id,
            "split": split,
            "missing_sites": ",".join(missing_for_image),
        })

    if args.write_missing_csv:
        missing_csv_path = Path(args.write_missing_csv)
        missing_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "split", "missing_sites"])
            writer.writeheader()
            writer.writerows(incomplete_rows)

    total_images = len(rows)
    total_roi_files = sum(1 for _ in output_dir.glob("*/*.png"))
    expected_roi_files = total_images * len(ROI_SITES)

    print("Osteophyte ROI Coverage")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Total images: {total_images}")
    print(f"Complete images: {complete}")
    print(f"Partial images: {partial}")
    print(f"Missing images: {missing}")
    print(f"ROI files present: {total_roi_files}/{expected_roi_files}")
    print("")
    for split in ("train", "val", "test"):
        summary = split_summary[split]
        print(
            f"{split}: complete={summary['complete']} "
            f"partial={summary['partial']} missing={summary['missing']}"
        )
    if args.write_missing_csv:
        print("")
        print(f"Missing CSV: {args.write_missing_csv}")


if __name__ == "__main__":
    main()
